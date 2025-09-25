from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import hashlib
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from langchain_core.embeddings import Embeddings

from ..config import Settings
from ..utils.importance import DocumentImportanceScorer
from ..utils.logging import get_logger
from ..utils.subjects import SubjectClassifier
from ..utils.text import resolve_document_id
from .chunking import AdaptiveChunker
from .loaders import SUPPORTED_EXTENSIONS, load_documents

logger = get_logger(__name__)


@dataclass
class ChunkRow:
    """Representation of the payload we persist in TigerGraph."""

    id: str
    doc_id: str
    chunk_id: int
    title: str
    source: str
    uri: str
    http_url: str
    content: str
    metadata: dict
    embedding: List[float]
    created_at: datetime
    private_embedding: List[int] = field(default_factory=list)
    submission_id: str | None = None

    def to_upsert_payload(self) -> dict:
        return {
            "attributes": {
                "doc_id": self.doc_id,
                "chunk_id": self.chunk_id,
                "title": self.title,
                "source": self.source,
                "uri": self.uri,
                "http_url": self.http_url,
                "content": self.content,
                "metadata": json.dumps(self.metadata),
                "embedding": self.embedding,
                "private_embedding": self.private_embedding,
                "submission_id": self.submission_id,
                "created_at": self.created_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
        }


@dataclass
class IngestedDocument:
    doc_id: str
    owner_id: str | None
    categories: list[str]
    model_alias: str | None
    source_path: Path
    uri: str
    http_url: str
    metadata: dict
    submission_id: str
    private_embedding_uri: str | None
    embedding_scope: str
    file_size_bytes: int
    source_checksum: str


@dataclass
class IngestionResult:
    chunks: List[ChunkRow]
    documents: List[IngestedDocument]


class EmbeddingScope(str, Enum):
    """Permitted visibility modes for generated embeddings."""

    PUBLIC = "public"
    PRIVATE = "private"
    BOTH = "both"


class DocumentIngestionPipeline:
    """Coordinates parsing, embedding and persistence for raw documents."""

    def __init__(
        self,
        settings: Settings,
        embeddings: Embeddings,
        tigergraph_client: "TigerGraphClient",
        object_store: "ObjectStore",
    ) -> None:
        self.settings = settings
        self.embeddings = embeddings
        self.tigergraph_client = tigergraph_client
        self.object_store = object_store
        self.chunker = AdaptiveChunker(settings)
        self.subject_classifier = SubjectClassifier()
        self.importance_scorer = DocumentImportanceScorer()

    def ingest_directory(
        self,
        directory: Path,
        *,
        owner_id: str | None = None,
        categories: Iterable[str] | None = None,
        model_alias: str | None = None,
        extra_metadata: dict | None = None,
        embedding_scope: str = "both",
        submission_id: str | None = None,
    ) -> IngestionResult:
        scope = self._normalise_embedding_scope(embedding_scope)
        file_paths = [p for p in directory.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS]
        if not file_paths:
            logger.warning("No supported files discovered in %s", directory)
            return IngestionResult(chunks=[], documents=[])
        return self.ingest_files(
            file_paths,
            owner_id=owner_id,
            categories=categories,
            model_alias=model_alias,
            extra_metadata=extra_metadata,
            embedding_scope=scope,
            submission_id=submission_id,
        )

    def ingest_files(
        self,
        file_paths: Iterable[Path],
        *,
        owner_id: str | None = None,
        categories: Iterable[str] | None = None,
        model_alias: str | None = None,
        extra_metadata: dict | None = None,
        embedding_scope: str | EmbeddingScope = "both",
        submission_id: str | None = None,
    ) -> IngestionResult:
        scope = self._normalise_embedding_scope(embedding_scope)
        documents = load_documents(file_paths)
        if not documents:
            logger.warning("No documents parsed from inputs")
            return IngestionResult(chunks=[], documents=[])

        chunks = self.chunker.split_documents(documents)
        if not chunks:
            logger.warning("No chunks generated; check splitter configuration")
            return IngestionResult(chunks=[], documents=[])

        submission_id = submission_id or uuid.uuid4().hex
        dense_embeddings, private_embeddings = self._generate_embeddings(chunks, scope)
        rows: List[ChunkRow] = []

        doc_id_map: dict[str, str] = {}
        doc_counter: dict[str, int] = {}
        upload_cache: dict[str, tuple[str, str]] = {}
        summaries: dict[str, IngestedDocument] = {}
        private_embedding_cache: dict[str, list[tuple[int, List[int]]]] = {}
        category_list = sorted({c.strip() for c in categories or [] if c and c.strip()})
        selected_model = model_alias or self.settings.default_agent
        subject_cache: dict[str, set[str]] = {}
        chunk_subject_cache: dict[tuple[str, int], set[str]] = {}

        for chunk, embedding, private_embedding in zip(chunks, dense_embeddings, private_embeddings):
            source_path = Path(chunk.metadata.get("source", "unknown"))
            source_key = source_path.as_posix()
            if source_key not in doc_id_map:
                doc_id_map[source_key] = resolve_document_id(
                    source_path,
                    doc_id_map.values(),
                    owner=str(owner_id) if owner_id is not None else None,
                )
                doc_counter[source_key] = 0

            chunk_index = doc_counter[source_key]
            doc_counter[source_key] += 1

            doc_id = doc_id_map[source_key]
            chunk_id = chunk_index
            row_id = f"{doc_id}::{chunk_id}::{uuid.uuid4().hex[:8]}"

            if source_key not in upload_cache:
                object_key = f"{doc_id}/{source_path.name}"
                upload_cache[source_key] = self.object_store.upload(source_path, object_key)
                uri, http_url = upload_cache[source_key]
                file_size = source_path.stat().st_size if source_path.exists() else 0
                checksum = self._compute_checksum(source_path)
                base_metadata = extra_metadata.copy() if extra_metadata else {}
                base_metadata.update(
                    {
                        "submission_id": submission_id,
                        "embedding_scope": scope.value,
                        "categories": category_list,
                        "source_file_size": file_size,
                        "source_checksum": checksum,
                    }
                )
                summaries[source_key] = IngestedDocument(
                    doc_id=doc_id,
                    owner_id=str(owner_id) if owner_id is not None else None,
                    categories=category_list,
                    model_alias=selected_model,
                    source_path=source_path,
                    uri=uri,
                    http_url=http_url,
                    metadata=base_metadata,
                    submission_id=submission_id,
                    private_embedding_uri=None,
                    embedding_scope=scope.value,
                    file_size_bytes=file_size,
                    source_checksum=checksum,
                )
                private_embedding_cache[source_key] = []
            uri, http_url = upload_cache[source_key]

            metadata = dict(chunk.metadata)
            metadata.update({
                "doc_id": doc_id,
                "chunk_index": chunk_index,
                "source_file": source_path.name,
                "owner_id": str(owner_id) if owner_id is not None else None,
                "categories": category_list,
                "model_alias": selected_model,
                "submission_id": submission_id,
                "embedding_scope": scope.value,
                "source_file_size": summaries[source_key].file_size_bytes,
                "source_checksum": summaries[source_key].source_checksum,
            })
            if extra_metadata:
                metadata.setdefault("ingestion_metadata", extra_metadata)

            subjects = self.subject_classifier.classify(chunk.page_content, metadata.get("subject_tags"))
            metadata["subject_tags"] = subjects
            subject_cache.setdefault(source_key, set()).update(subjects)
            chunk_subject_cache[(doc_id, chunk_id)] = set(subjects)

            if scope is not EmbeddingScope.PUBLIC:
                private_vector = list(map(int, private_embedding))
                private_embedding_cache[source_key].append((chunk_index, private_vector))
            else:
                private_vector = []

            rows.append(
                ChunkRow(
                    id=row_id,
                    doc_id=doc_id,
                    chunk_id=chunk_id,
                    title=metadata.get("title") or source_path.name,
                    source=str(source_path),
                    uri=uri,
                    http_url=http_url,
                    content=chunk.page_content,
                    metadata=metadata,
                    embedding=list(map(float, embedding)),
                    private_embedding=list(map(int, private_embedding)) if scope is not EmbeddingScope.PUBLIC else [],
                    submission_id=submission_id,
                    created_at=datetime.now(timezone.utc),
                )
            )

        doc_importance: dict[str, float] = {}
        for source_key, summary in summaries.items():
            subjects = sorted(subject_cache.get(source_key, set()))
            summary.metadata.setdefault("subjects", {})
            subjects_block = summary.metadata["subjects"]  # type: ignore[assignment]
            if isinstance(subjects_block, dict):
                subjects_block.update(
                    {
                        "tags": subjects,
                        "collections": self.subject_classifier.build_collections(subjects),
                    }
                )
            summary.metadata["subject_tags"] = subjects
            summary.metadata["subject_collections"] = self.subject_classifier.build_collections(subjects)
            doc_score = self.importance_scorer.score_document(
                file_size_bytes=summary.file_size_bytes,
                categories=summary.categories,
                metadata=summary.metadata,
                subject_tags=subjects,
            )
            summary.metadata["importance_score"] = doc_score
            summary.metadata["subject_weights"] = self.importance_scorer.rank_subjects(subjects, doc_score)
            doc_importance[summary.doc_id] = doc_score

        for row in rows:
            subjects = sorted(chunk_subject_cache.get((row.doc_id, row.chunk_id), set()))
            doc_score = doc_importance.get(row.doc_id, self.importance_scorer.base_score)
            chunk_score = self.importance_scorer.score_chunk(
                chunk_length=len(row.content),
                document_score=doc_score,
                subject_tags=subjects,
            )
            row.metadata["subject_tags"] = subjects
            row.metadata["importance_score"] = chunk_score
            row.metadata["doc_importance_score"] = doc_score

        self.tigergraph_client.upsert_chunk_rows(rows)
        self._persist_private_embeddings(private_embedding_cache, doc_id_map, submission_id, summaries, scope)
        logger.info("Persisted %s chunks to TigerGraph", len(rows))
        return IngestionResult(chunks=rows, documents=list(summaries.values()))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _generate_embeddings(
        self,
        chunks: list["Document"],
        scope: EmbeddingScope,
    ) -> Tuple[List[List[float]], List[List[int]]]:
        texts = [chunk.page_content for chunk in chunks]
        dense_vectors: List[List[float]]
        private_vectors: List[List[int]]
        if hasattr(self.embeddings, "embed_documents_with_private"):
            dense, private = self.embeddings.embed_documents_with_private(texts)  # type: ignore[attr-defined]
            dense_vectors = [list(map(float, vector)) for vector in dense]
            private_vectors = [list(map(int, vector)) for vector in private]
        else:
            dense_vectors = [list(map(float, vector)) for vector in self.embeddings.embed_documents(texts)]
            threshold = self.settings.bitwise_threshold
            private_vectors = [
                [1 if value >= threshold else 0 for value in vector]
                for vector in dense_vectors
            ]

        if len(dense_vectors) != len(private_vectors):
            raise ValueError("Embedding provider returned mismatched vector counts")

        if scope is EmbeddingScope.PUBLIC:
            private_vectors = [[] for _ in dense_vectors]

        return dense_vectors, private_vectors

    def _persist_private_embeddings(
        self,
        private_embedding_cache: dict[str, list[tuple[int, List[int]]]],
        doc_id_map: dict[str, str],
        submission_id: str,
        summaries: dict[str, IngestedDocument],
        scope: EmbeddingScope,
    ) -> None:
        if scope is EmbeddingScope.PUBLIC:
            return
        for source_key, embeddings in private_embedding_cache.items():
            if not embeddings:
                continue
            doc_id = doc_id_map[source_key]
            payload = {
                "doc_id": doc_id,
                "submission_id": submission_id,
                "chunks": [
                    {"chunk_index": index, "embedding": vector}
                    for index, vector in sorted(embeddings, key=lambda item: item[0])
                ],
            }
            object_key = f"{doc_id}/private_embeddings/{submission_id}.json"
            uri, http_url = self.object_store.upload_json(payload, object_key)
            summaries[source_key].metadata.setdefault("security", {})
            summaries[source_key].metadata["security"].update(
                {
                    "private_embedding_uri": uri,
                    "private_embedding_http_url": http_url,
                }
            )
            summaries[source_key].private_embedding_uri = uri

    def _normalise_embedding_scope(self, scope: str | EmbeddingScope) -> EmbeddingScope:
        if isinstance(scope, EmbeddingScope):
            return scope
        value = scope.lower().strip()
        try:
            return EmbeddingScope(value)
        except ValueError as exc:
            raise ValueError(
                f"Unsupported embedding scope '{scope}'. Expected one of: {', '.join(s.value for s in EmbeddingScope)}"
            ) from exc

    @staticmethod
    def _compute_checksum(path: Path) -> str:
        if not path.exists() or not path.is_file():
            return ""
        hasher = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()


class ObjectStore:
    """Minimal interface for object storage backends."""

    def upload(self, path: Path, key: str) -> tuple[str, str]:  # pragma: no cover - interface definition
        raise NotImplementedError

    def upload_json(self, payload: dict, key: str) -> tuple[str, str]:  # pragma: no cover - interface definition
        raise NotImplementedError


class TigerGraphClient:  # pragma: no cover - circular import guard
    def upsert_chunk_rows(self, rows: Sequence[ChunkRow]) -> None: ...
