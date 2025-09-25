from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from langchain_core.embeddings import Embeddings

from ..config import Settings
from ..utils.logging import get_logger
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
    private_embedding: List[int] = field(default_factory=list)
    submission_id: str | None = None
    created_at: datetime

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


@dataclass
class IngestionResult:
    chunks: List[ChunkRow]
    documents: List[IngestedDocument]


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
            embedding_scope=embedding_scope,
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
        embedding_scope: str = "both",
        submission_id: str | None = None,
    ) -> IngestionResult:
        documents = load_documents(file_paths)
        if not documents:
            logger.warning("No documents parsed from inputs")
            return IngestionResult(chunks=[], documents=[])

        chunks = self.chunker.split_documents(documents)
        if not chunks:
            logger.warning("No chunks generated; check splitter configuration")
            return IngestionResult(chunks=[], documents=[])

        submission_id = submission_id or uuid.uuid4().hex
        dense_embeddings, private_embeddings = self._generate_embeddings(chunks)
        rows: List[ChunkRow] = []

        doc_id_map: dict[str, str] = {}
        doc_counter: dict[str, int] = {}
        upload_cache: dict[str, tuple[str, str]] = {}
        summaries: dict[str, IngestedDocument] = {}
        private_embedding_cache: dict[str, list[tuple[int, List[int]]]] = {}
        category_list = sorted({c.strip() for c in categories or [] if c and c.strip()})
        selected_model = model_alias or self.settings.default_agent

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
                base_metadata = extra_metadata.copy() if extra_metadata else {}
                base_metadata.update(
                    {
                        "submission_id": submission_id,
                        "embedding_scope": embedding_scope,
                        "categories": category_list,
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
                    embedding_scope=embedding_scope,
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
                "embedding_scope": embedding_scope,
            })
            if extra_metadata:
                metadata.setdefault("ingestion_metadata", extra_metadata)

            private_embedding_cache[source_key].append((chunk_index, list(map(int, private_embedding))))

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
                    private_embedding=list(map(int, private_embedding)),
                    submission_id=submission_id,
                    created_at=datetime.utcnow(),
                )
            )

        self.tigergraph_client.upsert_chunk_rows(rows)
        self._persist_private_embeddings(private_embedding_cache, doc_id_map, submission_id, summaries)
        logger.info("Persisted %s chunks to TigerGraph", len(rows))
        return IngestionResult(chunks=rows, documents=list(summaries.values()))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _generate_embeddings(self, chunks: list["Document"]) -> Tuple[List[List[float]], List[List[int]]]:
        texts = [chunk.page_content for chunk in chunks]
        if hasattr(self.embeddings, "embed_documents_with_private"):
            return self.embeddings.embed_documents_with_private(texts)  # type: ignore[return-value]
        dense = self.embeddings.embed_documents(texts)
        private = [[1 if value >= 0 else 0 for value in vector] for vector in dense]
        return dense, private

    def _persist_private_embeddings(
        self,
        private_embedding_cache: dict[str, list[tuple[int, List[int]]]],
        doc_id_map: dict[str, str],
        submission_id: str,
        summaries: dict[str, IngestedDocument],
    ) -> None:
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


class ObjectStore:
    """Minimal interface for object storage backends."""

    def upload(self, path: Path, key: str) -> tuple[str, str]:  # pragma: no cover - interface definition
        raise NotImplementedError

    def upload_json(self, payload: dict, key: str) -> tuple[str, str]:  # pragma: no cover - interface definition
        raise NotImplementedError


class TigerGraphClient:  # pragma: no cover - circular import guard
    def upsert_chunk_rows(self, rows: Sequence[ChunkRow]) -> None: ...
