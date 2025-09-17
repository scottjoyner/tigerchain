from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Sequence

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.embeddings import Embeddings

from ..config import Settings
from ..utils.logging import get_logger
from ..utils.text import resolve_document_id
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
                "created_at": self.created_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
        }


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
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

    def ingest_directory(self, directory: Path) -> List[ChunkRow]:
        file_paths = [p for p in directory.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS]
        if not file_paths:
            logger.warning("No supported files discovered in %s", directory)
            return []
        return self.ingest_files(file_paths)

    def ingest_files(self, file_paths: Iterable[Path]) -> List[ChunkRow]:
        documents = load_documents(file_paths)
        if not documents:
            logger.warning("No documents parsed from inputs")
            return []

        chunks = self.splitter.split_documents(documents)
        if not chunks:
            logger.warning("No chunks generated; check splitter configuration")
            return []

        embeddings = self.embeddings.embed_documents([chunk.page_content for chunk in chunks])
        rows: List[ChunkRow] = []

        doc_id_map: dict[str, str] = {}
        doc_counter: dict[str, int] = {}
        upload_cache: dict[str, tuple[str, str]] = {}

        for chunk, embedding in zip(chunks, embeddings):
            source_path = Path(chunk.metadata.get("source", "unknown"))
            source_key = source_path.as_posix()
            if source_key not in doc_id_map:
                doc_id_map[source_key] = resolve_document_id(source_path, doc_id_map.values())
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

            metadata = dict(chunk.metadata)
            metadata.update({
                "doc_id": doc_id,
                "chunk_index": chunk_index,
                "source_file": source_path.name,
            })

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
                    created_at=datetime.utcnow(),
                )
            )

        self.tigergraph_client.upsert_chunk_rows(rows)
        logger.info("Persisted %s chunks to TigerGraph", len(rows))
        return rows


class ObjectStore:
    """Minimal interface for object storage backends."""

    def upload(self, path: Path, key: str) -> tuple[str, str]:  # pragma: no cover - interface definition
        raise NotImplementedError


class TigerGraphClient:  # pragma: no cover - circular import guard
    def upsert_chunk_rows(self, rows: Sequence[ChunkRow]) -> None: ...
