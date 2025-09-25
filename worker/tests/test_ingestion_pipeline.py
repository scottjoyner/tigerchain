from __future__ import annotations

from dataclasses import dataclass
import sys
import types
from pathlib import Path
from typing import Iterable, List

import pytest


def _install_langchain_stubs() -> None:
    if "langchain_core.embeddings" not in sys.modules:
        module = types.ModuleType("langchain_core.embeddings")

        class _Embeddings:  # pragma: no cover - runtime stub
            def embed_documents(self, texts: List[str]) -> List[List[float]]:
                raise NotImplementedError("LangChain embeddings are not installed")

        module.Embeddings = _Embeddings  # type: ignore[attr-defined]
        sys.modules[module.__name__] = module

    if "langchain_core.documents" not in sys.modules:
        module = types.ModuleType("langchain_core.documents")

        @dataclass
        class _Document:  # pragma: no cover - runtime stub
            page_content: str
            metadata: dict

        module.Document = _Document  # type: ignore[attr-defined]
        sys.modules[module.__name__] = module

    if "langchain_text_splitters" not in sys.modules:
        module = types.ModuleType("langchain_text_splitters")

        class _RecursiveCharacterTextSplitter:  # pragma: no cover - runtime stub
            def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
                self.chunk_size = chunk_size
                self.chunk_overlap = chunk_overlap

            def split_documents(self, documents: List[object]) -> List[object]:
                return list(documents)

        module.RecursiveCharacterTextSplitter = _RecursiveCharacterTextSplitter  # type: ignore[attr-defined]
        sys.modules[module.__name__] = module

    if "langchain_community.document_loaders" not in sys.modules:
        module = types.ModuleType("langchain_community.document_loaders")

        class _BaseLoader:  # pragma: no cover - runtime stub
            def __init__(self, path: str, **_: object) -> None:
                self.path = path

            def load(self) -> list:
                raise NotImplementedError

        class _TextLoader(_BaseLoader):  # pragma: no cover - runtime stub
            def load(self) -> list:
                from langchain_core.documents import Document

                content = Path(self.path).read_text(encoding="utf-8")
                return [Document(page_content=content, metadata={"source": self.path})]

        class _PyPDFLoader(_BaseLoader):  # pragma: no cover - runtime stub
            pass

        module.TextLoader = _TextLoader  # type: ignore[attr-defined]
        module.PyPDFLoader = _PyPDFLoader  # type: ignore[attr-defined]
        sys.modules[module.__name__] = module


_install_langchain_stubs()

from tigerchain_app.config import Settings
from tigerchain_app.ingestion.pipeline import DocumentIngestionPipeline, EmbeddingScope


class DummyEmbeddings:
    def __init__(self, base_value: float = 0.5) -> None:
        self.base_value = base_value
        self.calls: List[List[str]] = []

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        self.calls.append(list(texts))
        return [[self.base_value + index] for index, _ in enumerate(texts)]


class DummyTigerGraphClient:
    def __init__(self) -> None:
        self.rows: List[object] = []

    def upsert_chunk_rows(self, rows: Iterable[object]) -> None:
        self.rows.extend(rows)


class DummyObjectStore:
    def __init__(self) -> None:
        self.uploaded: dict[str, tuple[Path, str]] = {}
        self.uploaded_json: dict[str, dict] = {}

    def upload(self, path: Path, key: str) -> tuple[str, str]:
        self.uploaded[key] = (path, key)
        return (f"s3://bucket/{key}", f"http://minio/bucket/{key}")

    def upload_json(self, payload: dict, key: str) -> tuple[str, str]:
        self.uploaded_json[key] = payload
        return (f"s3://bucket/{key}", f"http://minio/bucket/{key}")


@pytest.fixture()
def pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> DocumentIngestionPipeline:
    embeddings = DummyEmbeddings()
    client = DummyTigerGraphClient()
    store = DummyObjectStore()
    settings = Settings()

    def _fake_loader(paths: Iterable[Path]) -> List[object]:
        docs: List[object] = []
        for path in paths:
            docs.append(types.SimpleNamespace(page_content=path.read_text(), metadata={"source": str(path)}))
        return docs

    from tigerchain_app.ingestion import pipeline as pipeline_module

    monkeypatch.setattr(pipeline_module, "load_documents", _fake_loader)

    return DocumentIngestionPipeline(settings, embeddings, client, store)


def test_ingest_files_populates_metadata(tmp_path: Path, pipeline: DocumentIngestionPipeline) -> None:
    source = tmp_path / "example.txt"
    source.write_text("Hello world")

    result = pipeline.ingest_files([source], owner_id="user-1", categories=["alpha", "alpha"], embedding_scope="both")

    assert result.chunks, "expected at least one chunk"
    row = result.chunks[0]
    assert "source_checksum" in row.metadata
    assert row.metadata["source_file_size"] == source.stat().st_size
    assert result.documents[0].source_checksum == row.metadata["source_checksum"]
    assert result.documents[0].file_size_bytes == source.stat().st_size
    assert row.metadata.get("subject_tags")
    assert "importance_score" in row.metadata
    document_metadata = result.documents[0].metadata
    assert document_metadata.get("subject_tags")
    assert document_metadata.get("importance_score") is not None

    store: DummyObjectStore = pipeline.object_store  # type: ignore[attr-defined]
    assert store.uploaded_json, "private embeddings should be persisted in 'both' scope"


def test_public_scope_skips_private_embeddings(tmp_path: Path, pipeline: DocumentIngestionPipeline) -> None:
    source = tmp_path / "only_public.txt"
    source.write_text("Content for public scope")

    result = pipeline.ingest_files([source], embedding_scope=EmbeddingScope.PUBLIC)

    assert all(not row.private_embedding for row in result.chunks)
    store: DummyObjectStore = pipeline.object_store  # type: ignore[attr-defined]
    assert not store.uploaded_json, "public scope should not persist private embeddings"


def test_invalid_scope_raises(tmp_path: Path, pipeline: DocumentIngestionPipeline) -> None:
    source = tmp_path / "invalid.txt"
    source.write_text("content")

    with pytest.raises(ValueError):
        pipeline.ingest_files([source], embedding_scope="invalid-scope")
