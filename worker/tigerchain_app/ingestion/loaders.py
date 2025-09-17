from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

from ..utils.logging import get_logger

logger = get_logger(__name__)


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


def _load_single_file(path: Path) -> List[Document]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        loader = PyPDFLoader(str(path))
    elif suffix in {".txt", ".md"}:
        loader = TextLoader(str(path), encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file extension: {path.suffix}")

    docs = loader.load()
    for doc in docs:
        doc.metadata.setdefault("source", str(path))
    return docs


def load_documents(paths: Iterable[Path]) -> List[Document]:
    documents: List[Document] = []
    for path in paths:
        if not path.exists() or not path.is_file():
            logger.warning("Skipping missing file %s", path)
            continue
        try:
            documents.extend(_load_single_file(path))
        except Exception as exc:  # pragma: no cover - log for traceability
            logger.exception("Failed to parse %s: %s", path, exc)
    return documents
