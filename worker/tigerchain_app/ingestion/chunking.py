from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from ..config import Settings


class AdaptiveChunker:
    """Adaptive text splitter that tunes chunk parameters per document type."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._splitters: Dict[Tuple[int, int], RecursiveCharacterTextSplitter] = {}

    def split_documents(self, documents: list[Document]) -> list[Document]:
        chunks: list[Document] = []
        for document in documents:
            chunk_size, chunk_overlap = self._resolve_params(document)
            splitter = self._get_splitter(chunk_size, chunk_overlap)
            chunks.extend(splitter.split_documents([document]))
        return chunks

    def _get_splitter(self, chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
        key = (chunk_size, chunk_overlap)
        if key not in self._splitters:
            self._splitters[key] = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        return self._splitters[key]

    def _resolve_params(self, document: Document) -> tuple[int, int]:
        chunk_size = self.settings.chunk_size
        chunk_overlap = self.settings.chunk_overlap

        source = document.metadata.get("source")
        suffix = Path(source).suffix.lower() if isinstance(source, str) else ""
        content_length = len(document.page_content or "")

        if suffix == ".md":
            chunk_size = max(400, chunk_size // 2)
            chunk_overlap = min(chunk_overlap, chunk_size // 4)
        elif suffix == ".pdf" and content_length > 4000:
            chunk_size = min(chunk_size + 200, 1200)
        elif suffix == ".txt" and content_length < chunk_size:
            chunk_size = max(256, max(content_length // 2, 128))
            chunk_overlap = min(chunk_overlap, chunk_size // 4)

        return chunk_size, chunk_overlap
