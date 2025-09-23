from __future__ import annotations

import json

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Iterable, List, Optional, Set

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever

from ..config import Settings
from ..ingestion.tigergraph import TigerGraphClient
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RetrievalContext:
    owner_id: Optional[str] = None
    categories: Optional[Iterable[str]] = None
    model_alias: Optional[str] = None


class TigerGraphVectorRetriever(BaseRetriever):
    """LangChain retriever backed by TigerGraph vector similarity query."""

    def __init__(
        self,
        settings: Settings,
        embeddings: Embeddings,
        tigergraph_client: TigerGraphClient,
    ) -> None:
        super().__init__()
        self.settings = settings
        self.embeddings = embeddings
        self.tigergraph_client = tigergraph_client
        self._context: ContextVar[RetrievalContext | None] = ContextVar(
            f"tg_retriever_filters_{id(self)}",
            default=None,
        )

    def _get_relevant_documents(self, query: str) -> List[Document]:  # type: ignore[override]
        logger.debug("Embedding query for retrieval")
        embedding = self.embeddings.embed_query(query)
        response = self.tigergraph_client.top_k_similar(embedding, self.settings.top_k)
        return self._parse_response(response)

    async def _aget_relevant_documents(self, query: str) -> List[Document]:  # type: ignore[override]
        return self._get_relevant_documents(query)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @contextmanager
    def use_context(self, context: RetrievalContext):
        token = self._context.set(context)
        try:
            yield
        finally:  # pragma: no branch - ensure token reset
            self._context.reset(token)

    def _parse_response(self, payload: dict) -> List[Document]:
        results = []
        for result in payload.get("results", []):
            for entries in result.get("Start", []):
                attributes = entries.get("attributes", {})
                score = entries.get("score")
                if score is None:
                    score = attributes.get("score")
                metadata = {
                    "doc_id": attributes.get("doc_id"),
                    "chunk_id": attributes.get("chunk_id"),
                    "title": attributes.get("title"),
                    "source": attributes.get("source"),
                    "uri": attributes.get("uri"),
                    "http_url": attributes.get("http_url"),
                    "score": score,
                }
                try:
                    metadata.update(json.loads(attributes.get("metadata", "{}")))
                except Exception:  # pragma: no cover - fallback when metadata is invalid JSON
                    pass
                results.append(
                    Document(
                        page_content=attributes.get("content", ""),
                        metadata=metadata,
                    )
                )
        return self._filter_documents(results)

    def _filter_documents(self, documents: List[Document]) -> List[Document]:
        context = self._context.get()
        if context is None:
            return documents

        owner_id = str(context.owner_id) if context.owner_id is not None else None
        category_filters: Optional[Set[str]] = None
        if context.categories:
            category_filters = {str(cat).strip() for cat in context.categories if str(cat).strip()}
        model_alias = context.model_alias

        filtered: List[Document] = []
        for doc in documents:
            metadata = doc.metadata or {}
            if owner_id is not None and str(metadata.get("owner_id")) != owner_id:
                continue
            if category_filters:
                doc_categories = metadata.get("categories") or []
                if isinstance(doc_categories, str):
                    doc_categories = [doc_categories]
                doc_category_set = {str(cat).strip() for cat in doc_categories if str(cat).strip()}
                if not doc_category_set & category_filters:
                    continue
            if model_alias and metadata.get("model_alias") not in {model_alias, None}:
                continue
            filtered.append(doc)
        return filtered
