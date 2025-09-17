from __future__ import annotations

import json

from typing import List

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever

from ..config import Settings
from ..ingestion.tigergraph import TigerGraphClient
from ..utils.logging import get_logger

logger = get_logger(__name__)


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
        return results
