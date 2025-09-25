from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings

from ..config import Settings
from ..utils.logging import get_logger

logger = get_logger(__name__)


class DualEmbeddingProvider(Embeddings):
    """Wrapper that generates dense and bitwise embeddings in tandem."""

    def __init__(self, base: Embeddings, *, bitwise_threshold: float) -> None:
        self._base = base
        self._threshold = bitwise_threshold

    # ------------------------------------------------------------------
    # Standard embedding interface
    # ------------------------------------------------------------------
    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:  # type: ignore[override]
        # Returning the dense embeddings maintains compatibility with callers that
        # expect the LangChain Embeddings protocol. Additional helpers expose the
        # paired private representation.
        dense = self._base.embed_documents(texts)
        return dense

    def embed_query(self, text: str) -> List[float]:  # type: ignore[override]
        return self._base.embed_query(text)

    # ------------------------------------------------------------------
    # Extended helpers
    # ------------------------------------------------------------------
    def embed_documents_with_private(
        self, texts: Sequence[str]
    ) -> Tuple[List[List[float]], List[List[int]]]:
        dense_vectors = self._base.embed_documents(texts)
        private_vectors = [self._to_bitwise(vector) for vector in dense_vectors]
        return dense_vectors, private_vectors

    def embed_query_with_mode(self, text: str, mode: str) -> List[float | int]:
        if mode == "private":
            return self._to_bitwise(self._base.embed_query(text))
        return self._base.embed_query(text)

    def _to_bitwise(self, vector: Iterable[float]) -> List[int]:
        threshold = self._threshold
        return [1 if value >= threshold else 0 for value in vector]


def create_embeddings(settings: Settings) -> DualEmbeddingProvider:
    logger.info("Loading embedding model %s", settings.embed_model)
    model_kwargs = {}
    if settings.embed_device:
        model_kwargs["device"] = settings.embed_device
    encode_kwargs = {"batch_size": settings.embed_batch_size}
    base = HuggingFaceEmbeddings(
        model_name=settings.embed_model,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )
    return DualEmbeddingProvider(base, bitwise_threshold=settings.bitwise_threshold)
