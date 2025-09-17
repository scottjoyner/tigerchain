from __future__ import annotations

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings

from ..config import Settings
from ..utils.logging import get_logger

logger = get_logger(__name__)


def create_embeddings(settings: Settings) -> Embeddings:
    logger.info("Loading embedding model %s", settings.embed_model)
    model_kwargs = {}
    if settings.embed_device:
        model_kwargs["device"] = settings.embed_device
    encode_kwargs = {"batch_size": settings.embed_batch_size}
    return HuggingFaceEmbeddings(
        model_name=settings.embed_model,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )
