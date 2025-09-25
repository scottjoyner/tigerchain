from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

import json

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central application configuration loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=Path(__file__).resolve().parents[2] / ".env", env_file_encoding="utf-8", env_prefix="", case_sensitive=False)

    # TigerGraph
    tg_host: str = Field(default="tigergraph", description="TigerGraph REST++ host")
    tg_rest_port: int = Field(default=9000, description="TigerGraph REST++ port")
    tg_gsql_port: int = Field(default=14240, description="TigerGraph GSQL port")
    tg_user: str = Field(default="tigergraph", description="TigerGraph admin username")
    tg_password: str = Field(default="tigergraph", description="TigerGraph admin password")
    tg_graph: str = Field(default="DocGraph", description="Target TigerGraph graph name")
    tg_token_ttl: int = Field(default=2_592_000, description="REST token lifetime (seconds)")

    # Object storage
    minio_endpoint: str = Field(default="http://minio:9000", description="MinIO endpoint URL")
    minio_access_key: str = Field(default="admin", description="MinIO access key")
    minio_secret_key: str = Field(default="changeme-strong", description="MinIO secret key")
    minio_bucket: str = Field(default="docs", description="Bucket for original source files")
    minio_secure: bool = Field(default=False, description="Use HTTPS for MinIO connections")

    # Embeddings and chunking
    embed_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2", description="Embedding model name")
    embed_device: Optional[str] = Field(default=None, description="Optional device override for embeddings (cpu, cuda)")
    embed_batch_size: int = Field(default=32, description="Batch size for embedding generation")
    chunk_size: int = Field(default=900, description="Character chunk size for document splitting")
    chunk_overlap: int = Field(default=150, description="Chunk overlap for recursive splitter")
    bitwise_threshold: float = Field(
        default=0.0,
        description=(
            "Threshold used to convert dense embeddings into bitwise private embeddings. "
            "Values greater than or equal to the threshold are mapped to 1, others to 0."
        ),
    )

    # Retrieval
    top_k: int = Field(default=5, description="Number of similar chunks to fetch per query")

    # LLM providers
    llm_provider: Literal["ollama", "vllm", "openai", "anthropic"] = Field(default="ollama", description="LLM backend provider")
    llm_model: str = Field(default="llama2", description="Model name for the selected provider")
    llm_temperature: float = Field(default=0.1, description="Response creativity control")
    openai_api_key: Optional[str] = Field(default=None, description="Optional OpenAI compatible API key")
    openai_api_base: Optional[str] = Field(default=None, description="Override base URL for OpenAI compatible endpoints")
    ollama_base_url: str = Field(default="http://ollama:11434", description="Ollama service URL")
    vllm_api_base: str = Field(default="http://vllm:8000/v1", description="vLLM OpenAI-compatible endpoint")
    model_registry: dict[str, dict[str, str]] = Field(
        default_factory=lambda: {
            "default": {"provider": "ollama", "model": "llama2", "temperature": 0.1},
        },
        description=(
            "Mapping of logical agent names to provider/model definitions. "
            "Values may be overridden using the MODEL_REGISTRY environment variable containing a JSON object."
        ),
    )
    default_agent: str = Field(
        default="default",
        description="Agent identifier to use when a user has not selected a preferred model.",
    )

    # Authentication & persistence
    database_url: str = Field(
        default="sqlite:///./tigerchain.db",
        description="SQL database URL for authentication and metadata",
    )
    jwt_secret_key: str = Field(
        default="change-this-secret",
        description="Signing key for JSON web tokens",
    )
    jwt_algorithm: str = Field(default="HS256", description="JWT signing algorithm")
    access_token_expire_minutes: int = Field(
        default=60,
        description="Minutes before issued access tokens expire",
    )

    @field_validator("model_registry", mode="before")
    @classmethod
    def _parse_model_registry(cls, value: object) -> dict[str, dict[str, str]]:
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                if not isinstance(parsed, dict):
                    raise ValueError("MODEL_REGISTRY must be a JSON object")
                return parsed
            except json.JSONDecodeError as exc:  # pragma: no cover - defensive logging
                raise ValueError("MODEL_REGISTRY must be valid JSON") from exc
        return value  # type: ignore[return-value]

    # API
    api_host: str = Field(default="0.0.0.0", description="FastAPI bind host")
    api_port: int = Field(default=8000, description="FastAPI bind port")
    api_reload: bool = Field(default=False, description="Enable autoreload (development only)")
    storage_base_path: Path = Field(default=Path("/tmp/tigerchain"), description="Temporary path for uploaded files")

    class Config:
        env_prefix = ""


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load and cache Settings."""

    return Settings()
