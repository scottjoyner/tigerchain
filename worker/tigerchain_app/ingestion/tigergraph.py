from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable, Sequence

import pyTigerGraph as tg
import requests

from ..config import Settings
from ..utils.logging import get_logger
from .pipeline import ChunkRow

logger = get_logger(__name__)


class TigerGraphClient:
    """Thin wrapper around TigerGraph REST++ and pyTigerGraph helpers."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.conn = tg.TigerGraphConnection(
            host=f"http://{settings.tg_host}",
            restppPort=settings.tg_rest_port,
            gsqlPort=settings.tg_gsql_port,
            graphname=settings.tg_graph,
            username=settings.tg_user,
            password=settings.tg_password,
            useCert=False,
        )
        self._token: str | None = None
        self._token_expiry: float = 0.0

    # ------------------------------------------------------------------
    # Token management
    # ------------------------------------------------------------------
    def get_token(self) -> str:
        now = time.time()
        if self._token and now < self._token_expiry - 60:
            return self._token
        token, expiry = self.conn.getToken(self.settings.tg_token_ttl)
        self._token = token
        self._token_expiry = now + float(expiry)
        return token

    # ------------------------------------------------------------------
    # Schema installation utilities
    # ------------------------------------------------------------------
    def run_gsql_file(self, path: Path) -> None:
        logger.info("Applying GSQL script %s", path)
        script = Path(path).read_text(encoding="utf-8")
        script = script.replace("$TG_GRAPH", self.settings.tg_graph)
        self.conn.gsql(script)

    # ------------------------------------------------------------------
    # Data operations
    # ------------------------------------------------------------------
    def upsert_chunk_rows(self, rows: Sequence[ChunkRow]) -> None:
        logger.debug("Upserting %s chunk rows", len(rows))
        for row in rows:
            self.conn.upsertVertex("Document", row.id, row.to_upsert_payload()["attributes"])

    def top_k_similar(self, embedding: Iterable[float | int], top_k: int, embedding_type: str = "public") -> dict:
        token = self.get_token()
        response = requests.post(
            f"http://{self.settings.tg_host}:{self.settings.tg_rest_port}/query/{self.settings.tg_graph}/SimilarChunks",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json={"query_embedding": list(embedding), "topk": top_k, "embedding_type": embedding_type},
            timeout=120,
        )
        response.raise_for_status()
        return response.json()
