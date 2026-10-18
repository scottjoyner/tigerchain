from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import pyTigerGraph as tg

from ..config import Settings
from ..utils.logging import get_logger
from .pipeline import ChunkRow

logger = get_logger(__name__)


class TigerGraphClient:
    """Explicitly owner-scoped graph adapter. Graph migrations are never run at startup."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        host = settings.tg_host
        if not host.startswith(("http://", "https://")):
            host = f"http://{host}"
        self.conn = tg.TigerGraphConnection(
            host=host,
            restppPort=settings.tg_rest_port,
            gsPort=settings.tg_gsql_port,
            graphname=settings.tg_graph,
            username=settings.tg_user,
            password=settings.tg_password,
            apiToken=settings.tg_api_token or "",
        )

    def run_gsql_file(self, path: Path) -> None:
        """Operator-only migration helper; never called during application startup."""
        script = Path(path).read_text(encoding="utf-8").replace("$TG_GRAPH", self.settings.tg_graph)
        result = self.conn.gsql(script)
        if isinstance(result, str) and ("Query installation failed" in result or "Syntax error" in result or "Semantic Check Error" in result):
            raise RuntimeError(f"GSQL migration failed for {path.name}")

    def upsert_chunk_rows(self, rows: Sequence[ChunkRow]) -> None:
        for row in rows:
            if not row.owner_id:
                raise ValueError("Graph upsert requires owner_id")
            self.conn.upsertVertex("Document", row.id, row.to_upsert_payload()["attributes"])

    def top_k_similar(
        self, embedding: Iterable[float | int], top_k: int, embedding_type: str = "public", *, owner_id: str
    ) -> dict:
        if not owner_id:
            raise ValueError("owner_id required for TigerGraph retrieval")
        if embedding_type != "public":
            raise ValueError("Private vector retrieval is disabled")
        if not 1 <= top_k <= 100:
            raise ValueError("top_k must be within [1, 100]")
        vector = [float(v) for v in embedding]
        if not vector:
            raise ValueError("Query embedding must not be empty")
        # Use the supported pyTigerGraph serializer for LIST<DOUBLE> parameters.
        # The old custom POST serialized the query parameters incorrectly.
        result = self.conn.runInstalledQuery(
            "SimilarChunks",
            params={"q": vector, "topk": top_k, "embedding_type": "public", "owner_id": owner_id},
            usePost=True,
        )
        return {"results": result}
