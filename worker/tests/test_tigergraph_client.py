"""Tests for the pyTigerGraph query contract; no live database required."""

import pytest

from tigerchain_app.config import Settings
from tigerchain_app.ingestion.tigergraph import TigerGraphClient


class StubConnection:
    def __init__(self):
        self.name = None
        self.kwargs = None

    def runInstalledQuery(self, name, *, params, usePost):
        self.name = name
        self.kwargs = {"params": params, "usePost": usePost}
        return [{"Ranked": []}]


def test_tigergraph_query_parameter_names_and_owner():
    client = object.__new__(TigerGraphClient)
    client.settings = Settings()
    client.conn = StubConnection()
    with pytest.raises(ValueError, match="owner_id"):
        client.top_k_similar([1.0], 5, owner_id="")
    with pytest.raises(ValueError, match="Private"):
        client.top_k_similar([1.0], 5, embedding_type="private", owner_id="alpha")
    with pytest.raises(ValueError, match="top_k"):
        client.top_k_similar([1.0], 101, owner_id="alpha")
    with pytest.raises(ValueError, match="embedding"):
        client.top_k_similar([], 5, owner_id="alpha")
    assert client.top_k_similar([0.1, 0.2], 3, owner_id="alpha") == {"results": [{"Ranked": []}]}
    assert client.conn.name == "SimilarChunks"
    assert client.conn.kwargs == {"params": {
        "q": [0.1, 0.2], "topk": 3, "embedding_type": "public", "owner_id": "alpha"
    }, "usePost": True}
