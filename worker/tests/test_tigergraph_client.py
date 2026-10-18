"""Client query contract requiring real project dependencies."""
import pytest
from tigerchain_app.ingestion.tigergraph import TigerGraphClient
from tigerchain_app.config import Settings

def test_tigergraph_query_parameter_names_and_owner(monkeypatch):
    import tigerchain_app.ingestion.tigergraph as module
    client = object.__new__(TigerGraphClient)
    client.settings = Settings(tg_host="localhost", tg_rest_port=9000, tg_graph="DocGraph")
    monkeypatch.setattr(client, "get_token", lambda: "synthetic-test-token")
    seen = {}
    class Response:
        def raise_for_status(self):
            pass
        def json(self):
            return {"results": [{"Ranked": []}]}
    def fake_post(url, *, headers, json, timeout):
        seen.update({"url": url, "payload": json, "timeout": timeout})
        return Response()
    monkeypatch.setattr(module.requests, "post", fake_post)
    with pytest.raises(ValueError, match="owner_id"):
        client.top_k_similar([1.0], 5, owner_id="")
    client.top_k_similar([0.1, 0.2], 3, owner_id="alice")
    assert seen["payload"] == {"q": [0.1, 0.2], "topk": 3, "embedding_type": "public", "owner_id": "alice"}
