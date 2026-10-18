"""P0 authentication and query-contract regression tests (requires service deps)."""
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from tigerchain_app.config import Settings
from tigerchain_app.auth.security import require_admin_user, require_signing_key
from tigerchain_app.ingestion.tigergraph import TigerGraphClient


def test_signing_key_missing_or_known_defaults_fail_closed():
    for value in (None, "", "change-this-secret", "short"):
        with pytest.raises(RuntimeError, match="JWT_SECRET_KEY"):
            require_signing_key(Settings(jwt_secret_key=value))
    key = "unique-signing-key-material-with-at-least-32-bytes"
    assert require_signing_key(Settings(jwt_secret_key=key)) == key


def test_agent_administration_denied_without_explicit_allowlist():
    user = SimpleNamespace(id=42)
    with pytest.raises(HTTPException) as exc:
        require_admin_user(user, Settings(admin_user_ids=""))
    assert exc.value.status_code == 403
    with pytest.raises(HTTPException) as exc:
        require_admin_user(user, Settings(admin_user_ids="43, 99"))
    assert exc.value.status_code == 403
    assert require_admin_user(user, Settings(admin_user_ids="41, 42, 43")) == user


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
