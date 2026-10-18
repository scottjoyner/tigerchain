"""API authentication must run before expensive context construction.

This tests HTTP routing without booting embeddings or external services.
"""

from fastapi.testclient import TestClient

import server


def test_unauthenticated_api_paths_are_rejected_before_context_initialization():
    def forbidden_context():
        raise AssertionError("Unauthenticated request initialized graph/model context")

    # API routes use server.get_context; builder routes use server.build_context.
    server.app.dependency_overrides[server.get_context] = forbidden_context
    server.app.dependency_overrides[server.build_context] = forbidden_context
    try:
        client = TestClient(server.app)
        responses = [
            client.get("/agents"),
            client.post("/query", json={"question": "synthetic"}),
            client.post("/ingest/arxiv", json={"identifier": "2301.12345"}),
            client.get("/builder/snapshot"),
            client.get("/agent-builder"),
        ]
        assert all(response.status_code == 401 for response in responses)
    finally:
        server.app.dependency_overrides.pop(server.get_context, None)
        server.app.dependency_overrides.pop(server.build_context, None)
