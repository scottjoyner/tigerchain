"""Run explicitly against an isolated TigerGraph instance with TIGERCHAIN_CANARY_URL set.

Requires an installed SimilarChunks query and an empty/synthetic disposable graph.
Never configure this test against any production graph: it inserts test records.
"""
import os
import uuid
import json

import pytest
import pyTigerGraph


@pytest.mark.integration
def test_real_owner_boundaries():
    endpoint = os.getenv("TIGERCHAIN_CANARY_URL")
    if not endpoint:
        pytest.skip("TIGERCHAIN_CANARY_URL not set; real graph test is opt-in")
    graph = os.getenv("TIGERCHAIN_CANARY_GRAPH", "DocGraph")
    client = pyTigerGraph.TigerGraphConnection(host=endpoint, graphname=graph)
    seed = uuid.uuid4().hex[:12]
    ids = []
    try:
        for owner, vector in (("alpha", [1.0, 0.0]), ("beta", [0.0, 1.0])):
            identifier = f"tigerchain-canary-{seed}-{owner}"
            ids.append(identifier)
            assert client.upsertVertex("Document", identifier, {
                "owner_id": owner, "doc_id": identifier, "chunk_id": 0,
                "title": f"synthetic-{owner}", "content": f"SYNTHETIC_{owner.upper()}",
                "embedding": vector, "metadata": json.dumps({"owner_id": owner, "embedding_scope": "public"}),
            }) == 1
        for owner, expected in (("alpha", 1), ("beta", 1), ("absent", 0)):
            result = client.runInstalledQuery("SimilarChunks", params={
                "q": [1.0, 0.0], "topk": 100,
                "embedding_type": "public", "owner_id": owner,
            }, usePost=True)
            matches = [item for item in result[0]["Ranked"] if item["v_id"] in ids]
            assert len(matches) == expected
            assert all(row["attributes"]["owner_id"] == owner for row in matches)
        private = client.runInstalledQuery("SimilarChunks", params={
            "q": [1.0, 0.0], "topk": 100, "embedding_type": "private", "owner_id": "alpha",
        }, usePost=True)
        assert all(item["v_id"] not in ids for item in private[0]["Ranked"])
    finally:
        # Only delete exact synthetic IDs created during this test.
        for identifier in ids:
            try:
                client.delVerticesById("Document", [identifier])
            except Exception:
                pass
