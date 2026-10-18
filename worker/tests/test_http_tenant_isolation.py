"""Authenticated multi-user HTTP test with real SQLite/JWT and synthetic stores.

No network, external LLM, filesystem corpus, or Docker are required. The API
routes and real ingestion/retrieval pipeline run; graph and object I/O are
in-memory fakes so this test *does not* certify the real backends.
"""
import json
import os
import ipaddress
from urllib.parse import urlparse
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

import server
from tigerchain_app.agents.orchestrator import AgentQueryResult
from tigerchain_app.auth.database import get_session
from tigerchain_app.auth.models import User
from tigerchain_app.config import Settings, get_settings
from tigerchain_app.ingestion.pipeline import DocumentIngestionPipeline
from tigerchain_app.rag.retriever import RetrievalContext, TigerGraphVectorRetriever
from langchain_core.embeddings import Embeddings


class SyntheticEmbeddings(Embeddings):
    def embed_query(self, text):
        return [1.0, 0.0]

    def embed_documents(self, texts):
        return [[1.0, 0.0] for _ in texts]


class MemoryObjectStore:
    def __init__(self):
        self.files = {}

    def upload(self, path, key):
        self.files[key] = Path(path).read_bytes()
        return f"s3://synthetic/{key}", f"http://synthetic.invalid/{key}"


class MemoryGraph:
    def __init__(self):
        self.rows = {}

    def upsert_chunk_rows(self, rows):
        for row in rows:
            self.rows[row.id] = row

    def top_k_similar(self, embedding, top_k, embedding_type="public", *, owner_id):
        assert owner_id
        if embedding_type != "public":
            raise ValueError("Private scope is not enabled")
        hits = [
            {"v_id": row.id, "attributes": row.to_upsert_payload()["attributes"]}
            for row in self.rows.values()
            if row.owner_id == owner_id
        ]
        return {"results": [{"Ranked": hits[:top_k]}]}


class SyntheticOrchestrator:
    def __init__(self, retriever):
        self.retriever = retriever

    def available_agents(self):
        return ["default"]

    async def run_query(self, question, agent_names=None, query_context=None, mode="sequential"):
        assert query_context is not None
        with self.retriever.use_context(RetrievalContext(
            owner_id=query_context.owner_id,
            categories=query_context.categories,
            model_alias=query_context.model_alias,
            embedding_scope=query_context.embedding_scope,
        )):
            documents = self.retriever._get_relevant_documents(question)
        answer = " | ".join(doc.page_content for doc in documents)
        return [AgentQueryResult(agent="default", answer=answer, sources=[])]


@pytest.fixture
def environment(tmp_path):
    settings = Settings(
        jwt_secret_key="synthetic-test-only-signing-key-at-least-32-bytes",
        admin_user_ids="",
        enable_public_registration=True,
        database_url="sqlite://",
        storage_base_path=tmp_path / "staging",
        retrieval_overfetch_factor=1.0,
        top_k=10,
    )
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)

    def session_override():
        with Session(engine) as session:
            yield session

    embeddings = SyntheticEmbeddings()
    if os.environ.get("TIGERCHAIN_CANARY_ACK") == "SYNTHETIC_ONLY":
        # Opt-in real backends ONLY against private, disposable synthetic canaries.
        # This fixture does not start/stop containers; use a unique Compose project.
        from tigerchain_app.ingestion.tigergraph import TigerGraphClient
        from tigerchain_app.ingestion.storage import MinioObjectStore

        def private_url(variable):
            url = os.environ[variable]
            parsed = urlparse(url)
            if parsed.scheme != "http" or not parsed.hostname:
                raise ValueError(f"{variable} must use a plain HTTP isolated bridge address")
            if not ipaddress.ip_address(parsed.hostname).is_private:
                raise ValueError(f"{variable} must point to a private disposable canary")
            return url

        settings.tg_host = private_url("TIGERCHAIN_CANARY_GRAPH_URL")
        settings.minio_endpoint = private_url("TIGERCHAIN_CANARY_S3_URL")
        settings.minio_access_key = os.environ["TIGERCHAIN_CANARY_S3_ACCESS_KEY"]
        settings.minio_secret_key = os.environ["TIGERCHAIN_CANARY_S3_SECRET_KEY"]
        settings.minio_bucket = "tigerchain-http-20261018"

        class TrackingGraph(TigerGraphClient):
            def __init__(self, cfg):
                super().__init__(cfg)
                self.rows = {}

            def upsert_chunk_rows(self, rows):
                super().upsert_chunk_rows(rows)
                for row in rows:
                    self.rows[row.id] = row

        class TrackingStore(MinioObjectStore):
            def __init__(self, cfg):
                super().__init__(cfg)
                self.files = {}

            def upload(self, path, key):
                uri = super().upload(path, key)
                # Confirm the actual object-storage round-trip, not merely method calls.
                uploaded = self.client.get_object(Bucket=self.settings.minio_bucket, Key=key)["Body"].read()
                assert uploaded == Path(path).read_bytes()
                self.files[key] = uploaded
                return uri

        graph = TrackingGraph(settings)
        store = TrackingStore(settings)
    else:
        graph = MemoryGraph()
        store = MemoryObjectStore()
    retriever = TigerGraphVectorRetriever(settings, embeddings, graph)
    ctx = SimpleNamespace(
        settings=settings,
        pipeline=DocumentIngestionPipeline(settings, embeddings, graph, store),
        agent_orchestrator=SyntheticOrchestrator(retriever),
    )
    server.app.dependency_overrides[get_settings] = lambda: settings
    server.app.dependency_overrides[get_session] = session_override
    server.app.dependency_overrides[server.get_context] = lambda: ctx
    try:
        yield TestClient(server.app), engine, graph, store, tmp_path
    finally:
        server.app.dependency_overrides.pop(get_settings, None)
        server.app.dependency_overrides.pop(get_session, None)
        server.app.dependency_overrides.pop(server.get_context, None)
        engine.dispose()


def _register_and_login(client, username):
    email = f"{username}@synthetic.example"
    password = "example-long-password-only-for-isolated-tests"
    response = client.post("/auth/register", json={"email": email, "password": password})
    assert response.status_code == 200, response.text
    token = client.post("/auth/token", data={"username": email, "password": password})
    assert token.status_code == 200, token.text
    return response.json()["id"], {"Authorization": f"Bearer {token.json()['access_token']}"}


def test_real_http_jwt_upload_query_isolation_and_revocation(environment):
    client, engine, graph, store, tempdir = environment
    alice_id, alice_headers = _register_and_login(client, "alice")
    bob_id, bob_headers = _register_and_login(client, "bob")
    assert alice_id != bob_id

    for label, headers in (("alice", alice_headers), ("bob", bob_headers)):
        filename = f"{label}.txt"
        marker = f"ISOLATED_SYNTHETIC_{label.upper()}_DOCUMENT"
        response = client.post("/ingest", headers=headers,
            files=[("files", (filename, marker.encode(), "text/plain"))],
            data={"embedding_scope": "public"})
        assert response.status_code == 200, response.text
        assert response.json()["ingested_chunks"] == 1
        assert response.json()["documents"][0]["filename"] == filename

    for label, headers in (("alice", alice_headers), ("bob", bob_headers)):
        other = "bob" if label == "alice" else "alice"
        docs = client.get("/documents", headers=headers)
        assert docs.status_code == 200, docs.text
        assert [doc["filename"] for doc in docs.json()] == [f"{label}.txt"]
        query = client.post("/query", headers=headers, json={"question": "retrieve the note"})
        assert query.status_code == 200, query.text
        assert f"ISOLATED_SYNTHETIC_{label.upper()}" in query.json()["answer"]
        assert f"ISOLATED_SYNTHETIC_{other.upper()}" not in query.json()["answer"]
        assert client.get("/builder/agents", headers=headers).status_code == 403
        assert client.get("/agent-builder", headers=headers).status_code == 403
        assert client.post("/query", headers=headers, json={
            "question": "retrieve", "embedding_scope": "private"
        }).status_code == 403

    assert len(store.files) == 2
    assert len(graph.rows) == 2
    assert client.get("/documents").status_code == 401
    assert client.post("/query", json={"question": "missing token"}).status_code == 401
    assert client.get("/documents", headers={"Authorization": "Bearer invalid"}).status_code == 401

    with Session(engine) as session:
        alice = session.get(User, alice_id)
        alice.is_active = False
        session.add(alice)
        session.commit()
    assert client.get("/documents", headers=alice_headers).status_code in (400, 401, 403)
    assert client.post("/query", headers=alice_headers,
        json={"question": "revoked"}).status_code in (400, 401, 403)
    assert client.get("/documents", headers=bob_headers).status_code == 200

    assert not any(tempdir.joinpath("staging").iterdir()), "temporary uploads must be removed"


def test_self_registration_disabled_by_default_even_without_admin_ids(environment):
    client, engine, graph, store, tmp_path = environment
    settings = Settings(
        jwt_secret_key="synthetic-test-only-signing-key-at-least-32-bytes",
        enable_public_registration=False,
        admin_user_ids="",
    )
    server.app.dependency_overrides[get_settings] = lambda: settings
    response = client.post("/auth/register", json={
        "email": "stranger@synthetic.example", "password": "example-long-password-only-for-tests"
    })
    assert response.status_code == 403
    with Session(engine) as session:
        from sqlmodel import select
        assert session.exec(select(User)).all() == []


def test_open_registration_rejected_with_numeric_admin_allowlist(environment):
    client, engine, graph, store, tmp_path = environment
    settings = Settings(
        jwt_secret_key="synthetic-test-only-signing-key-at-least-32-bytes",
        enable_public_registration=True,
        admin_user_ids="1",
    )
    server.app.dependency_overrides[get_settings] = lambda: settings
    response = client.post("/auth/register", json={
        "email": "would-be-admin@synthetic.example", "password": "example-long-password-only-for-tests"
    })
    assert response.status_code == 403
    with Session(engine) as session:
        from sqlmodel import select
        assert session.exec(select(User)).all() == []


@pytest.mark.parametrize("case,expected", [
    ("spoofed_owner", 400),
    ("forged_arxiv_provenance", 400),
    ("shared_without_acls", 403),
    ("too_many_files", 413),
    ("oversized_metadata", 413),
])
def test_ingestion_rejects_provenance_spoofing_and_unbounded_requests(environment, case, expected):
    client, engine, graph, store, tmp_path = environment
    _, headers = _register_and_login(client, f"negative-{case}")
    payload = b"SYNTHETIC_CANARY_ONLY"
    if case == "forged_arxiv_provenance":
        result = client.post("/ingest/arxiv", headers=headers, json={
            "identifier": "2301.12345", "embedding_scope": "public",
            "sharing_preference": "private", "metadata": {"owner_id": "another-user"},
        })
    else:
        files = [("files", ("synthetic.txt", payload, "text/plain"))]
        data = {"embedding_scope": "public"}
        if case == "spoofed_owner":
            data["metadata"] = json.dumps({"owner_id": "another-user"})
        elif case == "shared_without_acls":
            data["sharing_preference"] = "both"
        elif case == "too_many_files":
            files = [("files", (f"doc-{i}.txt", payload, "text/plain")) for i in range(9)]
        elif case == "oversized_metadata":
            data["metadata"] = json.dumps({"description": "A" * (17 * 1024)})
        result = client.post("/ingest", headers=headers, files=files, data=data)
    assert result.status_code == expected, result.text
    assert not graph.rows
    assert not store.files
