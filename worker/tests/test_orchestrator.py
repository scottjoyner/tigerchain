import asyncio
import sys
import types
from typing import Dict

import pytest

if "langchain.chains" not in sys.modules:
    chains_module = types.ModuleType("langchain.chains")

    class _RetrievalQA:  # pragma: no cover - minimal async stub
        async def ainvoke(self, _: dict) -> dict:
            return {}

    chains_module.RetrievalQA = _RetrievalQA  # type: ignore[attr-defined]
    sys.modules[chains_module.__name__] = chains_module

if "langchain_core.embeddings" not in sys.modules:
    embeddings_module = types.ModuleType("langchain_core.embeddings")

    class _Embeddings:  # pragma: no cover - runtime stub
        ...

    embeddings_module.Embeddings = _Embeddings  # type: ignore[attr-defined]
    sys.modules[embeddings_module.__name__] = embeddings_module

if "pyTigerGraph" not in sys.modules:
    tigergraph_module = types.ModuleType("pyTigerGraph")
    tigergraph_module.TigerGraphConnection = object  # type: ignore[attr-defined]
    sys.modules[tigergraph_module.__name__] = tigergraph_module

if "langchain_openai" not in sys.modules:
    openai_module = types.ModuleType("langchain_openai")

    class _ChatOpenAI:  # pragma: no cover - runtime stub
        ...

    openai_module.ChatOpenAI = _ChatOpenAI  # type: ignore[attr-defined]
    sys.modules[openai_module.__name__] = openai_module

from tigerchain_app.agents.orchestrator import AgentOrchestrator, AgentQueryResult, QueryContext
from tigerchain_app.config import Settings


class _StubAgent:
    def __init__(self, name: str) -> None:
        self.name = name
        self.contexts: list[QueryContext] = []

    async def arun(self, question: str, context: QueryContext) -> AgentQueryResult:
        self.contexts.append(context)
        return AgentQueryResult(agent=self.name, answer=f"{self.name}:{question}", sources=[])


class _DummyEmbeddings:
    pass


class _DummyTigerGraphClient:
    pass


def test_orchestrator_prioritises_subject_alignment(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings()
    settings.model_registry = {
        "alpha": {"subject_specialities": ["compliance"]},
        "beta": {"subject_specialities": ["product"]},
    }
    orchestrator = AgentOrchestrator(settings, _DummyEmbeddings(), _DummyTigerGraphClient())

    stubs: Dict[str, _StubAgent] = {name: _StubAgent(name) for name in settings.model_registry}
    monkeypatch.setattr(orchestrator, "_get_agent", lambda name: stubs[name])

    context = QueryContext(subject_priorities={"compliance": 0.9, "product": 0.4})
    results = asyncio.run(
        orchestrator.run_query(
            "How to comply?", agent_names=["beta", "alpha"], query_context=context, mode="parallel"
        )
    )

    assert [result.agent for result in results][:2] == ["alpha", "beta"]
    assert all(ctx.model_alias is None for ctx in stubs["alpha"].contexts)
    assert all(ctx.model_alias is None for ctx in stubs["beta"].contexts)
