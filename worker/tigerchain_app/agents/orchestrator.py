from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

from langchain.chains import RetrievalQA
from langchain_core.embeddings import Embeddings

from ..config import Settings
from ..rag.chain import build_qa_chain, format_sources
from ..rag.llms import create_llm_from_config
from ..rag.retriever import RetrievalContext, TigerGraphVectorRetriever
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class QueryContext:
    owner_id: Optional[str] = None
    categories: Optional[Iterable[str]] = None
    model_alias: Optional[str] = None
    embedding_scope: Optional[str] = None


@dataclass
class AgentQueryResult:
    agent: str
    answer: Optional[str]
    sources: List[Dict[str, Optional[str]]]


class RagAgent:
    def __init__(
        self,
        name: str,
        config: Dict[str, object],
        settings: Settings,
        embeddings: Embeddings,
        tigergraph_client: "TigerGraphClient",
    ) -> None:
        self.name = name
        self.config = config
        self.settings = settings
        self.embeddings = embeddings
        self.tigergraph_client = tigergraph_client
        self._retriever: TigerGraphVectorRetriever | None = None
        self._chain: RetrievalQA | None = None

    def _ensure_components(self) -> tuple[TigerGraphVectorRetriever, RetrievalQA]:
        if self._retriever is None:
            self._retriever = TigerGraphVectorRetriever(self.settings, self.embeddings, self.tigergraph_client)
        if self._chain is None:
            llm = create_llm_from_config(self.settings, self.config)
            self._chain = build_qa_chain(self.settings, self._retriever, llm)
        return self._retriever, self._chain

    async def arun(self, query: str, context: QueryContext) -> AgentQueryResult:
        retriever, chain = self._ensure_components()
        categories = set(context.categories or []) or None
        retrieval_context = RetrievalContext(
            owner_id=context.owner_id,
            categories=categories,
            model_alias=context.model_alias or self.name,
            embedding_scope=context.embedding_scope,
        )
        logger.debug("Agent %s executing query with filters: %s", self.name, retrieval_context)
        with retriever.use_context(retrieval_context):
            result = await chain.ainvoke({"query": query})
        answer = result.get("result") if isinstance(result, dict) else None
        sources = format_sources(result.get("source_documents", [])) if isinstance(result, dict) else []
        return AgentQueryResult(agent=self.name, answer=answer, sources=sources)


class AgentOrchestrator:
    """Coordinates multiple local or remote RAG agents."""

    def __init__(
        self,
        settings: Settings,
        embeddings: Embeddings,
        tigergraph_client: "TigerGraphClient",
    ) -> None:
        self.settings = settings
        self.embeddings = embeddings
        self.tigergraph_client = tigergraph_client
        self._agents: Dict[str, RagAgent] = {}

    # ------------------------------------------------------------------
    # Agent management
    # ------------------------------------------------------------------
    def available_agents(self) -> List[str]:
        return sorted(self.settings.model_registry.keys())

    def _get_agent(self, name: str) -> RagAgent:
        if name not in self.settings.model_registry:
            raise ValueError(f"Unknown agent '{name}'")
        if name not in self._agents:
            config = self.settings.model_registry[name]
            self._agents[name] = RagAgent(name, config, self.settings, self.embeddings, self.tigergraph_client)
        return self._agents[name]

    # ------------------------------------------------------------------
    # Query execution
    # ------------------------------------------------------------------
    async def run_query(
        self,
        question: str,
        agent_names: Optional[Sequence[str]] = None,
        query_context: Optional[QueryContext] = None,
        mode: str = "sequential",
    ) -> List[AgentQueryResult]:
        if not agent_names:
            agent_names = [self.settings.default_agent]
        query_context = query_context or QueryContext()

        agents = [self._get_agent(name) for name in agent_names]
        if mode == "parallel" and len(agents) > 1:
            tasks = [agent.arun(question, query_context) for agent in agents]
            results = await asyncio.gather(*tasks)
        else:
            results = []
            for agent in agents:
                results.append(await agent.arun(question, query_context))
        return results


class TigerGraphClient:  # pragma: no cover - circular reference helper
    ...
