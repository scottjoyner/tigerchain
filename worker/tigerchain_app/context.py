from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from langchain.chains import RetrievalQA
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel

from .agents import AgentOrchestrator
from .agents import models as _agent_models  # noqa: F401 - ensure models registered
from .agents.registry import AgentRegistryLoader, AgentRegistrySnapshot
from .auth import models as _auth_models  # noqa: F401 - ensure models registered
from .auth.database import init_db, session_scope
from .config import Settings, get_settings
from .ingestion.embeddings import create_embeddings
from .ingestion.pipeline import DocumentIngestionPipeline
from .ingestion.storage import MinioObjectStore
from .ingestion.tigergraph import TigerGraphClient
from .rag.chain import build_qa_chain
from .rag.llms import create_llm
from .rag.retriever import TigerGraphVectorRetriever
from .utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ApplicationContext:
    settings: Settings
    embeddings: Embeddings
    tigergraph: TigerGraphClient
    object_store: MinioObjectStore
    pipeline: DocumentIngestionPipeline
    retriever: TigerGraphVectorRetriever
    llm: BaseLanguageModel
    qa_chain: RetrievalQA
    agent_orchestrator: AgentOrchestrator


_context: ApplicationContext | None = None


def build_context(force: bool = False) -> ApplicationContext:
    global _context
    if _context is not None and not force:
        return _context

    settings = get_settings()
    init_db(settings)
    base_registry = dict(settings.model_registry)
    registry_snapshot = AgentRegistrySnapshot()
    with session_scope(settings) as session:
        loader = AgentRegistryLoader(session, settings)
        registry_snapshot = loader.build_snapshot()
    if registry_snapshot.registry:
        combined_registry = dict(base_registry)
        combined_registry.update(registry_snapshot.registry)
        settings.model_registry = combined_registry

    embeddings = create_embeddings(settings)
    tigergraph = TigerGraphClient(settings)
    _bootstrap_gsql(tigergraph)
    object_store = MinioObjectStore(settings)
    pipeline = DocumentIngestionPipeline(settings, embeddings, tigergraph, object_store)
    retriever = TigerGraphVectorRetriever(settings, embeddings, tigergraph)
    llm = create_llm(settings)
    qa_chain = build_qa_chain(settings, retriever, llm)
    agent_orchestrator = AgentOrchestrator(
        settings,
        embeddings,
        tigergraph,
        base_registry=base_registry,
        snapshot=registry_snapshot,
    )

    _context = ApplicationContext(
        settings=settings,
        embeddings=embeddings,
        tigergraph=tigergraph,
        object_store=object_store,
        pipeline=pipeline,
        retriever=retriever,
        llm=llm,
        qa_chain=qa_chain,
        agent_orchestrator=agent_orchestrator,
    )
    logger.info("Application context initialised")
    return _context


def _bootstrap_gsql(tigergraph: TigerGraphClient) -> None:
    gsql_dir = Path(__file__).resolve().parents[2] / "gsql"
    for script_name in ["schema.gsql", "queries.gsql"]:
        script_path = gsql_dir / script_name
        if not script_path.exists():
            continue
        try:
            tigergraph.run_gsql_file(script_path)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("Failed to apply %s: %s", script_path, exc)
