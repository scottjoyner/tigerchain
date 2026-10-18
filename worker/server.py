from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Literal, Optional

from fastapi import Body, Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlmodel import Session, select

from tigerchain_app.agents import QueryContext
from tigerchain_app.auth.database import get_session
from tigerchain_app.auth.models import DocumentUpload, User
from tigerchain_app.auth.router import router as auth_router
from tigerchain_app.agents.router import router as builder_router
from tigerchain_app.auth.schemas import DocumentRecord
from tigerchain_app.auth.security import get_current_active_user
from tigerchain_app.auth.service import DocumentService
from tigerchain_app.context import build_context
from tigerchain_app.config import get_settings
from tigerchain_app.auth.security import require_signing_key
from tigerchain_app.ingestion.upload_validation import validate_upload_filename
from tigerchain_app.ingestion.pipeline import IngestionResult
from tigerchain_app.utils.logging import configure_logging, get_logger
from tigerchain_app.utils.subjects import SubjectClassifier

configure_logging()
logger = get_logger(__name__)
app = FastAPI(title="TigerChain RAG Service")
subject_classifier = SubjectClassifier()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in get_settings().cors_allowed_origins.split(",") if origin.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router, prefix="/auth", tags=["auth"])
app.include_router(builder_router)


@app.get("/agent-builder", response_class=HTMLResponse)
async def agent_builder_ui() -> HTMLResponse:
    template_path = Path(__file__).resolve().parent / "tigerchain_app" / "agents" / "templates" / "agent_builder.html"
    if not template_path.exists():
        raise HTTPException(status_code=404, detail="Agent builder UI not found")
    return HTMLResponse(content=template_path.read_text(encoding="utf-8"))


class QueryRequest(BaseModel):
    question: str
    agent: Optional[str] = None
    agents: Optional[List[str]] = None
    mode: Literal["sequential", "parallel"] = "sequential"
    categories: Optional[List[str]] = None
    embedding_scope: Optional[Literal["public", "private"]] = None


class IngestedDocumentResponse(BaseModel):
    doc_id: str
    filename: str
    categories: List[str]
    model_alias: Optional[str]
    object_uri: Optional[str]
    http_url: Optional[str]
    metadata: Optional[dict] = None
    submission_id: str
    private_embedding_uri: Optional[str]
    embedding_scope: str
    sharing_preference: str
    file_size_bytes: Optional[int] = None
    source_checksum: Optional[str] = None


class IngestResponse(BaseModel):
    ingested_chunks: int
    documents: List[IngestedDocumentResponse]
    agent: str


class ArxivIngestRequest(BaseModel):
    identifier: str
    category: Optional[str] = None
    categories: Optional[List[str]] = None
    model_alias: Optional[str] = None
    metadata: Optional[dict] = None
    embedding_scope: Literal["public", "private", "both"] = "public"
    sharing_preference: Literal["public", "private", "both"] = "both"
    submission_id: Optional[str] = None


def _require_embedding_scope(scope: str) -> None:
    if scope != "public" and not get_settings().enable_private_embeddings:
        raise HTTPException(status_code=403, detail="Private embeddings disabled pending storage authorization review")


def _select_agent_alias(context, current_user: User, model_alias: Optional[str]) -> str:
    orchestrator = context.agent_orchestrator
    available_agents = set(orchestrator.available_agents())
    preferred_agent = model_alias or current_user.preferred_agent or context.settings.default_agent
    if preferred_agent not in available_agents:
        raise HTTPException(status_code=400, detail=f"Unknown agent '{preferred_agent}'")
    return preferred_agent


def _merge_categories(current_user: User, category: Optional[str], categories: Optional[List[str]]) -> set[str]:
    category_values = set(current_user.categories or [])
    if category:
        category_values.add(category)
    if categories:
        category_values.update({value for value in categories if value})
    return {value for value in category_values if value}


def _persist_ingestion_records(
    ingestion_result: IngestionResult,
    *,
    document_service: DocumentService,
    current_user: User,
    sharing_preference: str,
) -> List[IngestedDocumentResponse]:
    response_docs: List[IngestedDocumentResponse] = []
    for summary in ingestion_result.documents:
        record = document_service.record_upload(
            user_id=current_user.id,
            doc_id=summary.doc_id,
            filename=summary.source_path.name,
            categories=summary.categories,
            model_alias=summary.model_alias,
            object_uri=summary.uri,
            http_url=summary.http_url,
            metadata=summary.metadata,
            submission_id=summary.submission_id,
            embedding_scope=summary.embedding_scope,
            sharing_preference=sharing_preference,
            private_embedding_uri=summary.private_embedding_uri,
        )
        response_docs.append(
            IngestedDocumentResponse(
                doc_id=record.doc_id,
                filename=record.filename,
                categories=record.categories,
                model_alias=record.model_alias,
                object_uri=record.object_uri,
                http_url=record.http_url,
                metadata=record.extra_metadata,
                submission_id=record.submission_id or summary.submission_id,
                private_embedding_uri=record.private_embedding_uri or summary.private_embedding_uri,
                embedding_scope=record.embedding_scope or summary.embedding_scope,
                sharing_preference=record.sharing_preference or sharing_preference,
                file_size_bytes=summary.file_size_bytes,
                source_checksum=summary.source_checksum,
            )
        )
    return response_docs


class AgentResult(BaseModel):
    agent: str
    answer: Optional[str]
    sources: List[dict]
    metadata: Optional[dict] = None


class QueryResponse(BaseModel):
    question: str
    mode: str
    answer: Optional[str]
    results: List[AgentResult]


@app.on_event("startup")
async def on_startup() -> None:
    require_signing_key(get_settings())
    build_context(force=True)
    logger.info("TigerChain service started")


def get_context():
    return build_context()


@app.post("/ingest", response_model=IngestResponse)
async def ingest_documents(
    files: List[UploadFile] = File(..., description="Documents to ingest"),
    category: Optional[str] = Form(default=None, description="Primary category for the upload"),
    categories: Optional[List[str]] = Form(default=None, description="Additional categories"),
    model_alias: Optional[str] = Form(default=None, description="Agent/model alias to tag the document"),
    metadata: Optional[str] = Form(default=None, description="Optional JSON metadata to persist with the upload"),
    embedding_scope: Literal["public", "private", "both"] = Form(
        default="public", description="Embedding visibility preference"
    ),
    sharing_preference: Literal["public", "private", "both"] = Form(
        default="both", description="Embedding sets that may be shared"
    ),
    submission_id: Optional[str] = Form(
        default=None,
        description="Optional client-provided submission identifier to tag stored embeddings",
    ),
    context=Depends(get_context),
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    pipeline = context.pipeline
    storage_dir = context.settings.storage_base_path
    storage_dir.mkdir(parents=True, exist_ok=True)
    _require_embedding_scope(embedding_scope)
    with TemporaryDirectory(prefix="tigerchain-ingest-", dir=storage_dir) as staged_dir:
        paths: List[Path] = []
        extra_metadata: dict | None = None
        if metadata:
            try:
                extra_metadata = json.loads(metadata)
            except json.JSONDecodeError as exc:
                raise HTTPException(status_code=400, detail="metadata must be valid JSON") from exc
            if not isinstance(extra_metadata, dict):
                raise HTTPException(status_code=400, detail="metadata must be a JSON object")

        seen: set[str] = set()
        for upload in files:
            try:
                name = validate_upload_filename(upload.filename)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            if name in seen:
                raise HTTPException(status_code=400, detail="Duplicate upload filename")
            seen.add(name)
            temp_path = Path(staged_dir) / name
            size = 0
            with temp_path.open("xb") as buffer:
                while block := await upload.read(1024 * 1024):
                    size += len(block)
                    if size > 32 * 1024 * 1024:
                        raise HTTPException(status_code=413, detail="Upload exceeds 32 MiB limit")
                    buffer.write(block)
            paths.append(temp_path)

        preferred_agent = _select_agent_alias(context, current_user, model_alias)
        category_values = _merge_categories(current_user, category, categories)
        ingestion_result = pipeline.ingest_files(
            paths,
            owner_id=str(current_user.id),
            categories=category_values,
            model_alias=preferred_agent,
            extra_metadata=extra_metadata,
            embedding_scope=embedding_scope,
            submission_id=submission_id,
        )
        document_service = DocumentService(session)
        response_docs = _persist_ingestion_records(
            ingestion_result,
            document_service=document_service,
            current_user=current_user,
            sharing_preference=sharing_preference,
        )
        return IngestResponse(ingested_chunks=len(ingestion_result.chunks), documents=response_docs, agent=preferred_agent)

@app.post("/ingest/arxiv", response_model=IngestResponse)
async def ingest_arxiv_document(
    request: ArxivIngestRequest,
    context=Depends(get_context),
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    _require_embedding_scope(request.embedding_scope)
    pipeline = context.pipeline
    preferred_agent = _select_agent_alias(context, current_user, request.model_alias)
    category_values = _merge_categories(current_user, request.category, request.categories)
    extra_metadata = dict(request.metadata) if request.metadata else None

    ingestion_result = pipeline.ingest_arxiv(
        request.identifier,
        owner_id=str(current_user.id),
        categories=category_values,
        model_alias=preferred_agent,
        extra_metadata=extra_metadata,
        embedding_scope=request.embedding_scope,
        submission_id=request.submission_id,
    )

    document_service = DocumentService(session)
    response_docs = _persist_ingestion_records(
        ingestion_result,
        document_service=document_service,
        current_user=current_user,
        sharing_preference=request.sharing_preference,
    )
    return IngestResponse(
        ingested_chunks=len(ingestion_result.chunks),
        documents=response_docs,
        agent=preferred_agent,
    )


@app.post("/query", response_model=QueryResponse)
async def query_rag(
    request: QueryRequest = Body(..., description="Query payload"),
    context=Depends(get_context),
    current_user: User = Depends(get_current_active_user),
):
    orchestrator = context.agent_orchestrator
    agent_names = request.agents or ([request.agent] if request.agent else None)
    if agent_names:
        available_agents = set(orchestrator.available_agents())
        invalid = [name for name in agent_names if name not in available_agents]
        if invalid:
            raise HTTPException(status_code=400, detail=f"Unknown agent(s): {', '.join(invalid)}")
    else:
        default_agent = current_user.preferred_agent or context.settings.default_agent
        agent_names = [default_agent]

    categories = request.categories or current_user.categories
    subject_priorities = subject_classifier.priorities_from_categories(categories)
    shared_alias = agent_names[0] if len(agent_names) == 1 else None
    query_context = QueryContext(
        owner_id=str(current_user.id),
        categories=categories,
        model_alias=shared_alias,
        embedding_scope=request.embedding_scope,
        subject_priorities=subject_priorities or None,
    )
    results = await orchestrator.run_query(
        question=request.question,
        agent_names=agent_names,
        query_context=query_context,
        mode=request.mode,
    )
    formatted_results = [
        AgentResult(
            agent=result.agent,
            answer=result.answer,
            sources=result.sources,
            metadata=result.metadata,
        )
        for result in results
    ]
    answer = formatted_results[0].answer if formatted_results else None
    return QueryResponse(question=request.question, mode=request.mode, answer=answer, results=formatted_results)


@app.get("/agents")
async def list_agents(context=Depends(get_context)):
    settings = context.settings
    agents = []
    for name in context.agent_orchestrator.available_agents():
        config = settings.model_registry.get(name, {})
        agents.append(
            {
                "name": name,
                "provider": config.get("provider"),
                "model": config.get("model"),
                "temperature": config.get("temperature"),
            }
        )
    return {"agents": agents}


@app.get("/documents", response_model=List[DocumentRecord])
async def list_user_documents(
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    statement = (
        select(DocumentUpload)
        .where(DocumentUpload.user_id == current_user.id)
        .order_by(DocumentUpload.created_at.desc())
        .limit(50)
    )
    records = session.exec(statement).all()
    return [DocumentRecord.model_validate(record) for record in records]


@app.get("/healthz")
async def healthcheck():
    return {"status": "alive", "readiness": "not_checked"}
