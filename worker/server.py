from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import List, Literal, Optional

from fastapi import Body, Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlmodel import Session, select

from tigerchain_app.agents import QueryContext
from tigerchain_app.auth.database import get_session
from tigerchain_app.auth.models import DocumentUpload, User
from tigerchain_app.auth.router import router as auth_router
from tigerchain_app.auth.schemas import DocumentRecord
from tigerchain_app.auth.security import get_current_active_user
from tigerchain_app.auth.service import DocumentService
from tigerchain_app.context import build_context
from tigerchain_app.utils.logging import configure_logging, get_logger

configure_logging()
logger = get_logger(__name__)
app = FastAPI(title="TigerChain RAG Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router, prefix="/auth", tags=["auth"])


class QueryRequest(BaseModel):
    question: str
    agent: Optional[str] = None
    agents: Optional[List[str]] = None
    mode: Literal["sequential", "parallel"] = "sequential"
    categories: Optional[List[str]] = None


class IngestedDocumentResponse(BaseModel):
    doc_id: str
    filename: str
    categories: List[str]
    model_alias: Optional[str]
    object_uri: Optional[str]
    http_url: Optional[str]
    metadata: Optional[dict] = None


class IngestResponse(BaseModel):
    ingested_chunks: int
    documents: List[IngestedDocumentResponse]
    agent: str


class AgentResult(BaseModel):
    agent: str
    answer: Optional[str]
    sources: List[dict]


class QueryResponse(BaseModel):
    question: str
    mode: str
    answer: Optional[str]
    results: List[AgentResult]


@app.on_event("startup")
async def on_startup() -> None:
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
    context=Depends(get_context),
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    pipeline = context.pipeline
    storage_dir = context.settings.storage_base_path
    storage_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []

    extra_metadata: dict | None = None
    if metadata:
        try:
            extra_metadata = json.loads(metadata)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="metadata must be valid JSON") from exc

    for upload in files:
        temp_path = storage_dir / upload.filename
        with temp_path.open("wb") as buffer:
            shutil.copyfileobj(upload.file, buffer)
        paths.append(temp_path)
        logger.info("Uploaded %s (%s bytes)", upload.filename, temp_path.stat().st_size)

    orchestrator = context.agent_orchestrator
    available_agents = set(orchestrator.available_agents())
    preferred_agent = model_alias or current_user.preferred_agent or context.settings.default_agent
    if preferred_agent not in available_agents:
        raise HTTPException(status_code=400, detail=f"Unknown agent '{preferred_agent}'")

    category_values = set(current_user.categories or [])
    if category:
        category_values.add(category)
    if categories:
        category_values.update({value for value in categories if value})

    ingestion_result = pipeline.ingest_files(
        paths,
        owner_id=str(current_user.id),
        categories=category_values,
        model_alias=preferred_agent,
        extra_metadata=extra_metadata,
    )

    document_service = DocumentService(session)
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
        )
        response_docs.append(
            IngestedDocumentResponse(
                doc_id=record.doc_id,
                filename=record.filename,
                categories=record.categories,
                model_alias=record.model_alias,
                object_uri=record.object_uri,
                http_url=record.http_url,
                metadata=record.metadata,
            )
        )

    for path in paths:
        try:
            path.unlink(missing_ok=True)
        except Exception:
            logger.warning("Failed to remove temporary file %s", path)
    return IngestResponse(ingested_chunks=len(ingestion_result.chunks), documents=response_docs, agent=preferred_agent)


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
    query_context = QueryContext(
        owner_id=str(current_user.id),
        categories=categories,
        model_alias=agent_names[0],
    )
    results = await orchestrator.run_query(
        question=request.question,
        agent_names=agent_names,
        query_context=query_context,
        mode=request.mode,
    )
    formatted_results = [
        AgentResult(agent=result.agent, answer=result.answer, sources=result.sources)
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
    return {"status": "ok"}
