from __future__ import annotations

import shutil
from pathlib import Path
from typing import List

from fastapi import Depends, FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from tigerchain_app.context import build_context
from tigerchain_app.rag.chain import format_sources
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


@app.on_event("startup")
async def on_startup() -> None:
    build_context(force=True)
    logger.info("TigerChain service started")


def get_context():
    return build_context()


@app.post("/ingest")
async def ingest_documents(
    files: List[UploadFile] = File(..., description="Documents to ingest"),
    context=Depends(get_context),
):
    pipeline = context.pipeline
    storage_dir = context.settings.storage_base_path
    storage_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []

    for upload in files:
        temp_path = storage_dir / upload.filename
        with temp_path.open("wb") as buffer:
            shutil.copyfileobj(upload.file, buffer)
        paths.append(temp_path)
        logger.info("Uploaded %s (%s bytes)", upload.filename, temp_path.stat().st_size)

    rows = pipeline.ingest_files(paths)
    for path in paths:
        try:
            path.unlink(missing_ok=True)
        except Exception:
            logger.warning("Failed to remove temporary file %s", path)
    return {"ingested_chunks": len(rows)}


@app.post("/query")
async def query_rag(
    question: str,
    context=Depends(get_context),
):
    result = context.qa_chain.invoke({"query": question})
    return {
        "question": question,
        "answer": result.get("result"),
        "sources": format_sources(result.get("source_documents", [])),
    }


@app.get("/healthz")
async def healthcheck():
    return {"status": "ok"}
