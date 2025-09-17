from __future__ import annotations

import json
from pathlib import Path

import typer

from tigerchain_app.context import build_context
from tigerchain_app.rag.chain import format_sources
from tigerchain_app.utils.logging import configure_logging, get_logger

app = typer.Typer(help="TigerChain CLI utilities")
logger = get_logger(__name__)


@app.command()
def ingest(path: Path = typer.Argument(..., exists=True, help="Path to a document file or directory")) -> None:
    """Parse, embed and upsert documents into TigerGraph."""

    configure_logging()
    context = build_context(force=True)
    pipeline = context.pipeline

    if path.is_dir():
        logger.info("Ingesting directory %s", path)
        pipeline.ingest_directory(path)
    else:
        logger.info("Ingesting file %s", path)
        pipeline.ingest_files([path])


@app.command()
def query(question: str = typer.Argument(..., help="Natural language question to ask")) -> None:
    """Execute retrieval-augmented generation over TigerGraph content."""

    configure_logging()
    context = build_context(force=True)
    qa_chain = context.qa_chain

    result = qa_chain.invoke({"query": question})
    output = {
        "question": question,
        "answer": result.get("result"),
        "sources": format_sources(result.get("source_documents", [])),
    }
    typer.echo(json.dumps(output, indent=2))


if __name__ == "__main__":
    app()
