from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import List, Optional

import typer

from tigerchain_app.agents import QueryContext
from tigerchain_app.context import build_context
from tigerchain_app.utils.logging import configure_logging, get_logger

app = typer.Typer(help="TigerChain CLI utilities")
logger = get_logger(__name__)


@app.command()
def ingest(
    path: Path = typer.Argument(..., exists=True, help="Path to a document file or directory"),
    owner: Optional[str] = typer.Option(None, "--owner", help="Optional owner identifier to scope the document"),
    category: List[str] = typer.Option([], "--category", help="Categories associated with the document"),
    agent: Optional[str] = typer.Option(None, "--agent", help="Agent/model alias to tag the ingestion with"),
) -> None:
    """Parse, embed and upsert documents into TigerGraph."""

    configure_logging()
    context = build_context(force=True)
    pipeline = context.pipeline

    if path.is_dir():
        logger.info("Ingesting directory %s", path)
        result = pipeline.ingest_directory(path, owner_id=owner, categories=category, model_alias=agent)
    else:
        logger.info("Ingesting file %s", path)
        result = pipeline.ingest_files([path], owner_id=owner, categories=category, model_alias=agent)
    typer.echo(f"Ingested {len(result.chunks)} chunks from {len(result.documents)} document(s)")


@app.command()
def query(
    question: str = typer.Argument(..., help="Natural language question to ask"),
    agent: Optional[str] = typer.Option(None, "--agent", help="Agent/model alias to execute the query"),
    mode: str = typer.Option("sequential", "--mode", help="Execution mode: sequential or parallel"),
    owner: Optional[str] = typer.Option(None, "--owner", help="Optional owner identifier to filter results"),
    category: List[str] = typer.Option([], "--category", help="Filter results by categories"),
) -> None:
    """Execute retrieval-augmented generation over TigerGraph content."""

    configure_logging()
    context = build_context(force=True)
    orchestrator = context.agent_orchestrator
    agent_names = [agent] if agent else None
    results = asyncio.run(
        orchestrator.run_query(
            question=question,
            agent_names=agent_names,
            query_context=QueryContext(owner_id=owner, categories=category, model_alias=agent),
            mode=mode,
        )
    )
    output = {
        "question": question,
        "mode": mode,
        "results": [
            {
                "agent": res.agent,
                "answer": res.answer,
                "sources": res.sources,
            }
            for res in results
        ],
    }
    if output["results"]:
        output["answer"] = output["results"][0]["answer"]
    typer.echo(json.dumps(output, indent=2))


if __name__ == "__main__":
    app()
