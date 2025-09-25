from __future__ import annotations

import asyncio
from collections import Counter
import json
from pathlib import Path
from typing import List, Optional

import typer

from tigerchain_app.agents import QueryContext
from tigerchain_app.context import build_context
from tigerchain_app.utils.logging import configure_logging, get_logger

app = typer.Typer(help="TigerChain CLI utilities")
logger = get_logger(__name__)


def _validate_mode(value: str) -> str:
    choice = value.lower().strip()
    if choice not in {"sequential", "parallel"}:
        raise typer.BadParameter("mode must be 'sequential' or 'parallel'")
    return choice


def _format_bytes(size: int) -> str:
    if size < 1024:
        return f"{size} B"
    units = ["KB", "MB", "GB", "TB"]
    value = float(size)
    for unit in units:
        value /= 1024.0
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}"
    return f"{value:.1f} PB"


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
    if result.documents:
        typer.echo("\nIngestion summary:")
        chunk_counts = Counter(row.doc_id for row in result.chunks)
        for summary in result.documents:
            chunk_total = chunk_counts.get(summary.doc_id, 0)
            typer.echo(
                f"- {summary.doc_id} · {summary.source_path.name} · {chunk_total} chunk(s) · "
                f"{_format_bytes(summary.file_size_bytes)} · scope={summary.embedding_scope}"
            )
            if summary.categories:
                typer.echo(f"  Categories: {', '.join(summary.categories)}")
            typer.echo(f"  Object URI: {summary.uri}")
            if summary.private_embedding_uri:
                typer.echo(f"  Private embeddings: {summary.private_embedding_uri}")


@app.command()
def query(
    question: str = typer.Argument(..., help="Natural language question to ask"),
    agent: Optional[str] = typer.Option(None, "--agent", help="Agent/model alias to execute the query"),
    mode: str = typer.Option(
        "sequential",
        "--mode",
        help="Execution mode: sequential or parallel",
        show_default=True,
        callback=_validate_mode,
    ),
    owner: Optional[str] = typer.Option(None, "--owner", help="Optional owner identifier to filter results"),
    category: List[str] = typer.Option([], "--category", help="Filter results by categories"),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Return the raw JSON response instead of a formatted summary",
    ),
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
    if json_output:
        typer.echo(json.dumps(output, indent=2))
        return

    typer.echo(f"Question: {question}")
    for item in output["results"]:
        typer.echo(f"\nAgent: {item['agent']}")
        typer.echo(f"Answer: {item['answer'] or 'No answer returned'}")
        sources = item.get("sources") or []
        if sources:
            typer.echo("Sources:")
            for source in sources:
                label = source.get("title") or source.get("source") or "Unknown"
                uri = source.get("http_url") or source.get("uri")
                score = source.get("score")
                details = f"  - {label}"
                if uri:
                    details += f" ({uri})"
                if score is not None:
                    details += f" [score={score}]"
                typer.echo(details)


@app.command("agents")
def list_agents() -> None:
    """List configured agent aliases available for querying."""

    configure_logging()
    context = build_context(force=True)
    names = context.agent_orchestrator.available_agents()
    if not names:
        typer.echo("No agents are currently configured.")
        return
    typer.echo("Available agents:")
    for name in names:
        typer.echo(f"- {name}")


if __name__ == "__main__":
    app()
