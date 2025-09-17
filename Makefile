COMPOSE=docker compose

.PHONY: help up down logs ps ingest query bootstrap

help:
	@echo "Available targets:"
	@echo "  up        - Build and start all services"
	@echo "  down      - Stop and remove containers"
	@echo "  logs      - Follow service logs"
	@echo "  ps        - Show container status"
	@echo "  ingest    - Ingest sample documents via CLI"
	@echo "  query     - Run an ad-hoc query via CLI (use Q=question)"
	@echo "  bootstrap - Re-run TigerGraph bootstrap script"

up:
	$(COMPOSE) up -d --build

bootstrap:
	$(COMPOSE) run --rm tigergraph-bootstrap

ps:
	$(COMPOSE) ps

logs:
	$(COMPOSE) logs -f

down:
	$(COMPOSE) down -v

ingest:
	$(COMPOSE) run --rm rag-api python cli.py ingest /data/sample_docs

query:
	@if [ -z "$(Q)" ]; then \
	echo "Usage: make query Q='your question'"; \
	exit 1; \
	fi
	$(COMPOSE) run --rm rag-api python cli.py query "$(Q)"
