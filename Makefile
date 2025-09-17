# =============================================================================
# Makefile wrappers for scripts/dev.sh
# =============================================================================
SHELL := /bin/bash
.DEFAULT_GOAL := help

# Load .env if present (for ports, names, etc.)
ifneq ("$(wildcard .env)","")
  include .env
  export
endif

DEV := scripts/dev.sh

# ---------- Generic helpers ----------
help: ## Show available targets
	@echo "Usage: make <target> [VAR=...]"
	@echo
	@echo "Core:"
	@echo "  make dev-up                   # build & start all services"
	@echo "  make dev-init                 # run TigerGraph init (schema/loaders/queries)"
	@echo "  make dev-ingest [DIR=path] [EXPORT=1]"
	@echo "  make dev-query Q=\"text\" [SCOPE=doc|page|chunk] [TOPK=5]"
	@echo "  make dev-logs [SERVICE=worker]"
	@echo "  make dev-ps                   # compose ps"
	@echo "  make dev-restart [SERVICE=...]"
	@echo "  make dev-rebuild [SERVICE=...] [NO_CACHE=1]"
	@echo "  make dev-down                 # stop and remove containers"
	@echo "  make dev-clean                # down -v + prune"
	@echo "  make dev-status               # quick HTTP health checks"
	@echo "  make dev-exec SERVICE=name [CMD=\"/bin/bash\"]"
	@echo
	@echo "Examples:"
	@echo "  make dev-up"
	@echo "  make dev-init"
	@echo "  make dev-ingest DIR=worker/sample_docs EXPORT=1"
	@echo "  make dev-query Q='cosine similarity' SCOPE=page TOPK=5"
	@echo "  make dev-logs SERVICE=tigergraph"
	@echo "  make dev-rebuild SERVICE=worker NO_CACHE=1"
	@echo "  make dev-exec SERVICE=worker CMD='/bin/bash'"

# ---------- Targets mapped to dev.sh ----------
.PHONY: dev-up dev-init dev-ingest dev-query dev-logs dev-ps dev-restart dev-rebuild dev-down dev-clean dev-status dev-exec

dev-up: ## Build & start services
	$(DEV) up

dev-init: dev-up ## Run TigerGraph initialization script
	$(DEV) init

dev-ingest: ## Ingest (.pdf/.txt/.md) -> MinIO -> embeddings -> TG
	@if [ -n "$(DIR)" ]; then \
	  DIR_ARG="--dir $(DIR)"; \
	else \
	  DIR_ARG="--dir worker/sample_docs"; \
	fi; \
	EXPORT_FLAG=""; \
	if [ "$(EXPORT)" = "1" ]; then EXPORT_FLAG="--export-pagesim"; fi; \
	$(DEV) ingest $$DIR_ARG $$EXPORT_FLAG

dev-query: ## Query via API (fallback to CLI): Q="text" [SCOPE=doc|page|chunk] [TOPK=5]
	@if [ -z "$(Q)" ]; then \
	  echo "Usage: make dev-query Q='text' [SCOPE=doc|page|chunk] [TOPK=5]"; exit 1; \
	fi; \
	SCOPE_ARG="--scope $${SCOPE:-doc}"; \
	TOPK_ARG="--topk $${TOPK:-5}"; \
	$(DEV) query "$(Q)" $$SCOPE_ARG $$TOPK_ARG

dev-logs: ## Tail logs for a service (default: tigergraph)
	$(DEV) logs $(SERVICE)

dev-ps: ## docker compose ps
	$(DEV) ps

dev-restart: ## Restart one service or all if SERVICE is empty
	$(DEV) restart $(SERVICE)

dev-rebuild: ## Rebuild images; pass SERVICE=... and NO_CACHE=1 optionally
	@if [ "$(NO_CACHE)" = "1" ]; then \
	  $(DEV) rebuild $(SERVICE) --no-cache; \
	else \
	  $(DEV) rebuild $(SERVICE); \
	fi

dev-down: ## Stop and remove containers (volumes preserved)
	$(DEV) down

dev-clean: ## Full teardown: down -v + prune
	$(DEV) clean

dev-status: ## Quick health checks
	$(DEV) status

dev-exec: ## Exec into a service: SERVICE=name [CMD='/bin/bash']
	@if [ -z "$(SERVICE)" ]; then \
	  echo "Usage: make dev-exec SERVICE=name [CMD='/bin/bash']"; exit 1; \
	fi; \
	if [ -n "$(CMD)" ]; then \
	  $(DEV) exec $(SERVICE) $(CMD); \
	else \
	  $(DEV) exec $(SERVICE); \
	fi
