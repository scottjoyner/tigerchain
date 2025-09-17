#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# TigerGraph + MinIO + Worker (FastAPI) Dev Helper
# -----------------------------------------------------------------------------
# Usage:
#   scripts/dev.sh up
#   scripts/dev.sh init
#   scripts/dev.sh ingest [--dir worker/sample_docs] [--export-pagesim]
#   scripts/dev.sh query "your text" [--scope doc|page|chunk] [--topk 5]
#   scripts/dev.sh logs [service]
#   scripts/dev.sh ps
#   scripts/dev.sh restart [service]
#   scripts/dev.sh rebuild [service] [--no-cache]
#   scripts/dev.sh down
#   scripts/dev.sh clean
#   scripts/dev.sh status
#   scripts/dev.sh exec [service] [cmd...]
# =============================================================================

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Colors
BOLD="\033[1m"; GREEN="\033[32m"; YELLOW="\033[33m"; RED="\033[31m"; NC="\033[0m"

COMPOSE="docker compose"

die() { echo -e "${RED}Error:${NC} $*" >&2; exit 1; }
say() { echo -e "${GREEN}▶${NC} $*"; }
warn(){ echo -e "${YELLOW}⚠${NC} $*"; }

need() { command -v "$1" >/dev/null 2>&1 || die "Missing dependency: $1"; }

load_env() {
  if [[ -f .env ]]; then
    # Source .env safely; this supports inline comments like VAR=val  # comment
    set -a
    # shellcheck disable=SC1091
    . ./.env
    set +a
  else
    warn "No .env found. Using defaults from docker-compose.yml and worker/.env.example."
  fi
  : "${API_PORT:=8000}"
  : "${TG_REST_PORT:=9000}"
  : "${TG_GSQL_PORT:=14240}"
  : "${MINIO_PORT:=9000}"
  : "${MINIO_CONSOLE_PORT:=9001}"
}


ensure_tools() {
  need docker
  # docker compose v2 (built-in) recommended
  if ! $COMPOSE version >/dev/null 2>&1; then
    die "'docker compose' not found; install Docker Desktop or Compose v2."
  fi
}

help() {
  cat <<'EOF'
Usage:
  scripts/dev.sh <command> [args]

Commands:
  up                       Build images and start all services in the background
  init                     Run TigerGraph initialization (schema, loaders, queries)
  ingest [opts]            Ingest .pdf/.txt/.md → MinIO → embeddings → TigerGraph
                           Options:
                             --dir <path>         (default: worker/sample_docs)
                             --export-pagesim     (also compute PageSim edges)
  query "<text>" [opts]    Search via worker API (fallback to CLI)
                           Options:
                             --scope doc|page|chunk   (default: doc)
                             --topk N                 (default: 5)
  logs [service]           Tail logs (default: tigergraph)
  ps                       docker compose ps
  restart [service]        Restart one service or all if omitted
  rebuild [service]        Build updated images (all or one service); add --no-cache
  down                     Stop and remove containers (keep volumes)
  clean                    Full teardown: down -v + prune dangling images/volumes
  status                   Quick health checks for ports and endpoints
  exec <service> [cmd..]   Exec into a service container (default cmd: /bin/bash)

Examples:
  scripts/dev.sh up
  scripts/dev.sh init
  scripts/dev.sh ingest --dir worker/sample_docs --export-pagesim
  scripts/dev.sh query "cosine similarity" --scope chunk --topk 5
  scripts/dev.sh logs worker
  scripts/dev.sh rebuild worker --no-cache
  scripts/dev.sh restart tigergraph
  scripts/dev.sh down
  scripts/dev.sh clean
EOF
}

cmd_up() {
  say "Building and starting services..."
  $COMPOSE up -d --build
  say "Services are up. Try: scripts/dev.sh init"
}

cmd_init() {
  say "Ensuring TigerGraph is running..."
  $COMPOSE up -d "${TG_CONTAINER_NAME:-tigergraph}"

  say "Waiting for REST++ on http://localhost:${TG_REST_PORT:-9000}/echo ..."
  for i in {1..60}; do
    if curl -fsS "http://localhost:${TG_REST_PORT:-9000}/echo" >/dev/null 2>&1; then
      break
    fi
    sleep 2
  done

  say "Initializing TigerGraph graph/schema/loaders/queries..."
  $COMPOSE exec -T "${TG_CONTAINER_NAME:-tigergraph}" bash -lc "/opt/scripts/init_tigergraph.sh"
  say "TigerGraph initialized."
}


cmd_ingest() {
  local dir="worker/sample_docs"
  local export_pagesim=0
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --dir) dir="$2"; shift 2;;
      --export-pagesim) export_pagesim=1; shift;;
      *) die "Unknown ingest option: $1";;
    esac
  done
  say "Running ingestion from: ${dir}"
  if [[ $export_pagesim -eq 1 ]]; then
    $COMPOSE run --rm worker python /app/ingest_docs.py --dir "/app/${dir#worker/}" --upsert --export-pagesim
  else
    $COMPOSE run --rm worker python /app/ingest_docs.py --dir "/app/${dir#worker/}" --upsert
  fi
  say "Ingestion complete."
}

cmd_query() {
  [[ $# -ge 1 ]] || die 'Usage: scripts/dev.sh query "your text" [--scope doc|page|chunk] [--topk N]'
  local text="$1"; shift
  local scope="doc"; local topk="5"
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --scope) scope="$2"; shift 2;;
      --topk)  topk="$2"; shift 2;;
      *) die "Unknown query option: $1";;
    esac
  done
  say "Querying scope=${scope} topk=${topk}"

  # Prefer API if reachable; fallback to CLI
  if curl -fsS "http://localhost:${API_PORT}/openapi.json" >/dev/null 2>&1; then
    curl -sS -X POST "http://localhost:${API_PORT}/search" \
      -H 'Content-Type: application/json' \
      -d "{\"query\":\"${text}\",\"topk\":${topk},\"scope\":\"${scope}\"}" | jq .
  else
    warn "API not reachable; falling back to CLI query via worker container."
    # Map scope to queries: doc->TopKSimilarDocs, page->TopKSimilarPages, chunk->TopKSimilarChunks
    # The CLI tool calls page query by default; we’ll run it via REST using ingest script.
    $COMPOSE run --rm worker python - <<'PYCODE'
import os, sys, json, requests
from sentence_transformers import SentenceTransformer

query = os.environ.get("QTEXT")
scope = os.environ.get("QSCOPE","doc")
topk  = int(os.environ.get("QTOPK","5"))
tg_host = "tigergraph"; rest_port = 9000; graph = os.environ.get("TG_GRAPH","DocGraph")
REST_BASE = f"http://{tg_host}:{rest_port}"
m = SentenceTransformer(os.environ.get("EMBED_MODEL","sentence-transformers/all-MiniLM-L6-v2"))
vec = m.encode([query], normalize_embeddings=False)[0].tolist()
secret = requests.post(f"{REST_BASE}/requestsecret").json()["secret"]
token  = requests.post(f"{REST_BASE}/requesttoken", params={"secret": secret, "lifetime": 3600}).json()["token"]
qname = {"doc":"TopKSimilarDocs","page":"TopKSimilarPages","chunk":"TopKSimilarChunks"}[scope]
res = requests.post(f"{REST_BASE}/query/{graph}/{qname}", headers={"Authorization": f"Bearer {token}"}, json={"q": vec, "topk": topk})
res.raise_for_status()
print(json.dumps(res.json(), indent=2))
PYCODE
  fi
}

cmd_logs() {
  local svc="${1:-tigergraph}"
  say "Tailing logs for: ${svc} (Ctrl-C to stop)"
  $COMPOSE logs -f "$svc"
}

cmd_ps() {
  $COMPOSE ps
}

cmd_restart() {
  local svc="${1:-}"
  if [[ -n "$svc" ]]; then
    say "Restarting service: $svc"
    $COMPOSE restart "$svc"
  else
    say "Restarting all services"
    $COMPOSE restart
  fi
}

cmd_rebuild() {
  local svc=""
  local no_cache=0
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --no-cache) no_cache=1; shift;;
      *) svc="$1"; shift;;
    esac
  done
  local args=(build)
  [[ $no_cache -eq 1 ]] && args+=(--no-cache)
  if [[ -n "$svc" ]]; then
    say "Rebuilding service: $svc"
    $COMPOSE "${args[@]}" "$svc"
  else
    say "Rebuilding all services"
    $COMPOSE "${args[@]}"
  fi
  say "Restarting updated containers..."
  $COMPOSE up -d
}

cmd_down() {
  say "Stopping and removing containers (volumes preserved)"
  $COMPOSE down
}

cmd_clean() {
  say "Full teardown: down -v"
  $COMPOSE down -v || true
  say "Pruning dangling images/volumes (safe)"
  docker image prune -f >/dev/null || true
  docker volume prune -f >/dev/null || true
  say "Clean completed."
}

cmd_status() {
  say "docker compose ps"
  $COMPOSE ps
  echo
  say "HTTP checks:"
  printf "  TigerGraph REST:  "; curl -fsS "http://localhost:${TG_REST_PORT}/echo" >/dev/null && echo "OK" || echo "N/A"
  printf "  TG Studio:        "; curl -fsS "http://localhost:${TG_GSQL_PORT}" >/dev/null && echo "OK" || echo "N/A"
  printf "  MinIO API:        "; curl -fsS "http://localhost:${MINIO_PORT}/minio/health/ready" >/dev/null && echo "OK" || echo "N/A"
  printf "  Worker API:       "; curl -fsS "http://localhost:${API_PORT}/openapi.json" >/dev/null && echo "OK" || echo "N/A"
}

cmd_exec() {
  [[ $# -ge 1 ]] || die "Usage: scripts/dev.sh exec <service> [cmd...]"
  local svc="$1"; shift
  if [[ $# -gt 0 ]]; then
    $COMPOSE exec "$svc" "$@"
  else
    $COMPOSE exec "$svc" /bin/bash
  fi
}

main() {
  ensure_tools
  load_env

  local cmd="${1:-help}"; shift || true
  case "$cmd" in
    up)         cmd_up "$@";;
    init)       cmd_init "$@";;
    ingest)     cmd_ingest "$@";;
    query)      cmd_query "$@";;
    logs)       cmd_logs "$@";;
    ps)         cmd_ps "$@";;
    restart)    cmd_restart "$@";;
    rebuild)    cmd_rebuild "$@";;
    down)       cmd_down "$@";;
    clean)      cmd_clean "$@";;
    status)     cmd_status "$@";;
    exec)       cmd_exec "$@";;
    help|-h|--help) help;;
    *)          help; echo; die "Unknown command: $cmd";;
  esac
}

main "$@"
