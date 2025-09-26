# TigerChain: LangChain + TigerGraph RAG Platform

TigerChain bootstraps a complete retrieval-augmented generation (RAG) stack
around **TigerGraph Enterprise**, **MinIO** object storage, and a modular
LangChain application layer. Documents uploaded as PDFs/Markdown/Plain text are
chunked, embedded, upserted into TigerGraph with vector similarity, and exposed
through both a FastAPI service and a Typer CLI. Optional **vLLM** and
**Ollama** services provide scalable model hosting for local and GPU-backed
language models.

## Platform Components

| Service | Purpose |
| ------- | ------- |
| `tigergraph` | TigerGraph Enterprise database (requires valid license). |
| `minio-tg` | S3-compatible object storage for original documents. |
| `minio-setup` | Creates the configured MinIO bucket. |
| `rag-api` | FastAPI app exposing onboarding, ingestion & question answering endpoints. |
| `vllm` *(optional)* | OpenAI-compatible endpoint powered by [vLLM](https://vllm.ai/) (GPU). |
| `ollama` *(optional)* | Lightweight local model runner for CPU/GPU LLMs. |

The Python service uses LangChain for document parsing, chunking, embedding, and
retrieval. TigerGraph stores each chunk as a `Document` vertex containing the
source metadata, text content, and dense vector embedding. A `SimilarChunks`
GSQL query performs cosine similarity search for retrieval.

## Prerequisites

- **Docker** 24+ with Compose V2 for the containerised stack. The helper
  scripts expect the `docker compose` subcommand to be available.
- **Python** 3.10 or newer if you plan to run the FastAPI service or CLI from a
  local virtual environment instead of the container image.
- **GNU Make** (optional) for the convenience targets defined in the
  repository's `Makefile`.
- **jq** (optional) to parse JSON responses in the shell snippets below.

Copy the sample environment and adjust any secrets or ports before starting:

```bash
cp .env.example .env
# update credentials, ports, and LLM provider settings as needed
```

## Running with Docker Compose (recommended)

1. **Start the services:**

   ```bash
   docker compose up -d --build
   ```

2. **Initialize TigerGraph** once the database is reachable. The helper script
   waits for REST++ and installs the schema, loaders, and queries:

   ```bash
   make dev-init
   ```

3. **Seed sample content (optional):**

   ```bash
   make dev-ingest
   make dev-query Q="What does the knowledge graph support?"
   ```

4. **Monitor or stop the stack** using the provided wrappers:

   ```bash
   make dev-logs SERVICE=rag-api   # follow API logs
   make dev-status                 # quick health checks
   make dev-down                   # stop containers (volumes preserved)
   make dev-clean                  # full teardown with volume prune
   ```

The `scripts/dev.sh` helpers that back these targets encapsulate common Docker
Compose workflows such as rebuilds, exec, and health probing so you do not need
to memorise lengthy commands.

## Running the API locally (without the container)

You can run the FastAPI worker in a local Python environment while keeping
TigerGraph and MinIO inside containers for consistency.

1. **Start infrastructure services** (TigerGraph, MinIO, bootstrap helpers):

   ```bash
   docker compose up -d tigergraph minio-tg minio-setup
   make dev-init
   ```

2. **Create and activate a virtual environment:**

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r worker/requirements.txt
   ```

3. **Expose configuration to the application.** The FastAPI service reads the
   same `.env` file used by Docker. Ensure the relevant TigerGraph and MinIO
   hosts point at the container endpoints (defaults already match):

   ```bash
   export $(grep -v '^#' .env | xargs)  # or use direnv/uvicorn --env-file
   ```

4. **Run the API with auto-reload from the repository root:**

   ```bash
   cd worker
   uvicorn server:app --reload --env-file ../.env --host 0.0.0.0 --port 8000
   ```

   The service will rebuild the LangChain context on startup using the TigerGraph
   REST interface and MinIO credentials provided in `.env`.

5. **Use the CLI locally** against the same backing services:

   ```bash
   python cli.py ingest --dir sample_docs
   python cli.py query "List the supported embedding scopes"
   ```

When you are finished, stop the infrastructure containers with
`docker compose down` (or `make dev-down`).

## User onboarding guide

Once the API is running—either via Docker Compose or a local Python process—you
can follow the workflow below to onboard new users and begin interacting with
the RAG system.

1. **Register an account:**

   ```bash
   curl -X POST "http://localhost:${API_PORT:-8000}/auth/register" \
     -H "Content-Type: application/json" \
     -d '{"email":"user@example.com","password":"ChangeMe123","full_name":"Example User"}'
   ```

2. **Authenticate and capture the bearer token:**

   ```bash
   TOKEN=$(curl -s -X POST "http://localhost:${API_PORT:-8000}/auth/token" \
     -H "Content-Type: application/x-www-form-urlencoded" \
     -d 'username=user@example.com&password=ChangeMe123' | jq -r .access_token)
   ```

3. **Store onboarding preferences** (categories, default agent selection, and
   optional metadata) so the orchestrator can tailor retrieval and LLM routing:

   ```bash
   curl -X POST "http://localhost:${API_PORT:-8000}/auth/onboarding" \
     -H "Authorization: Bearer ${TOKEN}" \
     -H "Content-Type: application/json" \
     -d '{"categories":["product","architecture"],"preferred_agent":"default"}'
   ```

4. **Upload documents for ingestion.** Files are written to MinIO, chunked,
   embedded, and persisted in TigerGraph with both dense and bitwise vectors:

   ```bash
   curl -X POST "http://localhost:${API_PORT:-8000}/ingest" \
     -H "Authorization: Bearer ${TOKEN}" \
     -F "files=@worker/sample_docs/tigergraph_overview.pdf" \
     -F "categories=product" \
     -F "model_alias=default" \
     -F "embedding_scope=both" \
     -F "sharing_preference=both"
   ```

5. **Ask questions against the knowledge base** using sequential or parallel
   agent execution modes:

   ```bash
   curl -X POST "http://localhost:${API_PORT:-8000}/query" \
     -H "Authorization: Bearer ${TOKEN}" \
     -H "Content-Type: application/json" \
     -d '{"question":"Summarise the onboarding steps","mode":"parallel","agents":["default"]}'
   ```

6. **Review uploads and monitor activity** via supporting endpoints:

   ```bash
   curl -H "Authorization: Bearer ${TOKEN}" http://localhost:${API_PORT:-8000}/documents
   curl http://localhost:${API_PORT:-8000}/agents
   curl http://localhost:${API_PORT:-8000}/healthz
   ```

For day-to-day operations, the `/auth/onboarding` preferences drive which
categories and agents are used by default, while ingestion requests can override
those values per upload. The returned metadata from ingestion includes the
MinIO object URI, submission identifiers, and embedding visibility so clients
can build audit trails or user-facing dashboards.

## CLI Usage

The `rag-api` container image bundles a Typer CLI for local operations. Use
`docker compose run --rm rag-api python cli.py --help` for full details.

- `make ingest` – embed and upsert all files under `worker/sample_docs` (supports `--owner`, `--agent`, and `--category`).
- `make query Q="..."` – run an ad-hoc question against TigerGraph-stored data with optional `--agent`, `--mode`, and `--category` filters.
- `docker compose run --rm rag-api python cli.py agents` – inspect the configured agent registry.

The CLI now prints a concise ingestion summary highlighting chunk counts, file sizes, and private embedding artifacts for each processed document. The `query` command also accepts `--json` to emit machine-readable responses for automation workflows.

## API Endpoints

- `POST /auth/register` – Register a new user account.
- `POST /auth/token` – Obtain a JWT access token via username/password.
- `POST /auth/onboarding` – Store the user's preferred agent and categories.
- `POST /ingest` – Accepts one or more file uploads with optional metadata,
  parses + ingests them into TigerGraph, and returns ingestion details.
  - New form fields: `embedding_scope` (`public`/`private`/`both`), `sharing_preference`
    (`public`/`private`/`both`), and optional `submission_id` to tag the upload.
  - Response includes the `submission_id`, embedding scope, and the MinIO URI of
    the private bitwise embedding artifact.
- `POST /query` – Accepts JSON `{ "question": "...", "mode": "sequential"|"parallel", ... }`
  and returns agent-specific answers plus source metadata.
  - Optional `embedding_scope` flag allows users to switch between dense
    (public) and bitwise (private) retrieval modes on demand.
- `GET /agents` – Lists configured model agents.
- `GET /documents` – Returns the authenticated user's recent document uploads.
- `GET /healthz` – Basic liveness check.

Both endpoints rely on the shared application context which:

1. Bootstraps TigerGraph using the bundled `gsql/schema.gsql` and
   `gsql/queries.gsql` scripts.
2. Loads a HuggingFace embedding model (`EMBED_MODEL`) wrapped by the
   `DualEmbeddingProvider` to generate dense + bitwise vectors per chunk.
3. Configures MinIO for document storage.
4. Creates a LangChain retriever using the TigerGraph `SimilarChunks` query.
5. Instantiates the configured LLM provider and builds a RetrievalQA chain.

## LLM & Embedding Providers

Configure providers through environment variables (`.env` or Docker Compose):

- `LLM_PROVIDER` – `ollama`, `vllm`, or `openai`.
- `LLM_MODEL` – Model name used by the provider.
- `OLLAMA_BASE_URL` – URL of the Ollama service (default `http://ollama:11434`).
- `VLLM_API_BASE` – OpenAI-compatible base URL exposed by the vLLM container.
- `OPENAI_API_KEY`, `OPENAI_API_BASE` – Set when targeting hosted OpenAI-compatible services.

To run local models:

- Start Ollama or vLLM via `docker compose --profile llm up -d ollama` or
  `docker compose --profile llm up -d vllm`.
- Update `.env` to point `LLM_PROVIDER`/`LLM_MODEL` at the desired backend.
- Preload Ollama models with `docker compose exec ollama ollama pull llama2` etc.

Embeddings default to `sentence-transformers/all-MiniLM-L6-v2`. Override with
`EMBED_MODEL`, `EMBED_DEVICE`, and `EMBED_BATCH_SIZE` to match hardware. The
`BITWISE_THRESHOLD` setting controls how dense vectors are binarised for private
embedding generation.

## Testing & Coverage

Unit tests live under `worker/tests` and can be executed locally without the
full container stack:

```bash
cd worker
pytest
```

To generate coverage metrics, install `pytest-cov` and run:

```bash
pytest --cov=tigerchain_app --cov=cli --cov-report=term-missing
```

## Authentication & Persistence

Authentication, onboarding preferences, and document metadata are persisted in a
SQL database configured via the following environment variables:

- `DATABASE_URL` – defaults to `sqlite:///./tigerchain.db` inside the API container.
- `JWT_SECRET_KEY`, `JWT_ALGORITHM`, `ACCESS_TOKEN_EXPIRE_MINUTES` – tune token issuance settings.

Override these values in `.env` or the Docker Compose file to integrate with an
external PostgreSQL/MySQL instance. When running against SQLite the application
handles migrations automatically at startup.

## Development Notes

- The application automatically re-applies GSQL schema/query files at startup.
  Adjust the scripts in `gsql/` to evolve the data model.
- Uploaded documents are copied to MinIO under `<doc-id>/<filename>`, and
  temporary files are cleaned up after ingestion.
- `worker/tigerchain_app` contains modular components for embeddings,
  ingestion, TigerGraph interactions, authentication, and multi-agent orchestration.
- Extend the RAG flow by adding custom retrievers, evaluation routines, or
  alternative storage backends following the same interfaces.

## Troubleshooting

- Ensure the TigerGraph license file is readable by Docker. The compose file
  mounts it read-only at `/home/tigergraph/enterprise-license.txt`.
- If embeddings fail to load due to hardware constraints, switch `EMBED_DEVICE`
  to `cpu` or choose a smaller HuggingFace model.
- Use `make bootstrap` to re-run the TigerGraph schema deployment if the graph
  was reset or upgraded.
