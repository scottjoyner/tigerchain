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
| `tigergraph-bootstrap` | One-shot schema + query installation. |
| `minio` | S3-compatible object storage for original documents. |
| `minio-setup` | Creates the configured MinIO bucket. |
| `rag-api` | FastAPI app exposing onboarding, ingestion & question answering endpoints. |
| `vllm` *(optional)* | OpenAI-compatible endpoint powered by [vLLM](https://vllm.ai/) (GPU). |
| `ollama` *(optional)* | Lightweight local model runner for CPU/GPU LLMs. |

The Python service uses LangChain for document parsing, chunking, embedding, and
retrieval. TigerGraph stores each chunk as a `Document` vertex containing the
source metadata, text content, and dense vector embedding. A `SimilarChunks`
GSQL query performs cosine similarity search for retrieval.

## Getting Started

1. Copy the sample environment file and update values as needed (notably the
   TigerGraph license path):

   ```bash
   cp .env.example .env
   # edit .env to point TG_LICENSE_FILE to your enterprise-license.txt
   ```

2. Launch the full stack:

   ```bash
   docker compose up -d --build
   ```

   The bootstrap containers create the TigerGraph schema/queries and the MinIO
   bucket automatically.

3. Ingest the sample documents and run a test query via the CLI:

   ```bash
   make ingest
   make query Q="What does the knowledge graph support?"
   ```

4. Onboard a user and obtain an access token:

   ```bash
   curl -X POST "http://localhost:${API_PORT}/auth/register" \
     -H "Content-Type: application/json" \
     -d '{"email": "user@example.com", "password": "ChangeMe123", "full_name": "Example User"}'

   curl -X POST "http://localhost:${API_PORT}/auth/token" \
     -H "Content-Type: application/x-www-form-urlencoded" \
     -d 'username=user@example.com&password=ChangeMe123'
   ```

   Use the returned bearer token for subsequent requests and store preferences:

   ```bash
   curl -X POST "http://localhost:${API_PORT}/auth/onboarding" \
     -H "Authorization: Bearer ${TOKEN}" \
     -H "Content-Type: application/json" \
     -d '{"categories": ["product", "architecture"], "preferred_agent": "default"}'
   ```

5. Upload documents and issue queries:

   ```bash
   curl -X POST "http://localhost:${API_PORT}/ingest" \
     -H "Authorization: Bearer ${TOKEN}" \
     -F "files=@worker/sample_docs/tigergraph_overview.pdf" \
     -F "categories=product" -F "model_alias=default"

   curl -X POST "http://localhost:${API_PORT}/query" \
     -H "Authorization: Bearer ${TOKEN}" \
     -H "Content-Type: application/json" \
     -d '{"question": "Summarise the onboarding steps", "mode": "parallel", "agents": ["default"]}'
   ```

6. Tail logs or shut down when finished:

   ```bash
   make logs
   make down
   ```

> **Note:** TigerGraph Enterprise requires a valid license. Place the file at the
> path referenced by `TG_LICENSE_FILE` before starting the stack.

## CLI Usage

The `rag-api` container image bundles a Typer CLI for local operations. Use
`docker compose run --rm rag-api python cli.py --help` for full details.

- `make ingest` – embed and upsert all files under `worker/sample_docs` (supports `--owner`, `--agent`, and `--category`).
- `make query Q="..."` – run an ad-hoc question against TigerGraph-stored data with optional `--agent`, `--mode`, and `--category` filters.

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
