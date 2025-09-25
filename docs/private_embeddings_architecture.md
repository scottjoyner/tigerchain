# Private + Public Embedding Architecture

The ingestion service now maintains two distinct vector representations for every chunked document: a dense embedding for collaborative scenarios and a bitwise embedding for highly sensitive flows. This document summarises the changes across storage, APIs, and retrieval.

## Storage pipeline

1. **Upload to MinIO** – Source files continue to be streamed to the configured MinIO bucket using the logical key `<doc_id>/<filename>`.
2. **Chunking** – The adaptive chunker produces `ChunkRow` instances tagged with:
   - `submission_id` – stable identifier applied to every chunk in the submission.
   - `embedding_scope` – user-selected value (`public`, `private`, or `both`).
3. **Embedding generation** – The new `DualEmbeddingProvider` wraps the HuggingFace model and produces:
   - Dense vectors for traditional similarity search.
   - Bitwise vectors (thresholded) suitable for private, shareable fingerprints.
4. **Graph persistence** – Both vectors and the submission ID are stored as attributes on the `Document` vertex within TigerGraph. Metadata continues to hold owner/category information for filtering.
5. **Private artifact** – For every document we persist an aggregated JSON payload (`<doc_id>/private_embeddings/<submission_id>.json`) into MinIO containing the ordered bitwise embeddings per chunk. The URI is recorded in both metadata and the SQL metadata store for audit purposes.

## API surface

### Ingestion endpoint (`POST /ingest`)

New form fields provide fine-grained control:

- `embedding_scope`: dictates which embedding mode should be used during retrieval defaults to `both` (store both sets, prefer public for RAG).
- `sharing_preference`: captures the share policy (`public`, `private`, `both`). Stored alongside the document record.
- `submission_id`: optional client-specified tag; if omitted the service generates one automatically.

The response now includes the submission ID, chosen scope, and the MinIO URI for the private embedding artifact.

### Query endpoint (`POST /query`)

Clients can request either embedding set by setting `embedding_scope` (`public` or `private`). The retriever honours this preference by:

1. Embedding the query with the appropriate mode (bitwise vs dense).
2. Routing the request to TigerGraph with the new `embedding_type` parameter.
3. Filtering source documents so only chunks tagged with the requested scope (`public`/`private`/`both`) are returned.

## Metadata persistence

- **SQL metadata** – `document_uploads` includes `submission_id`, `embedding_scope`, `sharing_preference`, and `private_embedding_uri` for UI consumption and audit trails.
- **TigerGraph** – Additional attributes (`private_embedding`, `submission_id`) enrich downstream analytics and allow server-side filtering for scoped retrieval.
- **MinIO** – Stores both the raw files and the JSON snapshot of bitwise embeddings for reproducibility and offline inspection.

## Operational considerations

- `bitwise_threshold` in `Settings` can be tuned per environment to control bit density.
- The JSON artifacts enable rebuilds of private indices without re-embedding entire documents.
- Future work: enforce access policies on the MinIO private embedding prefix and add lifecycle policies for aging out unused submissions.

