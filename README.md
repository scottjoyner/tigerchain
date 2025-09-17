# TigerGraph + MinIO + Vector Embeddings (Containerized Example)

'''
tigergraph-vector-embeddings-minio/
├── README.md
├── .env.example
├── docker-compose.yml
├── Makefile
├── licenses/
│   └── .gitkeep
├── gsql/
│   ├── schema.gsql
│   ├── loading_jobs.gsql
│   └── queries.gsql
├── scripts/
│   ├── wait_for_tg.sh
│   ├── init_tigergraph.sh
│   └── init_minio.sh
└── worker/
    ├── Dockerfile
    ├── requirements.txt
    ├── ingest_docs.py
    ├── .env.example
    └── sample_docs/
        ├── example1.txt
        └── example2.md
'''

Production-ready Docker Compose stack for **TigerGraph Enterprise** + **MinIO** object storage + a **Python worker**
that ingests `.txt/.md/.pdf` documents, generates embeddings (Sentence-Transformers), uploads originals to MinIO,
and upserts `Document` vertices (with `embedding`) into TigerGraph. Includes GSQL schema, loading job, and a
cosine Top-K similarity query.

> ⚠️ You need a valid **TigerGraph Enterprise** license. Put it at `licenses/enterprise-license.txt`.

## Quick Start
```bash
cp .env.example .env
cp worker/.env.example worker/.env
# place TigerGraph license at licenses/enterprise-license.txt

docker compose up -d --build
make init
make ingest
make query Q="graph databases with vector search"
