#!/usr/bin/env python3
import os, io, json, csv, argparse, logging
from pathlib import Path
from typing import List, Tuple, Iterable

import requests
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF
import boto3
from botocore.config import Config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# -------------------------
# Config (from env)
# -------------------------
TG_HOST = os.getenv("TG_HOST", "tigergraph")
TG_REST_PORT = int(os.getenv("TG_REST_PORT", "9000"))
TG_USER = os.getenv("TG_USER", "tigergraph")
TG_PASSWORD = os.getenv("TG_PASSWORD", "tigergraph")
TG_GRAPH = os.getenv("TG_GRAPH", "DocGraph")
TG_TOKEN_TTL = int(os.getenv("TG_TOKEN_TTL", "2592000"))
REST_BASE = f"http://{TG_HOST}:{TG_REST_PORT}"

EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBED_DIM = int(os.getenv("EMBED_DIM", "384"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "admin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "changeme-strong")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "docs")


# -------------------------
# TigerGraph Auth
# -------------------------
def _get_secret_and_token(ttl: int = TG_TOKEN_TTL) -> Tuple[str, str]:
    s = requests.post(f"{REST_BASE}/requestsecret", timeout=30)
    s.raise_for_status()
    secret = s.json().get("secret")
    t = requests.post(
        f"{REST_BASE}/requesttoken", params={"secret": secret, "lifetime": ttl}, timeout=30
    )
    t.raise_for_status()
    token = t.json().get("token")
    return secret, token


# -------------------------
# Embeddings
# -------------------------
def _embed_texts(texts: List[str]) -> List[List[float]]:
    model = SentenceTransformer(EMBED_MODEL)
    vecs = model.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=True, normalize_embeddings=False)
    out = []
    for v in vecs:
        v = v.tolist()
        if len(v) != EMBED_DIM:
            raise ValueError(f"Embedding dim mismatch: got {len(v)} expected {EMBED_DIM}")
        out.append(v)
    return out


# -------------------------
# PDF Extraction
# -------------------------
def _extract_text_from_pdf(pdf_path: Path) -> str:
    try:
        doc = fitz.open(pdf_path)
        texts = []
        for page in doc:
            texts.append(page.get_text("text"))
        return "\n".join(texts).strip()
    except Exception as e:
        logging.error("PDF parse failed for %s: %s", pdf_path, e)
        return ""


# -------------------------
# File Iterator
# -------------------------
def _iter_docs(path: Path) -> Iterable[Tuple[str, Path, str]]:
    for p in sorted(path.rglob("*")):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        if ext in {".txt", ".md"}:
            yield (p.stem, p, p.read_text(encoding="utf-8", errors="ignore"))
        elif ext in {".pdf"}:
            yield (p.stem, p, _extract_text_from_pdf(p))


# -------------------------
# MinIO
# -------------------------
def _minio_client():
    return boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
        config=Config(s3={"addressing_style": "path"}),
        region_name="us-east-1",
    )


def _ensure_bucket(s3):
    try:
        s3.head_bucket(Bucket=MINIO_BUCKET)
    except Exception:
        s3.create_bucket(Bucket=MINIO_BUCKET)


def _upload_to_minio(s3, local_path: Path, key: str) -> Tuple[str, str]:
    s3.upload_file(str(local_path), MINIO_BUCKET, key)
    s3_uri = f"s3://{MINIO_BUCKET}/{key}"
    http_url = f"{MINIO_ENDPOINT.rstrip('/')}/{MINIO_BUCKET}/{key}"
    return s3_uri, http_url


# -------------------------
# CSV Writer
# -------------------------
def _write_docs_csv(rows: List[dict], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id","uri","http_url","title","text","embedding"])
        w.writeheader()
        for r in rows:
            r2 = dict(r)
            r2["embedding"] = json.dumps(r2["embedding"])
            w.writerow(r2)


# -------------------------
# TigerGraph Loader
# -------------------------
def load_via_rest_loader(csv_path: Path, token: str):
    files = {"file": (csv_path.name, csv_path.read_bytes(), "text/csv")}
    params = {"graph": TG_GRAPH, "tag": "docs", "filename": csv_path.name}
    headers = {"Authorization": f"Bearer {token}"}
    url = f"{REST_BASE}/ddl/loader?job=load_docs"
    logging.info("POST %s", url)
    r = requests.post(url, headers=headers, params=params, files=files, timeout=120)
    if r.status_code >= 300:
        logging.error("Loader error: %s", r.text[:500])
        r.raise_for_status()
    logging.info("Loader response: %s", r.text[:200])


# -------------------------
# TigerGraph Query
# -------------------------
def _topk_query(vec: List[float], topk: int, token: str):
    url = f"{REST_BASE}/query/{TG_GRAPH}/TopKSimilar"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {"q": vec, "topk": topk}
    r = requests.post(url, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser(description="Embed docs, upload to MinIO, and load to TigerGraph")
    ap.add_argument("--dir", default="/app/sample_docs", help="Directory with .txt/.md/.pdf")
    ap.add_argument("--csv-out", default="/app/docs.csv", help="CSV output path")
    ap.add_argument("--upsert", action="store_true", help="Run loader to upsert")
    ap.add_argument("--query", default=None, help="Text to search (skip ingest)")
    ap.add_argument("--topk", type=int, default=5)
    args = ap.parse_args()

    secret, token = _get_secret_and_token()
    logging.info("Token acquired: %s...", token[:12])

    if args.query:
        logging.info("Embedding ad-hoc query text")
        vec = _embed_texts([args.query])[0]
        res = _topk_query(vec, args.topk, token)
        print(json.dumps(res, indent=2))
        return

    src_dir = Path(args.dir)
    docs = list(_iter_docs(src_dir))
    if not docs:
        logging.warning("No docs found in %s", args.dir)
        return

    s3 = _minio_client()
    _ensure_bucket(s3)

    rows = []
    for stem, path, text in tqdm(docs, desc="Upload+Parse"):
        if not text.strip():
            logging.warning("Empty text extracted for %s; skipping", path.name)
            continue
        key = f"{stem}{path.suffix.lower()}"
        s3_uri, http_url = _upload_to_minio(s3, path, key)

        rows.append({
            "id": stem,
            "uri": s3_uri,
            "http_url": http_url,
            "title": path.name,
            "text": text,
            "embedding": _embed_texts([text])[0],
        })

    csv_path = Path(args.csv_out)
    _write_docs_csv(rows, csv_path)
    logging.info("Wrote %s", csv_path)

    if args.upsert:
        load_via_rest_loader(csv_path, token)
        logging.info("Upsert complete")


if __name__ == "__main__":
    main()
