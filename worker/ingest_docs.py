#!/usr/bin/env python3
import os, io, json, csv, argparse, logging, math, re, random
from pathlib import Path
from typing import List, Tuple, Iterable, Dict

import requests
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF
import boto3
from botocore.config import Config
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---- Env ----
TG_HOST = os.getenv("TG_HOST", "tigergraph")
TG_REST_PORT = int(os.getenv("TG_REST_PORT", "9000"))
TG_GRAPH = os.getenv("TG_GRAPH", "DocGraph")
TG_TOKEN_TTL = int(os.getenv("TG_TOKEN_TTL", "2592000"))
REST_BASE = f"http://{TG_HOST}:{TG_REST_PORT}"

EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBED_DIM = int(os.getenv("EMBED_DIM", "384"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))
AGG_METHOD = os.getenv("AGG_METHOD", "mean")
MAX_DOC_TEXT_CHARS = int(os.getenv("MAX_DOC_TEXT_CHARS", "200000"))

ENABLE_DOC_EMB = os.getenv("ENABLE_DOC_EMB", "1") == "1"
ENABLE_PAGE_EMB = os.getenv("ENABLE_PAGE_EMB", "1") == "1"
ENABLE_CHUNK_EMB = os.getenv("ENABLE_CHUNK_EMB", "1") == "1"

CHUNK_TOKENS = int(os.getenv("CHUNK_TOKENS", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "64"))

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "admin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "changeme-strong")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "docs")

PAGE_SIM_TOPK = int(os.getenv("PAGE_SIM_TOPK", "10"))
PAGE_SIM_SAMPLE = int(os.getenv("PAGE_SIM_SAMPLE", "0"))
PAGE_SIM_BATCH = int(os.getenv("PAGE_SIM_BATCH", "2048"))

# ---- Helpers ----
def _get_token() -> str:
    s = requests.post(f"{REST_BASE}/requestsecret", timeout=30)
    s.raise_for_status()
    secret = s.json()["secret"]
    t = requests.post(f"{REST_BASE}/requesttoken", params={"secret": secret, "lifetime": TG_TOKEN_TTL}, timeout=30)
    t.raise_for_status()
    return t.json()["token"]

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
    try: s3.head_bucket(Bucket=MINIO_BUCKET)
    except Exception: s3.create_bucket(Bucket=MINIO_BUCKET)

def _upload(s3, path: Path, key: str) -> Tuple[str, str]:
    s3.upload_file(str(path), MINIO_BUCKET, key)
    return f"s3://{MINIO_BUCKET}/{key}", f"{MINIO_ENDPOINT.rstrip('/')}/{MINIO_BUCKET}/{key}"

def _clean(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()

def _tok(s: str) -> List[str]:
    return s.split()

def _chunks(text: str, max_toks: int, overlap: int) -> List[Tuple[str, int]]:
    toks = _tok(text)
    if not toks: return []
    out, i, n = [], 0, len(toks)
    step = max(1, max_toks - overlap)
    while i < n:
        j = min(n, i + max_toks)
        seg = " ".join(toks[i:j]).strip()
        if seg: out.append((seg, len(seg.split())))
        i += step
    return out

def _extract_pdf_pages(p: Path) -> List[str]:
    doc = fitz.open(p)
    arr = []
    for page in doc:
        arr.append(_clean(page.get_text("text") or ""))
    return arr

def _iter_sources(src_dir: Path) -> Iterable[Tuple[str, Path, str, int, List[str]]]:
    for p in sorted(src_dir.rglob("*")):
        if not p.is_file(): continue
        ext = p.suffix.lower()
        if ext == ".pdf":
            pages = _extract_pdf_pages(p)
            full = "\n".join(pages)
            yield (p.stem, p, _clean(full), len(pages), pages)
        elif ext in (".txt", ".md"):
            t = _clean(p.read_text(encoding="utf-8", errors="ignore"))
            yield (p.stem, p, t, 1, [t])

# ---- Embeddings ----
_model = None
def _model_get():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
    return _model

def embed_texts(texts: List[str]) -> np.ndarray:
    m = _model_get()
    arr = m.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=False, normalize_embeddings=False)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.shape[1] != EMBED_DIM:
        raise ValueError(f"Embedding dim mismatch: got {arr.shape[1]} expected {EMBED_DIM}")
    return arr

def agg(vectors: np.ndarray) -> np.ndarray:
    if AGG_METHOD == "mean":
        return vectors.mean(axis=0)
    # extend with median if desired
    return vectors.mean(axis=0)

def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = np.linalg.norm(a, axis=1, keepdims=True) + 1e-9
    b_norm = np.linalg.norm(b, axis=1, keepdims=True) + 1e-9
    return (a @ b.T) / (a_norm * b_norm.T)

# ---- CSV Writers ----
def write_docs_csv(rows: List[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id","uri","http_url","title","text","pages","doc_embedding"])
        w.writeheader()
        for r in rows:
            r = dict(r)
            r["doc_embedding"] = json.dumps(r["doc_embedding"])
            w.writerow(r)

def write_page_levels_csv(rows: List[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["plid","doc_id","page_index","embedding"])
        w.writeheader()
        for r in rows:
            r = dict(r)
            r["embedding"] = json.dumps(r["embedding"])
            w.writerow(r)

def write_chunks_csv(rows: List[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["pid","doc_id","page_index","chunk_index","tokens","text","embedding"])
        w.writeheader()
        for r in rows:
            r = dict(r); r["embedding"] = json.dumps(r["embedding"])
            w.writerow(r)

def write_pagesim_csv(rows: List[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["src_plid","dst_plid","weight"])
        w.writeheader()
        for r in rows: w.writerow(r)

# ---- Loader ----
def load_job(job: str, csv_path: Path, token: str):
    url = f"{REST_BASE}/ddl/loader?job={job}"
    headers = {"Authorization": f"Bearer {token}"}
    files = {"file": (csv_path.name, csv_path.read_bytes(), "text/csv")}
    params = {"graph": TG_GRAPH, "tag": job, "filename": csv_path.name}
    r = requests.post(url, headers=headers, params=params, files=files, timeout=600)
    if r.status_code >= 300:
        logging.error("%s error: %s", job, r.text[:500]); r.raise_for_status()
    logging.info("%s OK: %s", job, r.text[:180])

# ---- Main ingest ----
def main():
    ap = argparse.ArgumentParser(description="Ingest docs with doc/page/chunk embeddings + pagesim export")
    ap.add_argument("--dir", default="/app/sample_docs")
    ap.add_argument("--out-dir", default="/app")
    ap.add_argument("--upsert", action="store_true")
    ap.add_argument("--export-pagesim", action="store_true", help="Compute and write pagesim.csv and load it")
    args = ap.parse_args()

    token = _get_token()
    s3 = _minio_client(); _ensure_bucket(s3)

    sources = list(_iter_sources(Path(args.dir)))
    if not sources: logging.warning("No files"); return

    docs_rows, page_level_rows, chunk_rows = [], [], []

    for doc_id, path, full_text, pages_count, page_texts in tqdm(sources, desc="Upload originals"):
        s3_uri, http_url = _upload(s3, path, f"{doc_id}{path.suffix.lower()}")

        # Per-page→chunks→embeddings
        page_embs = []
        for pi, ptxt in enumerate(page_texts):
            ptxt = _clean(ptxt)
            if not ptxt: continue
            chunks = _chunks(ptxt, CHUNK_TOKENS, CHUNK_OVERLAP) if ENABLE_CHUNK_EMB else [(ptxt, len(_tok(ptxt)))]
            texts = [c[0] for c in chunks]
            tokens = [c[1] for c in chunks]
            if texts:
                vecs = embed_texts(texts)
                if ENABLE_CHUNK_EMB:
                    for ci, (t, tok) in enumerate(zip(texts, tokens)):
                        chunk_rows.append({
                            "pid": f"{doc_id}#p{pi}#c{ci}",
                            "doc_id": doc_id,
                            "page_index": pi,
                            "chunk_index": ci,
                            "tokens": tok,
                            "text": t,
                            "embedding": vecs[ci].tolist()
                        })
                # Page-level aggregate
                if ENABLE_PAGE_EMB:
                    p_emb = agg(vecs).tolist()
                    page_level_rows.append({
                        "plid": f"{doc_id}#p{pi}",
                        "doc_id": doc_id,
                        "page_index": pi,
                        "embedding": p_emb
                    })
                    page_embs.append(np.asarray(p_emb, dtype=np.float32))

        # Document-level aggregate
        if ENABLE_DOC_EMB:
            if page_embs:
                d_emb = agg(np.stack(page_embs, axis=0)).tolist()
            else:
                # fallback: embed full text
                d_emb = embed_texts([full_text])[0].tolist()
        else:
            d_emb = [0.0]*EMBED_DIM

        docs_rows.append({
            "id": doc_id,
            "uri": s3_uri,
            "http_url": http_url,
            "title": path.name,
            "text": full_text[:MAX_DOC_TEXT_CHARS],
            "pages": pages_count,
            "doc_embedding": d_emb
        })

    out_dir = Path(args.out_dir)
    docs_csv   = out_dir / "docs.csv"
    pages_csv  = out_dir / "page_levels.csv"
    chunks_csv = out_dir / "chunks.csv"

    write_docs_csv(docs_rows, docs_csv)
    if ENABLE_PAGE_EMB:  write_page_levels_csv(page_level_rows, pages_csv)
    if ENABLE_CHUNK_EMB: write_chunks_csv(chunk_rows, chunks_csv)

    if args.upsert:
        load_job("load_docs", docs_csv, token)
        if ENABLE_PAGE_EMB and page_level_rows:
            load_job("load_page_levels", pages_csv, token)
        if ENABLE_CHUNK_EMB and chunk_rows:
            load_job("load_chunks", chunks_csv, token)

    # Page→Page similarity export
    if args.export_pagesim and ENABLE_PAGE_EMB and page_level_rows:
        # Build a matrix and compute topK neighbors
        plids = [r["plid"] for r in page_level_rows]
        vecs = np.stack([np.asarray(r["embedding"], dtype=np.float32) for r in page_level_rows], axis=0)

        # optional sampling to keep it light
        idx = list(range(len(plids)))
        if PAGE_SIM_SAMPLE and PAGE_SIM_SAMPLE < len(idx):
            random.seed(42)
            idx = random.sample(idx, PAGE_SIM_SAMPLE)

        pagesim_rows = []
        # compute in blocks to control memory
        for start in tqdm(range(0, len(idx), PAGE_SIM_BATCH), desc="PageSim"):
            block_idx = idx[start:start+PAGE_SIM_BATCH]
            A = vecs[block_idx]                    # (b, d)
            sims = cosine(A, vecs)                 # (b, N)
            # zero out self-sim and pick topK
            for bi, row in enumerate(sims):
                global_i = block_idx[bi]
                row[global_i] = -1.0
                # topK indices
                topk_idx = np.argpartition(-row, PAGE_SIM_TOPK)[:PAGE_SIM_TOPK]
                # sort
                topk_idx = topk_idx[np.argsort(-row[topk_idx])]
                for j in topk_idx:
                    pagesim_rows.append({
                        "src_plid": plids[global_i],
                        "dst_plid": plids[j],
                        "weight": float(row[j])
                    })

        pagesim_csv = out_dir / "pagesim.csv"
        write_pagesim_csv(pagesim_rows, pagesim_csv)
        if args.upsert and pagesim_rows:
            load_job("load_pagesim", pagesim_csv, token)
            logging.info("Loaded %d PageSim edges", len(pagesim_rows))

if __name__ == "__main__":
    main()
