# worker/api.py
from fastapi import FastAPI, Query
from pydantic import BaseModel
import os, requests, json
from typing import Literal, Optional, List
from sentence_transformers import SentenceTransformer

TG_HOST = os.getenv("TG_HOST", "tigergraph")
TG_REST_PORT = int(os.getenv("TG_REST_PORT", "9000"))
TG_GRAPH = os.getenv("TG_GRAPH", "DocGraph")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))

REST_BASE = f"http://{TG_HOST}:{TG_REST_PORT}"
_model = SentenceTransformer(EMBED_MODEL)

def _token():
    s = requests.post(f"{REST_BASE}/requestsecret", timeout=30).json()["secret"]
    t = requests.post(f"{REST_BASE}/requesttoken", params={"secret": s, "lifetime": 3600}, timeout=30).json()["token"]
    return t

app = FastAPI(title="Vector Store API")

class SearchReq(BaseModel):
    query: str
    topk: int = 5
    scope: Literal["doc","page","chunk"] = "doc"

@app.post("/search")
def search(req: SearchReq):
    vec = _model.encode([req.query], batch_size=BATCH_SIZE, normalize_embeddings=False)[0].tolist()
    token = _token()
    if req.scope == "doc":
        url = f"{REST_BASE}/query/{TG_GRAPH}/TopKSimilarDocs"
    elif req.scope == "page":
        url = f"{REST_BASE}/query/{TG_GRAPH}/TopKSimilarPages"
    else:
        url = f"{REST_BASE}/query/{TG_GRAPH}/TopKSimilarChunks"
    r = requests.post(url, headers={"Authorization": f"Bearer {token}"}, json={"q": vec, "topk": req.topk}, timeout=120)
    r.raise_for_status()
    return r.json()
