from fastapi import FastAPI, Query, HTTPException, Header
from typing import Optional, List, Dict, Any, Tuple
import json, os, re
from functools import lru_cache

import numpy as np
import httpx
from openai import OpenAI

app = FastAPI(title="Quran Proxy API (Hafs + Semantic + Audio)")

# =====================
# Env / Settings
# =====================
DATA_PATH = os.environ.get("HAFS_DATA_PATH", "hafsData_v2-0.json")
INDEX_DIR = os.environ.get("INDEX_DIR", "index")

API_KEY = os.environ.get("PROXY_API_KEY", "")
OPENAI_MODEL = os.environ.get("EMBED_MODEL", "text-embedding-3-small")
OPENAI_DIMS = os.environ.get("EMBED_DIMS")
OPENAI_DIMS = int(OPENAI_DIMS) if OPENAI_DIMS else None

QURAN_API_BASE = os.environ.get("QURAN_API_BASE", "https://api.quran.com/api/v4")
QURAN_VERSE_AUDIO_BASE = os.environ.get("QURAN_VERSE_AUDIO_BASE", "https://verses.quran.foundation/")

AR_NUM = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")


def norm_q(s: str) -> str:
    s = (s or "").strip().translate(AR_NUM)
    s = re.sub(r"\s+", " ", s)
    return s


def require_api_key(x_api_key: Optional[str]):
    if not API_KEY:
        raise HTTPException(status_code=500, detail="Server PROXY_API_KEY is not set")
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")


def match_word(text: str, token: str) -> bool:
    pat = r'(^|[^\w])' + re.escape(token) + r'([^\w]|$)'
    return re.search(pat, text, flags=re.UNICODE) is not None


def extract_topic_name(s: str) -> str:
    x = norm_q(s)
    patterns = [
        r'^(?:آيات|ايات|آية|اية)\s+عن\s+',
        r'^عن\s+',
        r'^(?:موضوع|الموضوع)\s+',
        r'^(?:بالمعنى|بمعنى)\s+',
        r'^(?:ابحث|دور|عايز|أريد|اريد)\s+(?:عن\s+)?'
    ]
    for p in patterns:
        x2 = re.sub(p, "", x)
        if x2 != x:
            x = x2.strip()
    x = re.sub(r'[؟\?\!\.\,\؛\:\-]+$', "", x).strip()
    x = re.sub(r'\s+(?:في\s+القرآن|بالقرآن|من\s+القرآن)$', "", x).strip()
    return x


# =====================
# Load Hafs records
# =====================
if not os.path.exists(DATA_PATH):
    raise RuntimeError(f"Data file not found: {DATA_PATH}")

with open(DATA_PATH, "r", encoding="utf-8") as f:
    raw = json.load(f)

records: List[Dict[str, Any]] = raw if isinstance(raw, list) else raw.get("data", [])
if not isinstance(records, list):
    raise RuntimeError("Unexpected JSON structure: expected list of records")

by_key: Dict[str, Dict[str, Any]] = {}
for r in records:
    s = r.get("sura_no")
    a = r.get("aya_no")
    if isinstance(s, int) and isinstance(a, int):
        by_key[f"{s}:{a}"] = r


# =====================
# Load Semantic Index
# =====================
SEM_READY = False
emb_matrix_f32: Optional[np.ndarray] = None  # float32 normalized (for fast dot)
verse_keys: List[str] = []
meta: Dict[str, Any] = {}

try:
    emb_path = os.path.join(INDEX_DIR, "embeddings_norm_f16.npy")
    vk_path = os.path.join(INDEX_DIR, "verse_keys.json")
    meta_path = os.path.join(INDEX_DIR, "meta.json")

    if os.path.exists(emb_path) and os.path.exists(vk_path):
        emb_f16 = np.load(emb_path)  # float16 normalized
        emb_matrix_f32 = emb_f16.astype(np.float32, copy=False)  # convert once
        with open(vk_path, "r", encoding="utf-8") as f:
            verse_keys = json.load(f)
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        SEM_READY = True
except Exception:
    SEM_READY = False
    emb_matrix_f32 = None
    verse_keys = []
    meta = {}


# =====================
# OpenAI client (for query embedding only)
# =====================
_openai_client = None


def get_openai_client():
    global _openai_client
    if _openai_client is None:
        if not os.environ.get("OPENAI_API_KEY"):
            raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not set on server")
        _openai_client = OpenAI()
    return _openai_client


@lru_cache(maxsize=512)
def _embed_query_cached(q: str) -> Tuple[float, ...]:
    client = get_openai_client()
    kwargs = {"model": OPENAI_MODEL, "input": [q]}
    if OPENAI_DIMS:
        kwargs["dimensions"] = OPENAI_DIMS
    resp = client.embeddings.create(**kwargs)
    v = np.array(resp.data[0].embedding, dtype=np.float32)
    v = v / (np.linalg.norm(v) + 1e-12)
    return tuple(float(x) for x in v)


def embed_query(q: str) -> np.ndarray:
    # numpy array from cached tuple
    return np.array(_embed_query_cached(q), dtype=np.float32)


def semantic_topk(q: str, k: int) -> List[Tuple[str, float]]:
    if not SEM_READY or emb_matrix_f32 is None or not verse_keys:
        raise HTTPException(status_code=503, detail="Semantic index is not ready on server")
    v = embed_query(q)
    sims = emb_matrix_f32 @ v  # cosine similarity (normalized)
    idx = np.argsort(-sims)[:k]
    out: List[Tuple[str, float]] = []
    for i in idx:
        ii = int(i)
        out.append((verse_keys[ii], float(sims[ii])))
    return out


# =====================
# Endpoints
# =====================
@app.get("/v1/health")
def health():
    return {
        "ok": True,
        "records": len(records),
        "semantic_ready": SEM_READY,
        "semantic_meta": meta,
        "semantic_count": len(verse_keys)
    }


@app.get("/v1/verse/{verse_key}")
def get_verse(verse_key: str, x_api_key: Optional[str] = Header(None)):
    require_api_key(x_api_key)
    vk = norm_q(verse_key)
    if ":" not in vk:
        raise HTTPException(status_code=400, detail="Use verse_key like 2:255")
    rec = by_key.get(vk)
    if not rec:
        raise HTTPException(status_code=404, detail="Not found")
    out = dict(rec)
    out["verse_key"] = vk
    return out


@app.get("/v1/search")
def search(
    q: str = Query(..., min_length=1),
    match: str = Query("word", pattern="^(word|phrase)$"),
    offset: int = Query(0, ge=0),
    limit: int = Query(25, ge=1, le=200),
    include_text: bool = Query(True),
    x_api_key: Optional[str] = Header(None)
):
    require_api_key(x_api_key)

    qn = norm_q(q)
    if not qn:
        raise HTTPException(status_code=400, detail="Empty query")

    results = []
    for r in records:
        t = r.get("aya_text_emlaey") or ""
        if not t:
            continue

        ok = (qn in t) if match == "phrase" else match_word(t, qn)
        if ok:
            s = r.get("sura_no")
            a = r.get("aya_no")
            item = {
                "verse_key": f"{s}:{a}",
                "sura_no": s,
                "aya_no": a,
                "id": r.get("id"),
                "matched_term": qn
            }
            if include_text:
                item["aya_text"] = r.get("aya_text")
                item["sura_name_ar"] = r.get("sura_name_ar")
                item["jozz"] = r.get("jozz")
                item["page"] = r.get("page")
            results.append(item)

    total = len(results)
    page = results[offset: offset + limit]
    return {
        "query": qn,
        "mode": "literal",
        "match": match,
        "total": total,
        "offset": offset,
        "limit": limit,
        "returned": len(page),
        "has_more": (offset + limit) < total,
        "results": page
    }


@app.get("/v1/semantic")
@app.get("/v1/semantic/")
def semantic(
    q: str = Query(..., min_length=1),
    offset: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=200),
    include_text: bool = Query(False),
    x_api_key: Optional[str] = Header(None)
):
    require_api_key(x_api_key)

    qn = extract_topic_name(q)
    if not qn:
        raise HTTPException(status_code=400, detail="Empty query")

    # Total corpus size (stable), not "k"
    total = len(verse_keys) if verse_keys else 0

    # We only compute top (offset+limit) to avoid unnecessary work
    k = min(200, offset + limit)
    pairs = semantic_topk(qn, k)
    page_pairs = pairs[offset: offset + limit]

    results = []
    for vk, score in page_pairs:
        rec = by_key.get(vk)
        if not rec:
            continue
        s, a = vk.split(":", 1)
        item = {
            "verse_key": vk,
            "sura_no": int(s),
            "aya_no": int(a),
            "score": score
        }
        if include_text:
            item["aya_text"] = rec.get("aya_text")
            item["sura_name_ar"] = rec.get("sura_name_ar")
            item["jozz"] = rec.get("jozz")
            item["page"] = rec.get("page")
        results.append(item)

    returned = len(results)
    has_more = (offset + limit) < min(200, total)  # due to k cap at 200
    return {
        "query": qn,
        "mode": "semantic",
        "total": total,
        "offset": offset,
        "limit": limit,
        "returned": returned,
        "has_more": has_more,
        "results": results
    }


@app.get("/v1/recitations")
async def recitations(
    language: str = Query("ar", min_length=2, max_length=5),
    x_api_key: Optional[str] = Header(None)
):
    require_api_key(x_api_key)
    url = f"{QURAN_API_BASE}/resources/recitations"
    params = {"language": language}
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(url, params=params)
        if r.status_code != 200:
            raise HTTPException(status_code=502, detail=f"Upstream error: {r.status_code}")
        return r.json()


@app.get("/v1/audio/ayah")
async def ayah_audio(
    verse_key: str = Query(..., min_length=3),
    recitation_id: int = Query(..., ge=1),
    x_api_key: Optional[str] = Header(None)
):
    require_api_key(x_api_key)
    vk = norm_q(verse_key)
    if ":" not in vk:
        raise HTTPException(status_code=400, detail="Use verse_key like 2:255")

    url = f"{QURAN_API_BASE}/recitations/{recitation_id}/by_ayah/{vk}"
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(url)
        if r.status_code != 200:
            raise HTTPException(status_code=502, detail=f"Upstream error: {r.status_code}")

        data = r.json()
        audio_files = data.get("audio_files") or []
        if not audio_files:
            raise HTTPException(status_code=404, detail="No audio found")

        u = audio_files[0].get("url")
        if not u:
            raise HTTPException(status_code=404, detail="No audio url")

        if isinstance(u, str) and not u.startswith("http"):
            u = QURAN_VERSE_AUDIO_BASE.rstrip("/") + "/" + u.lstrip("/")

        return {"verse_key": vk, "recitation_id": recitation_id, "audio_url": u}
