from fastapi import FastAPI, Query, HTTPException, Header
from typing import Optional, List, Dict, Any
import json, os, re

app = FastAPI(title="Quran Hafs Exact/Meaning Search API")

# ===== Settings =====
DATA_PATH = os.environ.get("HAFS_DATA_PATH", "hafsData_v2-0.json")
API_KEY = os.environ.get("PROXY_API_KEY", "")

TOPICS_PATH = os.environ.get("TOPICS_PATH", "topics_core.json")

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
    # Arabic-friendly token boundary: start/end OR whitespace OR punctuation
    boundary = r"(?:^|[\s\.,;:!\?\(\)\[\]\{\}\"'«»،؛ـ\-])"
    pat = boundary + re.escape(token) + boundary
    return re.search(pat, text) is not None

# Meaning group example (optional)
ALLAH_MEANING_TERMS = [
    "الله", "بالله", "لله", "فالله", "والله", "تالله",
    "رب", "الرب", "ربنا", "إله", "الرحمن", "الرحيم"
]
ALLAH_SEED_TERMS = set(ALLAH_MEANING_TERMS)

# ===== Load hafs data =====
if not os.path.exists(DATA_PATH):
    raise RuntimeError(f"Data file not found: {DATA_PATH}")

with open(DATA_PATH, "r", encoding="utf-8") as f:
    raw = json.load(f)

records: List[Dict[str, Any]] = raw if isinstance(raw, list) else raw.get("data", [])
if not isinstance(records, list):
    raise RuntimeError("Unexpected JSON structure: expected a list of ayah records.")

# ===== Build fast index for verse lookup =====
verse_index: Dict[str, Dict[str, Any]] = {}
for r in records:
    s_no = r.get("sura_no")
    a_no = r.get("aya_no")
    if isinstance(s_no, int) and isinstance(a_no, int):
        verse_index[f"{s_no}:{a_no}"] = r

# ===== Load topics =====
topics: Dict[str, Any] = {}
if os.path.exists(TOPICS_PATH):
    with open(TOPICS_PATH, "r", encoding="utf-8") as tf:
        topics = json.load(tf)

# Build label_ar -> topic_id index (so /v1/topic?name=الغيبة works)
topics_by_label: Dict[str, str] = {}
if isinstance(topics, dict):
    for tid, obj in topics.items():
        if isinstance(obj, dict):
            lab = norm_q(obj.get("label_ar", ""))
            if lab:
                topics_by_label[lab] = tid


@app.get("/v1/health")
def health():
    return {"ok": True, "records": len(records), "topics": len(topics) if isinstance(topics, dict) else 0}


@app.get("/v1/search")
def search(
    q: str = Query(..., min_length=1),
    mode: str = Query("literal", pattern="^(literal|meaning)$"),
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

    terms: List[str] = []
    if mode == "meaning":
        tokens = [t for t in qn.split(" ") if t]
        if any(t in ALLAH_SEED_TERMS for t in tokens):
            terms = ALLAH_MEANING_TERMS
        else:
            terms = tokens

    results: List[Dict[str, Any]] = []

    for r in records:
        t = r.get("aya_text_emlaey") or ""
        if not t:
            continue

        ok = False
        matched_term = None

        if mode == "literal":
            if match == "phrase":
                ok = qn in t
                if ok:
                    matched_term = qn
            else:
                ok = match_word(t, qn)
                if ok:
                    matched_term = qn
        else:
            for term in terms:
                if match_word(t, term):
                    ok = True
                    matched_term = term
                    break

        if ok:
            sura_no = r.get("sura_no")
            aya_no = r.get("aya_no")

            item: Dict[str, Any] = {
                "verse_key": f"{sura_no}:{aya_no}",
                "sura_no": sura_no,
                "aya_no": aya_no,
                "id": r.get("id"),
                "matched_term": matched_term
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
        "mode": mode,
        "match": match,
        "total": total,
        "offset": offset,
        "limit": limit,
        "results": page
    }


@app.get("/v1/verse/{verse_key}")
def get_verse(
    verse_key: str,
    x_api_key: Optional[str] = Header(None)
):
    require_api_key(x_api_key)

    vk = norm_q(verse_key)
    if ":" not in vk:
        raise HTTPException(status_code=400, detail="Use verse_key like 2:255")

    # Return from index (fast + stable)
    rec = verse_index.get(vk)
    if not rec:
        raise HTTPException(status_code=404, detail="Not found")

    out = dict(rec)
    out["verse_key"] = vk
    return out


@app.get("/v1/topic")
def get_topic(
    name: str = Query(..., min_length=1),
    offset: int = Query(0, ge=0),
    limit: int = Query(25, ge=1, le=200),
    include_text: bool = Query(True),
    x_api_key: Optional[str] = Header(None)
):
    require_api_key(x_api_key)

    n = norm_q(name)

    # Accept either topic_id (e.g., "ghiba") OR Arabic label (e.g., "الغيبة")
    tid = n if (isinstance(topics, dict) and n in topics) else topics_by_label.get(n)

    if not tid:
        return {
            "topic": n,
            "found": False,
            "total": 0,
            "offset": offset,
            "limit": limit,
            "results": []
        }

    topic_obj = topics.get(tid, {}) if isinstance(topics, dict) else {}
    verse_keys = topic_obj.get("verse_keys", []) or []

    total = len(verse_keys)
    page_keys = verse_keys[offset: offset + limit]

    results: List[Dict[str, Any]] = []
    for vk in page_keys:
        vk2 = norm_q(vk)
        rec = verse_index.get(vk2)
        if not rec:
            continue

        # minimal required fields
        s_no = rec.get("sura_no")
        a_no = rec.get("aya_no")

        item: Dict[str, Any] = {
            "verse_key": vk2,
            "sura_no": s_no,
            "aya_no": a_no
        }

        if include_text:
            item["aya_text"] = rec.get("aya_text")
            item["sura_name_ar"] = rec.get("sura_name_ar")
            item["jozz"] = rec.get("jozz")
            item["page"] = rec.get("page")

        results.append(item)

    return {
        "topic": topic_obj.get("label_ar", n),
        "topic_id": tid,
        "found": True,
        "description": topic_obj.get("definition"),
        "category": topic_obj.get("category"),
        "total": total,
        "offset": offset,
        "limit": limit,
        "results": results
    }
