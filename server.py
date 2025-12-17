from fastapi import FastAPI, Query, HTTPException, Header
from typing import Optional, List, Dict, Any
import json, os, re

app = FastAPI(title="Quran Hafs Exact/Meaning Search API")

# ===== Settings =====
DATA_PATH = os.environ.get("HAFS_DATA_PATH", "hafsData_v2-0.json")
API_KEY = os.environ.get("PROXY_API_KEY", "")  # set it in PowerShell before running

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
    # Match as a full word (not inside another word). Works well with Arabic in aya_text_emlaey.
    pat = r'(^|[^\w])' + re.escape(token) + r'([^\w]|$)'
    return re.search(pat, text, flags=re.UNICODE) is not None

# Meaning group: Allah/ رب and attached forms
ALLAH_MEANING_TERMS = [
    "الله", "بالله", "لله", "فالله", "والله", "تالله",
    "رب", "الرب", "ربنا", "إله", "الرحمن", "الرحيم"
]
ALLAH_SEED_TERMS = set(["الله", "رب", "الرب", "ربنا", "إله", "الرحمن", "الرحيم", "بالله", "لله", "فالله", "والله", "تالله"])

# ===== Load data =====
if not os.path.exists(DATA_PATH):
    raise RuntimeError(f"Data file not found: {DATA_PATH}")

with open(DATA_PATH, "r", encoding="utf-8") as f:
    raw = json.load(f)

records: List[Dict[str, Any]] = raw if isinstance(raw, list) else raw.get("data", [])
if not isinstance(records, list):
    raise RuntimeError("Unexpected JSON structure: expected a list of ayah records.")

@app.get("/v1/health")
def health():
    return {"ok": True, "records": len(records)}

@app.get("/v1/search")
def search(
    q: str = Query(..., min_length=1),
    mode: str = Query("literal", pattern="^(literal|meaning)$"),
    match: str = Query("word", pattern="^(word|phrase)$"),
    offset: int = Query(0, ge=0),
    limit: int = Query(25, ge=1, le=200),
    x_api_key: Optional[str] = Header(None)
):
    require_api_key(x_api_key)

    qn = norm_q(q)
    if not qn:
        raise HTTPException(status_code=400, detail="Empty query")

    # Build meaning terms if requested
    terms: List[str] = []
    if mode == "meaning":
        tokens = [t for t in qn.split(" ") if t]
        # If user query touches Allah/Rab group, expand it
        if any(t in ALLAH_SEED_TERMS for t in tokens):
            terms = ALLAH_MEANING_TERMS
        else:
            terms = tokens  # meaning mode but no known group: use tokens as OR list

    results = []
    for r in records:
        t = r.get("aya_text_emlaey") or ""
        if not t:
            continue

        ok = False
        matched_term = None

        if mode == "literal":
            if match == "phrase":
                ok = qn in t
                matched_term = qn if ok else None
            else:  # word
                ok = match_word(t, qn)
                matched_term = qn if ok else None
        else:  # meaning
            # Meaning = OR over terms, word-level matching
            for term in terms:
                if match_word(t, term):
                    ok = True
                    matched_term = term
                    break

        if ok:
            sura_no = r.get("sura_no")
            aya_no = r.get("aya_no")
            verse_key = f"{sura_no}:{aya_no}"
            results.append({
                "verse_key": verse_key,
                "sura_no": sura_no,
                "aya_no": aya_no,
                "id": r.get("id"),
                "matched_term": matched_term
            })

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

    s, a = vk.split(":", 1)
    try:
        s_no = int(s)
        a_no = int(a)
    except:
        raise HTTPException(status_code=400, detail="Invalid verse_key numbers")

    for r in records:
        if r.get("sura_no") == s_no and r.get("aya_no") == a_no:
            out = dict(r)
            out["verse_key"] = f"{s_no}:{a_no}"
            # IMPORTANT: aya_text is the only Quran text to display (exact as stored)
            return out

    raise HTTPException(status_code=404, detail="Not found")
