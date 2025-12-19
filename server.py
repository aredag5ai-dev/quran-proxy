from fastapi import FastAPI, Query, HTTPException, Header
from typing import Optional, List, Dict, Any
import json, os, re
import httpx

app = FastAPI(title="Quran Proxy API (Hafs + Topics/Lexicon + Audio)")

# =====================
# Settings / Env
# =====================
DATA_PATH = os.environ.get("HAFS_DATA_PATH", "hafsData_v2-0.json")

TOPICS_PATH = os.environ.get("TOPICS_PATH", "topics_core.json")
LEXICON_PATH = os.environ.get("LEXICON_PATH", "lexicon_core.json")
STOPWORDS_PATH = os.environ.get("STOPWORDS_PATH", "stopwords_ar_generic.json")

API_KEY = os.environ.get("PROXY_API_KEY", "")
QURAN_API_BASE = os.environ.get("QURAN_API_BASE", "https://api.quran.com/api/v4")
QURAN_VERSE_AUDIO_BASE = os.environ.get("QURAN_VERSE_AUDIO_BASE", "https://verses.quran.foundation/")

AR_NUM = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")

def norm_q(s: str) -> str:
    s = (s or "").strip().translate(AR_NUM)
    s = re.sub(r"\s+", " ", s)
    return s

def extract_topic_name(s: str) -> str:
    """
    Robust extraction so API accepts:
    - "آيات عن الصبر" -> "الصبر"
    - "ايات عن الصبر" -> "الصبر"
    - "آية عن الصبر" -> "الصبر"
    - "عن الصبر" -> "الصبر"
    - "موضوع الصبر" -> "الصبر"
    - "بالمعنى الصبر" -> "الصبر"
    """
    x = norm_q(s)

    # Remove common leading phrases (Arabic)
    # We keep it conservative: ONLY remove wrappers, not changing meaning.
    patterns = [
        r'^(?:آيات|ايات|آية|اية)\s+عن\s+',
        r'^عن\s+',
        r'^(?:موضوع|الموضوع)\s+',
        r'^(?:بالمعنى|بمعنى)\s+',
        r'^(?:ابحث|دور|عايز|أريد|اريد)\s+(?:عن\s+)?',
        r'^(?:آيات|ايات)\s+(?:تتحدث\s+)?عن\s+'
    ]
    for p in patterns:
        x2 = re.sub(p, "", x)
        if x2 != x:
            x = x2.strip()

    # Remove trailing punctuation
    x = re.sub(r'[؟\?\!\.\,\؛\:\-]+$', "", x).strip()

    # If user wrote "آيات عن الصبر في القرآن" -> remove ending wrappers
    x = re.sub(r'\s+(?:في\s+القرآن|بالقرآن|من\s+القرآن)$', "", x).strip()

    return x

def require_api_key(x_api_key: Optional[str]):
    if not API_KEY:
        raise HTTPException(status_code=500, detail="Server PROXY_API_KEY is not set")
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

def match_word(text: str, token: str) -> bool:
    pat = r'(^|[^\w])' + re.escape(token) + r'([^\w]|$)'
    return re.search(pat, text, flags=re.UNICODE) is not None

def load_json_if_exists(path: str, default):
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return default

# =====================
# Load Quran data
# =====================
if not os.path.exists(DATA_PATH):
    raise RuntimeError(f"Data file not found: {DATA_PATH}")

with open(DATA_PATH, "r", encoding="utf-8") as f:
    raw = json.load(f)

records: List[Dict[str, Any]] = raw if isinstance(raw, list) else raw.get("data", [])
if not isinstance(records, list):
    raise RuntimeError("Unexpected JSON structure: expected a list of ayah records.")

by_key: Dict[str, Dict[str, Any]] = {}
for r in records:
    s = r.get("sura_no")
    a = r.get("aya_no")
    if isinstance(s, int) and isinstance(a, int):
        by_key[f"{s}:{a}"] = r

# =====================
# Load topics / lexicon / stopwords
# =====================
topics = load_json_if_exists(TOPICS_PATH, {})
lexicon = load_json_if_exists(LEXICON_PATH, {})
stopwords_list = load_json_if_exists(STOPWORDS_PATH, [])

stopwords = set()
if isinstance(stopwords_list, list):
    for w in stopwords_list:
        if isinstance(w, str) and w.strip():
            stopwords.add(norm_q(w))

# =====================
# Health
# =====================
@app.get("/v1/health")
def health():
    return {
        "ok": True,
        "records": len(records),
        "topics": len(topics) if isinstance(topics, dict) else 0,
        "lexicon": len(lexicon) if isinstance(lexicon, dict) else 0,
        "stopwords": len(stopwords),
    }

# =====================
# Verse by key
# =====================
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

# =====================
# Literal search
# =====================
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

    # Stopword: count-only
    if qn in stopwords:
        total = 0
        for r in records:
            t = r.get("aya_text_emlaey") or ""
            if not t:
                continue
            ok = (qn in t) if match == "phrase" else match_word(t, qn)
            if ok:
                total += 1
        return {
            "query": qn,
            "mode": "literal",
            "match": match,
            "count_only": True,
            "total": total,
            "offset": offset,
            "limit": limit,
            "results": []
        }

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
        "results": page
    }

# =====================
# Topic (curated)
# =====================
@app.get("/v1/topic")
def get_topic(
    name: str = Query(..., min_length=1),
    offset: int = Query(0, ge=0),
    limit: int = Query(25, ge=1, le=200),
    include_text: bool = Query(True),
    x_api_key: Optional[str] = Header(None)
):
    require_api_key(x_api_key)
    n_raw = norm_q(name)
    n = extract_topic_name(n_raw)

    if not isinstance(topics, dict) or n not in topics:
        return {"topic": n, "found": False, "total": 0, "offset": offset, "limit": limit, "results": []}

    obj = topics[n] or {}
    verse_keys = obj.get("verse_keys", []) or []
    total = len(verse_keys)
    page_keys = verse_keys[offset: offset + limit]

    results = []
    for vk in page_keys:
        vk2 = norm_q(vk)
        rec = by_key.get(vk2)
        if not rec:
            continue
        s, a = vk2.split(":", 1)
        item = {"verse_key": vk2, "sura_no": int(s), "aya_no": int(a)}
        if include_text:
            item["aya_text"] = rec.get("aya_text")
            item["sura_name_ar"] = rec.get("sura_name_ar")
            item["jozz"] = rec.get("jozz")
            item["page"] = rec.get("page")
        results.append(item)

    return {
        "topic": n,
        "topic_id": obj.get("topic_id"),
        "found": True,
        "description": obj.get("description"),
        "category": obj.get("category"),
        "total": total,
        "offset": offset,
        "limit": limit,
        "results": results
    }

# =====================
# Meaning (lexicon-first hybrid A)
# =====================
@app.get("/v1/meaning")
def meaning(
    name: str = Query(..., min_length=1),
    prefer: str = Query("lexicon", pattern="^(lexicon|topic)$"),
    offset: int = Query(0, ge=0),
    limit: int = Query(25, ge=1, le=200),
    include_text: bool = Query(True),
    x_api_key: Optional[str] = Header(None)
):
    require_api_key(x_api_key)

    n_raw = norm_q(name)
    n = extract_topic_name(n_raw)

    # Try exact, then remove leading "ال" once
    candidates = [n]
    if n.startswith("ال") and len(n) > 2:
        candidates.append(n[2:])
    # also try adding "ال" if user wrote without it
    if not n.startswith("ال"):
        candidates.append("ال" + n)

    # 1) lexicon preferred
    if prefer == "lexicon" and isinstance(lexicon, dict):
        entry = None
        used_key = None
        for c in candidates:
            if c in lexicon:
                entry = lexicon[c]
                used_key = c
                break

        if entry is not None:
            entry = entry or {}
            tokens = entry.get("tokens", []) or []
            tokens = [norm_q(t) for t in tokens if isinstance(t, str) and t.strip()]

            # if tokens too common -> count_only guard
            if any(t in stopwords for t in tokens):
                return {
                    "query": used_key or n,
                    "mode": "lexicon",
                    "found": True,
                    "lexicon_id": entry.get("lexicon_id"),
                    "label_ar": entry.get("label_ar", used_key or n),
                    "category": entry.get("category"),
                    "tokens_used": tokens,
                    "count_only": True,
                    "total": 0,
                    "offset": offset,
                    "limit": limit,
                    "results": []
                }

            hits = []
            for r in records:
                t = r.get("aya_text_emlaey") or ""
                if not t:
                    continue
                ok = False
                for tok in tokens:
                    if tok and match_word(t, tok):
                        ok = True
                        break
                if ok:
                    s = r.get("sura_no")
                    a = r.get("aya_no")
                    hits.append({"verse_key": f"{s}:{a}", "sura_no": s, "aya_no": a})

            total = len(hits)
            page = hits[offset: offset + limit]

            if include_text:
                hydrated = []
                for item in page:
                    vk = item["verse_key"]
                    rec = by_key.get(vk)
                    if not rec:
                        continue
                    obj = dict(item)
                    obj["aya_text"] = rec.get("aya_text")
                    obj["sura_name_ar"] = rec.get("sura_name_ar")
                    obj["jozz"] = rec.get("jozz")
                    obj["page"] = rec.get("page")
                    hydrated.append(obj)
                page = hydrated

            return {
                "query": used_key or n,
                "mode": "lexicon",
                "found": True,
                "lexicon_id": entry.get("lexicon_id"),
                "label_ar": entry.get("label_ar", used_key or n),
                "category": entry.get("category"),
                "tokens_used": tokens,
                "total": total,
                "offset": offset,
                "limit": limit,
                "results": page
            }

    # 2) topic fallback
    if isinstance(topics, dict):
        obj = None
        used_key = None
        for c in candidates:
            if c in topics:
                obj = topics[c]
                used_key = c
                break

        if obj is not None:
            obj = obj or {}
            verse_keys = obj.get("verse_keys", []) or []
            total = len(verse_keys)
            page_keys = verse_keys[offset: offset + limit]
            results = []
            for vk in page_keys:
                vk2 = norm_q(vk)
                rec = by_key.get(vk2)
                if not rec:
                    continue
                s, a = vk2.split(":", 1)
                item = {"verse_key": vk2, "sura_no": int(s), "aya_no": int(a)}
                if include_text:
                    item["aya_text"] = rec.get("aya_text")
                    item["sura_name_ar"] = rec.get("sura_name_ar")
                    item["jozz"] = rec.get("jozz")
                    item["page"] = rec.get("page")
                results.append(item)

            return {
                "query": used_key or n,
                "mode": "topic",
                "found": True,
                "topic_id": obj.get("topic_id"),
                "description": obj.get("description"),
                "category": obj.get("category"),
                "total": total,
                "offset": offset,
                "limit": limit,
                "results": results
            }

    return {
        "query": n,
        "mode": "lexicon" if prefer == "lexicon" else "topic",
        "found": False,
        "total": 0,
        "offset": offset,
        "limit": limit,
        "results": []
    }

# =====================
# Audio: list recitations
# =====================
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

# =====================
# Audio: ayah recitation url
# =====================
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
