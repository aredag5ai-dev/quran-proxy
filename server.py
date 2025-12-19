from fastapi import FastAPI, Query, HTTPException, Header
from typing import Optional, List, Dict, Any, Set
import json, os, re

app = FastAPI(title="Quran Hafs Exact/Meaning Search API (Hybrid-A)")

# ======================
# Settings / File Paths
# ======================
DATA_PATH = os.environ.get("HAFS_DATA_PATH", "hafsData_v2-0.json")
API_KEY = os.environ.get("PROXY_API_KEY", "")

TOPICS_PATH = os.environ.get("TOPICS_PATH", "topics_core.json")
LEXICON_PATH = os.environ.get("LEXICON_PATH", "meaning_lexicon_core.json")
STOPWORDS_PATH = os.environ.get("STOPWORDS_PATH", "stopwords_ar_generic.json")

AR_NUM = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")

def norm_q(s: str) -> str:
    s = (s or "").strip().translate(AR_NUM)
    s = re.sub(r"\s+", " ", s)
    return s

def require_api_key(x_api_key: Optional[str]):
    if not API_KEY:
        raise HTTPException(status_code=500, detail="Server PROXY_API_KEY is not set")
    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="No Api Key or Invalid Key / Permission")

def match_word(text: str, token: str) -> bool:
    """
    Strict word-boundary matching suitable for Arabic text in aya_text_emlaey.
    Exact token matching with boundaries; no fuzzy matching.
    """
    boundary = r"(?:^|[\s\.,;:!\?\(\)\[\]\{\}\"'«»،؛ـ\-])"
    pat = boundary + re.escape(token) + boundary
    return re.search(pat, text) is not None

# ======================
# Load Hafs Data
# ======================
if not os.path.exists(DATA_PATH):
    raise RuntimeError(f"Data file not found: {DATA_PATH}")

with open(DATA_PATH, "r", encoding="utf-8") as f:
    raw = json.load(f)

records: List[Dict[str, Any]] = raw if isinstance(raw, list) else raw.get("data", [])
if not isinstance(records, list):
    raise RuntimeError("Unexpected JSON structure: expected a list of ayah records.")

# Fast verse lookup
verse_index: Dict[str, Dict[str, Any]] = {}
for r in records:
    s_no = r.get("sura_no")
    a_no = r.get("aya_no")
    if isinstance(s_no, int) and isinstance(a_no, int):
        verse_index[f"{s_no}:{a_no}"] = r

# ======================
# Load Topics (curated)
# ======================
topics: Dict[str, Any] = {}
if os.path.exists(TOPICS_PATH):
    with open(TOPICS_PATH, "r", encoding="utf-8") as tf:
        topics = json.load(tf)

topics_by_label: Dict[str, str] = {}
if isinstance(topics, dict):
    for tid, obj in topics.items():
        if isinstance(obj, dict):
            lab = norm_q(obj.get("label_ar", ""))
            if lab:
                topics_by_label[lab] = tid

# ======================
# Load Meaning Lexicon (Hybrid-A fallback)
# ======================
lexicon: Dict[str, Any] = {}
if os.path.exists(LEXICON_PATH):
    with open(LEXICON_PATH, "r", encoding="utf-8") as lf:
        lexicon = json.load(lf)

lexicon_by_label: Dict[str, str] = {}
if isinstance(lexicon, dict):
    for lid, obj in lexicon.items():
        if isinstance(obj, dict):
            lab = norm_q(obj.get("label_ar", ""))
            if lab:
                lexicon_by_label[lab] = lid

# ======================
# Load Stopwords
# ======================
stopwords: Set[str] = set()
if os.path.exists(STOPWORDS_PATH):
    try:
        with open(STOPWORDS_PATH, "r", encoding="utf-8") as sf:
            sw = json.load(sf)
        if isinstance(sw, list):
            stopwords = set(norm_q(x) for x in sw if norm_q(x))
        elif isinstance(sw, dict) and "stopwords" in sw and isinstance(sw["stopwords"], list):
            stopwords = set(norm_q(x) for x in sw["stopwords"] if norm_q(x))
    except Exception:
        stopwords = set()

# ======================
# Helpers
# ======================
def build_result_item(rec: Dict[str, Any], include_text: bool) -> Dict[str, Any]:
    s_no = rec.get("sura_no")
    a_no = rec.get("aya_no")
    item: Dict[str, Any] = {
        "verse_key": f"{s_no}:{a_no}",
        "sura_no": s_no,
        "aya_no": a_no
    }
    if include_text:
        item["aya_text"] = rec.get("aya_text")
        item["sura_name_ar"] = rec.get("sura_name_ar")
        item["jozz"] = rec.get("jozz")
        item["page"] = rec.get("page")
    return item

def paginate(items: List[Any], offset: int, limit: int) -> List[Any]:
    return items[offset: offset + limit]

def topic_lookup(name: str) -> Optional[Dict[str, Any]]:
    n = norm_q(name)
    if isinstance(topics, dict) and n in topics:
        return {"topic_id": n, "obj": topics[n]}
    tid = topics_by_label.get(n)
    if tid and isinstance(topics, dict) and tid in topics:
        return {"topic_id": tid, "obj": topics[tid]}
    return None

def lexicon_lookup(name: str) -> Optional[Dict[str, Any]]:
    n = norm_q(name)
    if isinstance(lexicon, dict) and n in lexicon:
        return {"lexicon_id": n, "obj": lexicon[n]}
    lid = lexicon_by_label.get(n)
    if lid and isinstance(lexicon, dict) and lid in lexicon:
        return {"lexicon_id": lid, "obj": lexicon[lid]}
    return None

def strict_token_search(tokens: List[str]) -> List[str]:
    found_keys: Set[str] = set()

    cleaned = [norm_q(t) for t in tokens if norm_q(t)]
    cleaned = [t for t in cleaned if t not in stopwords]
    if not cleaned:
        return []

    for r in records:
        t = r.get("aya_text_emlaey") or ""
        if not t:
            continue
        for tok in cleaned:
            if match_word(t, tok):
                s_no = r.get("sura_no")
                a_no = r.get("aya_no")
                if isinstance(s_no, int) and isinstance(a_no, int):
                    found_keys.add(f"{s_no}:{a_no}")
                break

    def keyfunc(vk: str):
        s, a = vk.split(":", 1)
        return (int(s), int(a))

    return sorted(found_keys, key=keyfunc)

def count_only_for_token(token: str) -> int:
    tok = norm_q(token)
    if not tok:
        return 0
    c = 0
    for r in records:
        t = r.get("aya_text_emlaey") or ""
        if t and match_word(t, tok):
            c += 1
    return c

def topic_response(q: str, thit: Dict[str, Any], offset: int, limit: int, include_text: bool) -> Dict[str, Any]:
    tid = thit["topic_id"]
    obj = thit["obj"] if isinstance(thit["obj"], dict) else {}
    verse_keys = obj.get("verse_keys", []) or []
    total = len(verse_keys)
    page_keys = paginate(verse_keys, offset, limit)

    results: List[Dict[str, Any]] = []
    for vk in page_keys:
        vk2 = norm_q(vk)
        rec = verse_index.get(vk2)
        if rec:
            results.append(build_result_item(rec, include_text))

    return {
        "query": q,
        "mode": "topic",
        "topic_id": tid,
        "topic": obj.get("label_ar", q),
        "found": True,
        "description": obj.get("definition"),
        "category": obj.get("category"),
        "total": total,
        "offset": offset,
        "limit": limit,
        "results": results
    }

def lexicon_response(q: str, lhit: Dict[str, Any], offset: int, limit: int, include_text: bool) -> Dict[str, Any]:
    lid = lhit["lexicon_id"]
    obj = lhit["obj"] if isinstance(lhit["obj"], dict) else {}
    tokens = obj.get("tokens", []) or []
    if not isinstance(tokens, list):
        tokens = []

    cleaned_tokens = [norm_q(t) for t in tokens if norm_q(t) and norm_q(t) not in stopwords]
    if not cleaned_tokens:
        return {
            "query": q,
            "mode": "count_only",
            "found": True,
            "reason": "empty_or_stopwords_tokens",
            "lexicon_id": lid,
            "label_ar": obj.get("label_ar"),
            "total": count_only_for_token(q),
            "offset": offset,
            "limit": limit,
            "results": []
        }

    keys = strict_token_search(cleaned_tokens)
    total = len(keys)
    page_keys = paginate(keys, offset, limit)

    results: List[Dict[str, Any]] = []
    for vk in page_keys:
        rec = verse_index.get(vk)
        if rec:
            results.append(build_result_item(rec, include_text))

    return {
        "query": q,
        "mode": "lexicon",
        "found": True,
        "lexicon_id": lid,
        "label_ar": obj.get("label_ar", q),
        "category": obj.get("category"),
        "tokens_used": cleaned_tokens,
        "total": total,
        "offset": offset,
        "limit": limit,
        "results": results
    }

# ======================
# Endpoints
# ======================
@app.get("/v1/health")
def health():
    return {
        "ok": True,
        "records": len(records),
        "topics": len(topics) if isinstance(topics, dict) else 0,
        "lexicon": len(lexicon) if isinstance(lexicon, dict) else 0,
        "stopwords": len(stopwords)
    }

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

    if qn in stopwords:
        return {
            "query": qn,
            "mode": mode,
            "match": match,
            "count_only": True,
            "total": count_only_for_token(qn),
            "offset": offset,
            "limit": limit,
            "results": []
        }

    terms: List[str] = []
    if mode == "meaning":
        terms = [t for t in qn.split(" ") if t]

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
                termn = norm_q(term)
                if termn and termn not in stopwords and match_word(t, termn):
                    ok = True
                    matched_term = termn
                    break

        if ok:
            item = build_result_item(r, include_text)
            item["id"] = r.get("id")
            item["matched_term"] = matched_term
            results.append(item)

    total = len(results)
    page = paginate(results, offset, limit)

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

    thit = topic_lookup(name)
    if not thit:
        return {"topic": norm_q(name), "found": False, "total": 0, "offset": offset, "limit": limit, "results": []}

    tid = thit["topic_id"]
    obj = thit["obj"] if isinstance(thit["obj"], dict) else {}
    verse_keys = obj.get("verse_keys", []) or []
    total = len(verse_keys)
    page_keys = paginate(verse_keys, offset, limit)

    results: List[Dict[str, Any]] = []
    for vk in page_keys:
        vk2 = norm_q(vk)
        rec = verse_index.get(vk2)
        if rec:
            results.append(build_result_item(rec, include_text))

    return {
        "topic": obj.get("label_ar", norm_q(name)),
        "topic_id": tid,
        "found": True,
        "description": obj.get("definition"),
        "category": obj.get("category"),
        "total": total,
        "offset": offset,
        "limit": limit,
        "results": results
    }

@app.get("/v1/meaning")
def meaning_hybrid_a(
    name: str = Query(..., min_length=1),
    prefer: str = Query("lexicon", pattern="^(lexicon|topic)$"),
    offset: int = Query(0, ge=0),
    limit: int = Query(25, ge=1, le=200),
    include_text: bool = Query(True),
    x_api_key: Optional[str] = Header(None)
):
    """
    Hybrid-A meaning retrieval with preference control:
    - prefer=lexicon (default): try lexicon first, then topic
    - prefer=topic: try topic first, then lexicon
    Both are strict (no synonyms, no interpretation).
    """
    require_api_key(x_api_key)

    q = norm_q(name)
    if not q:
        raise HTTPException(status_code=400, detail="Empty name")

    if q in stopwords:
        return {
            "query": q,
            "mode": "count_only",
            "found": True,
            "reason": "stopword",
            "total": count_only_for_token(q),
            "offset": offset,
            "limit": limit,
            "results": []
        }

    thit = topic_lookup(q)
    lhit = lexicon_lookup(q)

    # prefer logic
    if prefer == "topic":
        if thit:
            return topic_response(q, thit, offset, limit, include_text)
        if lhit:
            return lexicon_response(q, lhit, offset, limit, include_text)
    else:
        # prefer lexicon (default)
        if lhit:
            return lexicon_response(q, lhit, offset, limit, include_text)
        if thit:
            return topic_response(q, thit, offset, limit, include_text)

    return {
        "query": q,
        "mode": "lexicon",
        "found": False,
        "total": 0,
        "offset": offset,
        "limit": limit,
        "results": []
    }
