import os, json, time
from typing import List, Dict, Any
import numpy as np
from openai import OpenAI

DATA_PATH = os.environ.get("HAFS_DATA_PATH", "hafsData_v2-0.json")
OUT_DIR = os.environ.get("INDEX_DIR", "index")
MODEL = os.environ.get("EMBED_MODEL", "text-embedding-3-small")

# Optional: reduce dimension (supported for text-embedding-3 models)
# If you want 512 dims to reduce size, set EMBED_DIMS=512
EMBED_DIMS = os.environ.get("EMBED_DIMS")
EMBED_DIMS = int(EMBED_DIMS) if EMBED_DIMS else None

BATCH_SIZE = int(os.environ.get("EMBED_BATCH", "64"))

def load_records(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, list):
        return raw
    return raw.get("data", [])

def main():
    if not os.path.exists(DATA_PATH):
        raise RuntimeError(f"Missing data file: {DATA_PATH}")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in environment.")

    client = OpenAI()

    records = load_records(DATA_PATH)
    if not records:
        raise RuntimeError("No records found in dataset.")

    os.makedirs(OUT_DIR, exist_ok=True)

    verse_keys: List[str] = []
    texts: List[str] = []

    for r in records:
        s = r.get("sura_no")
        a = r.get("aya_no")
        if not isinstance(s, int) or not isinstance(a, int):
            continue
        vk = f"{s}:{a}"
        # Use aya_text_emlaey for embedding to avoid decorative marks; if missing fallback to aya_text
        t = (r.get("aya_text_emlaey") or r.get("aya_text") or "").strip()
        if not t:
            continue
        verse_keys.append(vk)
        texts.append(t)

    print(f"Records to embed: {len(texts)}")

    all_vecs = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i+BATCH_SIZE]
        while True:
            try:
                kwargs = {"model": MODEL, "input": batch}
                if EMBED_DIMS:
                    kwargs["dimensions"] = EMBED_DIMS
                resp = client.embeddings.create(**kwargs)
                vecs = [d.embedding for d in resp.data]
                all_vecs.extend(vecs)
                break
            except Exception as e:
                print("Retry after error:", str(e))
                time.sleep(2)

        if (i // BATCH_SIZE) % 10 == 0:
            print(f"Embedded {min(i+BATCH_SIZE, len(texts))}/{len(texts)}")

    emb = np.array(all_vecs, dtype=np.float32)
    # Normalize for cosine similarity via dot product
    norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
    emb = emb / norms

    # Save as float16 to reduce repo size
    emb16 = emb.astype(np.float16)

    np.save(os.path.join(OUT_DIR, "embeddings_norm_f16.npy"), emb16)
    with open(os.path.join(OUT_DIR, "verse_keys.json"), "w", encoding="utf-8") as f:
        json.dump(verse_keys, f, ensure_ascii=False)

    meta = {
        "model": MODEL,
        "dims": int(emb.shape[1]),
        "count": int(emb.shape[0]),
        "normalized": True,
        "dtype": "float16",
        "source": DATA_PATH
    }
    with open(os.path.join(OUT_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("DONE.")
    print("Wrote:")
    print("- index/embeddings_norm_f16.npy")
    print("- index/verse_keys.json")
    print("- index/meta.json")

if __name__ == "__main__":
    main()
