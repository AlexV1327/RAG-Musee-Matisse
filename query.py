from pathlib import Path
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from config import INDEX_FILE, META_FILE, EMB_MODEL_NAME, TOP_K

def load_meta(path: Path):
    metas = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                metas.append(json.loads(line))
            except Exception:
                continue
    return metas

def search(query: str, top_k: int = TOP_K):
    metas = load_meta(META_FILE)
    assert INDEX_FILE.exists(), f"Index manquant: {INDEX_FILE}"
    index = faiss.read_index(str(INDEX_FILE))

    model = SentenceTransformer(EMB_MODEL_NAME)
    q = f"query: {query}" if "e5" in EMB_MODEL_NAME.lower() else query
    qv = model.encode([q], normalize_embeddings=True).astype("float32")

    scores, idxs = index.search(qv, top_k)
    scores, idxs = scores[0], idxs[0]
    results = []
    for i, s in zip(idxs, scores):
        if i < 0 or i >= len(metas):
            continue
        m = dict(metas[i])
        m["_score"] = float(s)
        results.append(m)
    return results

def main():
    import sys
    q = " ".join(sys.argv[1:]) or "Quels sont les horaires d'ouverture ?"
    for r in search(q):
        print(f"score={r['_score']:.3f} | {r['title']} — {r['url']} (# {r['chunk_id']})")

if __name__ == "__main__":
    main()
