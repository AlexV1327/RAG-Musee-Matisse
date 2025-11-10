# 05_query.py
from pathlib import Path
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from config import DATA_DIR, INDEX_FILE, META_FILE, EMB_MODEL_NAME, TOP_K



def load_meta(path: Path):
    metas = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                metas.append(json.loads(line))
            except:
                continue
    return metas

def main():
    if not INDEX_FILE.exists() or not META_FILE.exists():
        print("[err] Index ou métadonnées manquants. Lance d'abord embed_index.py")
        return

    # 1) Charger index + meta
    index = faiss.read_index(str(INDEX_FILE))
    metas = load_meta(META_FILE)
    if index.ntotal != len(metas):
        print(f"[warn] Index={index.ntotal} != metas={len(metas)} (désalignement possible)")

    # 2) Charger modèle d'embeddings
    model = SentenceTransformer(EMB_MODEL_NAME)

    # 3) Lire la requête utilisateur
    query = input("Votre question (FR/EN) : ").strip()
    if not query:
        print("[err] Requête vide.")
        return

    # 4) Encoder la requête (E5: préfixe query)
    q_emb = model.encode([f"query: {query}"],
                         convert_to_numpy=True,
                         normalize_embeddings=True).astype("float32")

    # 5) Recherche FAISS
    scores, idxs = index.search(q_emb, TOP_K)  # (1, k)
    idxs = idxs[0]
    scores = scores[0]

    print("\n=== Résultats ===")
    for rank, (i, s) in enumerate(zip(idxs, scores), start=1):
        if i < 0 or i >= len(metas):
            continue
        m = metas[i]
        print(f"{rank:02d}. score={s:.3f}")
        print(f"    titre : {m.get('title') or '[Sans titre]'}")
        print(f"    url   : {m.get('url')}")
        print(f"    chunk : {m.get('chunk_id')}\n")

if __name__ == "__main__":
    main()
