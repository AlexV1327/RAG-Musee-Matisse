
from pathlib import Path
import json, math 
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from preview import iter_jsonl

from config import CHUNKS_FILE, EMB_MODEL_NAME, BATCH_SIZE, INDEX_FILE, META_FILE

def main():
    """
    
    """
    if not CHUNKS_FILE.exists():
        print(f"[err] Introuvable : {CHUNKS_FILE}")
        return
    
    # 1) Charger les chunks et préparer les textes + métadonnées

    texts = []
    metas = [] # on garde l'ordre strictement aligné avec 'texts'

    for rec in iter_jsonl(CHUNKS_FILE):
        t = (rec.get("text") or "").strip()
        if not t :
            continue

        # E5 : préfixe passage 

        texts.append(f"passage: {t}")
        metas.append({
            "chunk_id": rec.get("chunk_id"),
            "doc_id": rec.get("doc_id"),
            "url": rec.get("url"),
            "title": (rec.get('title') or "").strip(),
        })


    if not texts:
        print("[err] Aucun chunk à encoder")
        return
    
    print(f"[info] Chunks à encoder : {len(texts)}")

    # 2) Charger le modèle d'emmbeddings 
    model = SentenceTransformer(EMB_MODEL_NAME)

    # 3) Encoder en batch + normalisation (cosine (similarité))

    embs = []
    num_batches = math.ceil(len(texts)/ BATCH_SIZE)


    for b in range(num_batches):
        batch = texts[b*BATCH_SIZE: (b+1)*BATCH_SIZE]
        vecs = model.encode(
            batch,
            batch_size=BATCH_SIZE,
            convert_to_numpy=True,
            normalize_embeddings=True, # Crucial pour cosine
            show_progress_bar=True
        )
        embs.append(vecs.astype("float32"))
    embs = np.vstack(embs) # (N = nombre total de chunks, D = dimension des embeddings)
    N, D = embs.shape
    print(f"[info] Embeddings shape: {embs.shape}")

    # 4) Index FAISS (cosine -> IndexFlatIP car vecteurs normalisés)

    index = faiss.IndexFlatIP(D)
    index.add(embs)
    faiss.write_index(index, str(INDEX_FILE))
    print(f"[ok] Index écrit -> {INDEX_FILE}")

    # 5) Sauvegarder les métadonnées alignées (JSONL)

    with META_FILE.open("w", encoding="utf-8") as out:
        for m in metas :
            out.write(json.dumps(m, ensure_ascii=False) + "\n")
    print(f"[ok] Métadonnées écrites -> {META_FILE}")

    print(f"[done] {N} vecteurs indexés.")

if __name__ == "__main__":
    main()














