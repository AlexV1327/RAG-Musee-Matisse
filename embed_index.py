from pathlib import Path
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from config import CHUNKS_FILE, EMB_MODEL_NAME, BATCH_SIZE, INDEX_FILE, META_FILE

def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue

def batched(it, size):
    buf = []
    for x in it:
        buf.append(x)
        if len(buf) >= size:
            yield buf
            buf = []
    if buf:
        yield buf

def main():
    assert CHUNKS_FILE.exists(), f"Manquant: {CHUNKS_FILE} (lance text_chunker.py)"

    print(f"[load model] {EMB_MODEL_NAME}")
    model = SentenceTransformer(EMB_MODEL_NAME)

    metas, vecs = [], []
    for batch in batched(iter_jsonl(CHUNKS_FILE), BATCH_SIZE):
        texts = [b["text"] for b in batch]
        # E5 attend les préfixes
        texts = [f"passage: {t}" if "e5" in EMB_MODEL_NAME.lower() else t for t in texts]
        emb = model.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=True, normalize_embeddings=True)
        vecs.append(emb.astype("float32"))
        metas.extend(batch)

    X = np.vstack(vecs)
    print("[faiss] build index (IP, normalisé)")
    index = faiss.IndexFlatIP(X.shape[1])
    index.add(X)

    print(f"[save] index -> {INDEX_FILE}")
    faiss.write_index(index, str(INDEX_FILE))

    print(f"[save] meta  -> {META_FILE}")
    with META_FILE.open("w", encoding="utf-8") as f:
        for m in metas:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    print(f"[done] {len(metas)} chunks indexés.")

if __name__ == "__main__":
    main()
