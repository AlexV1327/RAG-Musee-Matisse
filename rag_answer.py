# 06_rag_answer.py
from pathlib import Path
import json, os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import httpx, certifi

from config import MAX_CONTEXT_CHARS, INDEX_FILE, META_FILE, EMB_MODEL_NAME, TOP_K, CHUNKS_FILE
from query import load_meta  # doit exister (celui de 05_query.py)



def nice_title_from_url(url: str) -> str:
    try:
        slug = url.rstrip("/").split("/")[-1].replace("-", " ").strip()
        return slug.capitalize() or "[Sans titre]"
    except:
        return "[Sans titre]"

def load_chunks_by_id(path: Path):
    by_id = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                cid = rec.get("chunk_id")
                if cid:
                    by_id[cid] = rec  # rec["text"] contient le texte du chunk
            except:
                pass
    return by_id

def gather_context(query: str):
    # 1) charger index + meta + modèle d'embeddings
    index = faiss.read_index(str(INDEX_FILE))
    metas = load_meta(META_FILE)
    model = SentenceTransformer(EMB_MODEL_NAME)

    # 2) encoder la requête
    q_text = f"query: {query}" if "e5" in EMB_MODEL_NAME.lower() else query
    q_emb = model.encode([q_text], normalize_embeddings=True, convert_to_numpy=True).astype("float32")

    # 3) recherche FAISS
    scores, idxs = index.search(q_emb, TOP_K)
    idxs, scores = idxs[0], scores[0]

    # 4) charger les textes réels des chunks
    by_chunk = load_chunks_by_id(CHUNKS_FILE)

    # 5) assembler le contexte : vrais textes + sources (avec limite globale)
    snippets = []
    sources  = []
    total_len = 0
    PER_CHUNK_LIMIT = 900  # coupe un chunk trop long pour laisser entrer plusieurs sources
    seen_urls = set()      # <-- dédup par URL

    for i, s in zip(idxs, scores):
        if i < 0 or i >= len(metas):
            continue
        m = metas[i]
        url = m.get("url")
        if not url:
            continue

        # dédup par URL
        if url in seen_urls:
            continue
        seen_urls.add(url)

        # titre propre
        title = (m.get("title") or "").strip() or nice_title_from_url(url)
        src = f"{title} — {url}"

        rec = by_chunk.get(m.get("chunk_id"))
        if not rec:
            continue

        chunk_text = (rec.get("text") or "").strip()
        if not chunk_text:
            continue

        # limite douce par chunk (coupe sur un espace)
        if len(chunk_text) > PER_CHUNK_LIMIT:
            chunk_text = chunk_text[:PER_CHUNK_LIMIT].rsplit(" ", 1)[0] + " …"

        block = f"{chunk_text}\n(Source: {src})"

        # respect de la limite globale
        if total_len + len(block) + 1 > MAX_CONTEXT_CHARS:
            break

        snippets.append(block)
        sources.append(src)
        total_len += len(block) + 1

    # Fallback si rien n'a été ajouté (rare)
    if not snippets:
        ctx = "\n".join(f"[Source] {s}" for s in sources) if sources else "[Contexte vide]"
    else:
        ctx = "\n\n---\n\n".join(snippets)

    return ctx, sources

def build_prompt(query: str, context: str):
    return f"""Tu es un assistant qui répond STRICTEMENT à partir du contexte fourni ci-dessous.
- Ne parle PAS d’exécution de scripts, de Streamlit, d’IDE ou de commandes shell.
- Ne donne PAS de conseils génériques si l’info n’est pas dans le contexte.
- Si une information manque, dis-le explicitement.

[Contexte]
{context}

[Question]
{query}

Consignes de réponse:
- Français clair et concis.
- Structure en puces si pertinent.
- À chaque affirmation factuelle, ajoute la source entre parenthèses, au format exact :
  (Source: Titre — URL)
- N’invente rien et ne sors pas du sujet.
"""

def answer_with_mistral(prompt: str) -> str:

    API_BASE  = os.getenv("API_BASE", "https://api.mistral.ai")
    API_KEY   = os.getenv("API_KEY") or os.getenv("MISTRAL_API_KEY")
    API_MODEL = os.getenv("MODEL", "mistral-small-latest")

    if not API_KEY:
        raise RuntimeError("Clé API manquante. Exporte API_KEY ou MISTRAL_API_KEY.")

    url = f"{API_BASE.rstrip('/')}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": API_MODEL,
        "messages": [
            {"role": "system", "content": "Tu es un assistant utile et factuel. Cite les sources."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 400,
    }

    with httpx.Client(http2=False, verify=certifi.where(), timeout=30.0) as client:
        resp = client.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()

    try:
        return data["choices"][0]["message"]["content"]
    except Exception:
        raise RuntimeError(f"Réponse inattendue de l'API Mistral: {data}")



def main():
    if not Path(INDEX_FILE).exists() or not Path(META_FILE).exists():
        print("[err] Index ou métadonnées manquants. Lance d'abord 04_embed_index.py")
        return

    query = input("Votre question : ").strip()
    if not query:
        print("[err] Requête vide.")
        return

    ctx, sources = gather_context(query)
    prompt = build_prompt(query, ctx)

    try:
        ans = answer_with_mistral(prompt)
    except Exception as e:
        print("[err]", e)
        return

    print("\n=== RÉPONSE ===\n")
    print(ans)
    print("\n=== SOURCES ===")
    # dédup à l'affichage aussi
    for s in dict.fromkeys(sources):
        print("-", s)

if __name__ == "__main__":
    main()
