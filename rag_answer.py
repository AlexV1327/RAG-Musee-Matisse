import os, time, random, json
from typing import List, Tuple
import httpx
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from config import (
    MAX_CONTEXT_CHARS, EMB_MODEL_NAME, TOP_K, META_FILE, INDEX_FILE,
    API_MODEL, MISTRAL_API_KEY
)

def load_meta(path):
    metas = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                metas.append(json.loads(line))
            except Exception:
                continue
    return metas

def nice_title_from_url(url: str) -> str:
    try:
        slug = url.rstrip("/").split("/")[-1].replace("-", " ").strip()
        return slug or url
    except Exception:
        return url

def _load_index_and_meta():
    metas = load_meta(META_FILE)
    index = faiss.read_index(str(INDEX_FILE))
    model = SentenceTransformer(EMB_MODEL_NAME)
    return model, index, metas

def gather_context(query: str, top_k: int = TOP_K) -> Tuple[str, List[str]]:
    model, index, metas = _load_index_and_meta()
    qtext = f"query: {query}" if "e5" in EMB_MODEL_NAME.lower() else query
    qv = model.encode([qtext], normalize_embeddings=True).astype("float32")
    scores, idxs = index.search(qv, top_k)
    scores, idxs = scores[0], idxs[0]

    blocks, sources = [], []
    for i, s in zip(idxs, scores):
        if i < 0 or i >= len(metas):
            continue
        m = metas[i]
        src = f"{m.get('title') or nice_title_from_url(m.get('url',''))} — {m.get('url')}"
        sources.append(src)
        block = f"[score={s:.3f}]\nTITLE: {m.get('title')}\nURL: {m.get('url')}\nTEXT: {m.get('text')}"
        blocks.append(block)

    ctx = "\n\n---\n\n".join(blocks)
    if len(ctx) > MAX_CONTEXT_CHARS:
        ctx = ctx[:MAX_CONTEXT_CHARS] + "\n[...trunc]\n"
    return ctx, sources

def build_prompt(user_query: str, context: str) -> str:
    return (
        "Tu es un assistant pour un musée. Réponds en français, concis et factuel.\n"
        "Utilise uniquement les informations du CONTEXTE fourni. "
        "Si l'information n'y est pas, dis que tu ne sais pas.\n"
        "Quand c'est pertinent, cite brièvement les sources (titres).\n\n"
        f"QUESTION:\n{user_query}\n\n"
        f"CONTEXTE:\n{context}\n\n"
        "RÉPONSE:"
    )

def answer_with_mistral(prompt: str) -> str:
    if not MISTRAL_API_KEY:
        raise RuntimeError("MISTRAL_API_KEY manquant dans l'environnement.")

    url = "https://api.mistral.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": API_MODEL,
        "messages": [
            {"role": "system", "content": "Tu es un assistant fiable, tu t'en tiens au contexte."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 512,
    }

    MAX_RETRIES = int(os.getenv("MISTRAL_MAX_RETRIES", "6"))
    INITIAL_BACKOFF = float(os.getenv("MISTRAL_INITIAL_BACKOFF", "0.8"))
    JITTER = 0.25

    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            with httpx.Client(timeout=60) as client:
                r = client.post(url, headers=headers, json=payload)
                # Rate limit ou 5xx => retry avec backoff
                if r.status_code == 429 or 500 <= r.status_code < 600:
                    wait = INITIAL_BACKOFF * (2 ** attempt) + random.uniform(0, JITTER)
                    ra = r.headers.get("Retry-After")
                    if ra:
                        try:
                            wait = max(wait, float(ra))
                        except Exception:
                            pass
                    time.sleep(wait)
                    last_err = r
                    continue
                r.raise_for_status()
                data = r.json()
                return data["choices"][0]["message"]["content"].strip()
        except httpx.HTTPStatusError as e:
            if e.response is not None and e.response.status_code in (401, 403):
                detail = ""
                try:
                    detail = e.response.json().get("error", {}).get("message", "")
                except Exception:
                    pass
                raise RuntimeError(
                    f"Erreur d'authentification/autorisation Mistral (HTTP {e.response.status_code}). {detail}".strip()
                )
            last_err = e
        except Exception as e:
            last_err = e
            time.sleep(INITIAL_BACKOFF * (2 ** attempt) + random.uniform(0, JITTER))

    # Fallback extractif: renvoyer les top passages si l'API reste indispo
    return (
        "⚠️ Impossible de contacter le LLM (limite/incident). "
        "Voici les meilleurs passages trouvés dans le corpus :\n\n"
        + prompt.split("CONTEXTE:", 1)[-1].strip()[:800]
        + ("\n\n[…troncature…]" if len(prompt) > 800 else "")
    )

def main():
    import sys
    q = " ".join(sys.argv[1:]) or "Quels sont les horaires d'ouverture ?"
    ctx, sources = gather_context(q)
    prompt = build_prompt(q, ctx)
    ans = answer_with_mistral(prompt)
    print("\n=== RÉPONSE ===\n\n" + ans)
    print("\n=== SOURCES ===")
    for s in dict.fromkeys(sources):
        print("-", s)

if __name__ == "__main__":
    main()
