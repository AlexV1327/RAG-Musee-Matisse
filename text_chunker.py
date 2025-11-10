from pathlib import Path 
import json
import re
from typing import Iterable, Dict, List

from config import PAGES_FILE, CHUNKS_FILE, TARGET_CHARS, MIN_CHARS, OVERLAP_CHARS
from preview import iter_jsonl



# Regex qui trouve la fin des phrases
SENT_SPLIT_RE = re.compile(r'(?<=[\.\!\?\:\;])\s+')  # coupe après la ponctuation forte

# Découpe en phrases 
def split_into_sentences(text: str) -> List[str]:
    """
    Découpe un gros en liste de phrases 
    """
    parts = SENT_SPLIT_RE.split(text.strip())
    return [p.strip() for p in parts if p and len(p.strip()) > 0]

# Fonction qui fabrique les chunks

def build_chunks(text: str,
                 target_chars: int = TARGET_CHARS,
                 min_chars: int = MIN_CHARS,
                 overlap_chars: int = OVERLAP_CHARS) -> List[str]:
    """
    Construit des chunks ~target_chars à partir de phrases,
    avec un chevauchement de overlap_chars entre chunks consécutifs.
    """
    sents = split_into_sentences(text)
    chunks = []
    buf = ""

    for s in sents:
        if not buf:
            buf = s
            continue
        # si ajouter la phrase dépasse la cible, on "flush" le buffer
        if len(buf) + 1 + len(s) >= target_chars:
            # si trop court, on force l’ajout de la phrase pour éviter un micro-chunk
            if len(buf) < min_chars:
                buf = buf + " " + s
            chunks.append(buf.strip())
            # chevauchement : on reprend la "fin" du buffer précédent
            if overlap_chars > 0:
                tail = buf[-overlap_chars:]
                # évite de couper un mot au milieu
                tail = tail[tail.find(" ")+1:] if " " in tail else tail
            else:
                tail = ""
            buf = (tail + " " + s).strip()
        else:
            buf = buf + " " + s

    if buf and len(buf.strip()) >= min_chars:
        chunks.append(buf.strip())

    # cas extrême : tout est trop court → garder 1 chunk quand même
    if not chunks and text.strip():
        chunks = [text.strip()]
    return chunks

def estimate_tokens(s: str) -> int:
    # Estimation grossière: ~4 caractères par token (FR/EN)
    return max(1, int(len(s) / 4))


# Lire les pages et produire les chunks

def main():
    if not PAGES_FILE.exists():
        print(f"[err] Fichier introuvable : {PAGES_FILE}")
        return

    CHUNKS_FILE.parent.mkdir(parents=True, exist_ok=True)
    out = CHUNKS_FILE.open("w", encoding="utf-8")

    total_docs = 0
    total_chunks = 0

    for rec in iter_jsonl(PAGES_FILE):
        total_docs += 1
        doc_id = rec.get("id") or ""
        url = rec.get("url") or ""
        title = (rec.get("title") or "").strip()
        text = (rec.get("text") or "").strip()
        if not text:
            continue

        chunks = build_chunks(text)
        for j, ch in enumerate(chunks):
            item = {
                "doc_id": doc_id,
                "chunk_id": f"{doc_id}-{j:03d}",
                "url": url,
                "title": title,
                "text": ch,
                "n_chars": len(ch),
                "n_tokens_est": estimate_tokens(ch),
            }
            out.write(json.dumps(item, ensure_ascii=False) + "\n")
            total_chunks += 1

    out.close()
    print(f"[done] documents traités : {total_docs}")
    print(f"[done] chunks écrits     : {total_chunks} -> {CHUNKS_FILE}")

if __name__ == "__main__":
    main()



















