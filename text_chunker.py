from pathlib import Path
import json, re
from typing import List
from config import PAGES_FILE, CHUNKS_FILE, TARGET_CHARS, MIN_CHARS, OVERLAP_CHARS

SENT_SPLIT_RE = re.compile(r"(?<=[\.\!\?:;])\s+")

def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue

def split_into_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", (text or "").strip())
    if not text:
        return []
    parts = SENT_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]

def chunk_sentences(sents: List[str]) -> List[str]:
    chunks, cur = [], []
    cur_len = 0
    for s in sents:
        if not s:
            continue
        if cur_len + len(s) + 1 <= TARGET_CHARS:
            cur.append(s); cur_len += len(s) + 1
        else:
            if cur_len >= MIN_CHARS:
                chunks.append(" ".join(cur))
                # overlap simple: on repart avec la dernière phrase
                cur = [s]; cur_len = len(s)
            else:
                # phrase très longue: coupe forcée
                buf = (" ".join(cur) + " " + s).strip()
                chunks.append(buf[:TARGET_CHARS])
                rest = buf[TARGET_CHARS:]
                cur = [rest]; cur_len = len(rest)
    if cur and cur_len >= MIN_CHARS:
        chunks.append(" ".join(cur))
    # post-overlap
    if OVERLAP_CHARS > 0 and len(chunks) >= 2:
        padded = []
        for i, ch in enumerate(chunks):
            if i == 0:
                padded.append(ch)
            else:
                prev = chunks[i-1]
                overlap = prev[-OVERLAP_CHARS:]
                padded.append((overlap + " " + ch).strip())
        chunks = padded
    return chunks

def main():
    CHUNKS_FILE.parent.mkdir(parents=True, exist_ok=True)
    out = CHUNKS_FILE.open("w", encoding="utf-8")
    saved = 0
    for rec in iter_jsonl(PAGES_FILE):
        url = rec.get("url")
        title = (rec.get("title") or "").strip() or "[Sans titre]"
        text = rec.get("text") or ""
        sents = split_into_sentences(text)
        for i, chunk in enumerate(chunk_sentences(sents)):
            out.write(json.dumps({
                "url": url,
                "title": title,
                "chunk_id": i,
                "text": chunk,
            }, ensure_ascii=False) + "\n")
            saved += 1
    out.close()
    print(f"[done] chunks écrits: {saved} -> {CHUNKS_FILE}")

if __name__ == "__main__":
    main()
