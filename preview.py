from pathlib import Path 
import json 
import textwrap
from config import PAGES_FILE

def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line :
                continue 
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue 

def main():
    if not PAGES_FILE.exists():
        print(f"[err] Fichier introuvable : {PAGES_FILE}")
        return 
    
    total = 0 
    print(f"[info] Aperçu de {PAGES_FILE} :\n")
    for i, rec in enumerate(iter_jsonl(PAGES_FILE), start=1):
        total += 1 
        title = (rec.get("title") or "").strip() or "[Sans titre]"
        url = rec.get("url", "")
        text = (rec.get("text") or "").strip()
        snippet = textwrap.shorten(text, width=220, placeholder = "...")
        print(f"{i:02d}. {title}")
        print(f"    URL   : {url}")
        print(f"    Extrait: {snippet}\n")

    print(f"[done] {total} enregistrements trouvés.")

if __name__ == "__main__":
    main()
