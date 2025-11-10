import time, json, urllib.parse, requests
from urllib.robotparser import RobotFileParser
from trafilatura import load_html, extract
from config import BASE_URL, DATA_DIR, TARGET_URLS, USER_AGENT, REQUEST_TIMEOUT

PAGES_FILE = DATA_DIR / "pages.jsonl"

def init_robots() -> RobotFileParser:
    rp = RobotFileParser()
    rp.set_url(urllib.parse.urljoin(BASE_URL, "/robots.txt"))
    rp.read()
    return rp

def fetch(url: str) -> str:
    r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.text

def scrape_targeted():
    rp = init_robots()
    out = PAGES_FILE.open("w", encoding="utf-8")
    saved = 0
    for url in TARGET_URLS:
        if not rp.can_fetch(USER_AGENT, url):
            print(f"[skip robots] {url}")
            continue
        try:
            html = fetch(url)
            doc = load_html(html)
            text = extract(doc, include_links=False, no_fallback=True) or ""
            rec = {
                "url": url,
                "title": None,  # pas critique, le texte prime dans le RAG
                "text": text.strip(),
            }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            saved += 1
            time.sleep(0.5)
            print(f"[ok] {url}")
        except Exception as e:
            print(f"[err] {url}: {e}")
    out.close()
    print(f"[done] pages enregistrées: {saved} -> {PAGES_FILE}")

if __name__ == "__main__":
    scrape_targeted()
