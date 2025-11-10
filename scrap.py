import json, requests, urllib.parse 

from trafilatura import load_html, extract
from config import BASE_URL, DATA_DIR, TARGET_URLS, USER_AGENT, REQUEST_TIMEOUT, sha16
from urllib.robotparser import RobotFileParser

# Verification du robots.txt

def init_robots(base_url: str) -> RobotFileParser:
    """
    Charge et parse le robots.txt du site de base.
    Retourne un objet RobotFileParser prêt à l'emploi.
    
    """

    rp = RobotFileParser()
    robots_url = urllib.parse.urljoin(base_url, "/robots.txt")
    try : 
        rp.set_url(robots_url)
        rp.read()
        print(f"[robots] Chargé depuis {robots_url}")
    except Exception as e :
        print(f"[robots] Impossible de lire robots.txt ({e}), on suppose tout autorisé.")
    return rp

def can_fetch(rp: RobotFileParser, url: str) -> bool:
    """
    Vérifie si l'URL est autorisée pour notre User-Agent
    """    
    try:
        return rp.can_fetch(USER_AGENT, url)
    except Exception:
        return True # en cas de doute, on autorise 
    

# Scrapping ciblé 

def scrape_targeted():
    # Initialiser robots.txt

    rp = init_robots(BASE_URL)

    out_path = f"{DATA_DIR}/pages.jsonl"
    out = open(out_path, "w", encoding="utf-8")
    saved = 0

    for url in TARGET_URLS:
        if not can_fetch(rp, url):
            print(f"[skip] Interdit par robots.txt : {url}")
            continue

        try:
            r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=REQUEST_TIMEOUT)
            if "text/html" not in r.headers.get("Content-Type", ""):
                print(f"[skip] Pas du HTML : {url}")
                continue

            html = r.text
            downloaded = load_html(html)
            data_json = extract(downloaded, output_format="json", no_fallback=True)
            if not data_json:
                print(f"[skip] Rien extrait : {url}")
                continue

            data = json.loads(data_json)
            text = (data.get("text") or "").strip()
            if len(text) < 400:
                print(f"[skip] Trop court : {url}")
                continue

            rec = {
                "id": sha16(url),
                "url": url,
                "title": (data.get("title") or "").strip(),
                "text": text
            }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            saved += 1
            print(f"[ok] {url}")

        except Exception as e:
            print(f"[err] {url} ({e})")
            continue

    out.close()
    print(f"[done] Pages enregistrées : {saved} -> {out_path}")


if __name__ == "__main__":
    scrape_targeted()