import os, re, csv
from collections import deque
from urllib.parse import urljoin, urlparse
import requests
from bs4 import BeautifulSoup

START_URL = "https://www.sunat.gob.pe/legislacion/aduanera/index.html"
OUT_MANIFEST = "data/manifest.csv"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36"
}

MAX_DEPTH = 3
MAX_PAGES = 60  # evita crawl infinito

def guess_filetype(url: str) -> str:
    u = url.lower().split("?")[0]
    if u.endswith(".pdf"):
        return "pdf"
    if u.endswith(".txt"):
        return "txt"
    if u.endswith(".htm") or u.endswith(".html"):
        return "html"
    return "unknown"

def clean_title(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())[:200]

def is_allowed(url: str) -> bool:
    try:
        p = urlparse(url)
        if p.scheme not in ("http", "https"):
            return False
        # ✅ aceptar www, sin www y subdominios
        return p.netloc.endswith("sunat.gob.pe")
    except Exception:
        return False

def fetch(url: str) -> str:
    r = requests.get(url, headers=HEADERS, timeout=45)
    r.raise_for_status()
    return r.text

def extract_links(page_url: str, html: str):
    soup = BeautifulSoup(html, "html.parser")

    title = soup.title.get_text(strip=True) if soup.title else ""
    h1 = soup.find("h1")
    category = clean_title(h1.get_text()) if h1 else clean_title(title) or "SUNAT Aduanera"

    links = []
    for a in soup.find_all("a"):
        href = a.get("href")
        if not href:
            continue
        if href.startswith("mailto:") or href.startswith("javascript:"):
            continue

        full = urljoin(page_url, href)
        if not is_allowed(full):
            continue

        text = clean_title(a.get_text())
        links.append((full, text, category))
    return links

def should_follow(url: str) -> bool:
    """
    Decide si una URL parece sub-índice / página de listado.
    Se puede afinar, pero esto ya abre bastante el crawl.
    """
    u = url.lower()
    # seguir páginas relacionadas a legislación/aduanas
    keywords = ["legislacion", "aduan", "aduanera", "proced", "resol", "valor", "consulta"]
    return any(k in u for k in keywords)

def main():
    os.makedirs("data", exist_ok=True)

    visited_pages = set()
    seen_links = set()
    rows = []

    q = deque([(START_URL, 0)])

    while q and len(visited_pages) < MAX_PAGES:
        url, depth = q.popleft()
        if url in visited_pages:
            continue
        visited_pages.add(url)

        try:
            html = fetch(url)
        except Exception as e:
            print("ERR page:", url, e)
            continue

        for link_url, link_text, category in extract_links(url, html):
            if link_url in seen_links:
                continue
            seen_links.add(link_url)

            ftype = guess_filetype(link_url)

            rows.append({
                "doc_id": "",
                "title": link_text or "SIN_TITULO",
                "category": category,
                "source_url": link_url,
                "file_type": ftype,
                "file_path": "",
                "status": "discovered",
                "pages": "",
                "error": "",
            })

            # ✅ seguir subpáginas (HTML o unknown) hasta profundidad 3
            if depth < MAX_DEPTH and ftype in ("html", "unknown") and should_follow(link_url):
                q.append((link_url, depth + 1))

    # dedup final por URL
    uniq = {}
    for r in rows:
        uniq[r["source_url"]] = r
    rows = list(uniq.values())

    fieldnames = ["doc_id","title","category","source_url","file_type","file_path","status","pages","error"]
    with open(OUT_MANIFEST, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"OK: {len(rows)} urls → {OUT_MANIFEST}")
    print(f"Visited pages: {len(visited_pages)} (max {MAX_PAGES})")

if __name__ == "__main__":
    main()
