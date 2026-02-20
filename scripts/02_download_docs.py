from __future__ import annotations

import re
import time
import unicodedata
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


MANIFEST = Path("data/manifest_filtered.csv")

OUT_PDF = Path("data/raw/pdf")
OUT_HTML = Path("data/raw/html")
OUT_TXT = Path("data/raw/txt")

TIMEOUT = 60
SLEEP_SEC = 0.25      # baja a 0.10 si va bien
MAX_DOWNLOADS = None  # pon 50 para prueba o None para todos


HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/122.0.0.0 Safari/537.36",
    "Accept": "text/html,application/pdf,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "es-PE,es;q=0.9,en;q=0.8",
    "Connection": "keep-alive",
}


def slugify(text: str, max_len: int = 90) -> str:
    if not text:
        return "sin_titulo"
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r'["“”]', "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    return (text[:max_len] or "sin_titulo")


def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(HEADERS)

    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s


def detect_type(resp: requests.Response, url: str, fallback: str) -> str:
    ct = (resp.headers.get("content-type") or "").lower()
    u = url.lower().split("?")[0]

    if "pdf" in ct or u.endswith(".pdf"):
        return "pdf"
    if "text/plain" in ct or u.endswith(".txt"):
        return "txt"
    if "text/html" in ct or u.endswith(".html") or u.endswith(".htm"):
        return "html"
    return fallback or "html"


def safe_filename(title: str, doc_id: str, ext: str) -> str:
    # slug + doc_id para trazabilidad
    slug = slugify(title)
    return f"{slug}_{doc_id}.{ext}"


def main():
    df = pd.read_csv(MANIFEST)

    # asegurar columnas
    for col in ["file_path", "status", "error"]:
        if col not in df.columns:
            df[col] = ""

    OUT_PDF.mkdir(parents=True, exist_ok=True)
    OUT_HTML.mkdir(parents=True, exist_ok=True)
    OUT_TXT.mkdir(parents=True, exist_ok=True)

    session = make_session()

    downloaded = 0
    skipped = 0
    failed = 0

    for i, row in df.iterrows():
        if MAX_DOWNLOADS is not None and downloaded >= MAX_DOWNLOADS:
            break

        url = str(row.get("source_url") or "").strip()
        doc_id = str(row.get("doc_id") or "").strip()
        title = str(row.get("title") or "").strip()
        fallback_type = str(row.get("file_type") or "").strip().lower()
        status = str(row.get("status") or "").strip().lower()
        file_path = str(row.get("file_path") or "").strip()

        if not url or not doc_id:
            df.at[i, "status"] = "failed"
            df.at[i, "error"] = "missing url/doc_id"
            failed += 1
            continue

        # si dice descargado y el archivo existe, skip
        if status == "downloaded" and file_path and Path(file_path).exists():
            skipped += 1
            continue

        try:
            r = session.get(url, timeout=TIMEOUT, allow_redirects=True)

            if r.status_code != 200:
                df.at[i, "status"] = "failed"
                df.at[i, "error"] = f"http {r.status_code}"
                failed += 1
                time.sleep(SLEEP_SEC)
                continue

            ftype = detect_type(r, url, fallback_type)
            df.at[i, "file_type"] = ftype

            # valida pdf real
            if ftype == "pdf" and not r.content.startswith(b"%PDF"):
                df.at[i, "status"] = "failed"
                df.at[i, "error"] = "not a real pdf"
                failed += 1
                time.sleep(SLEEP_SEC)
                continue

            if ftype == "pdf":
                out_dir = OUT_PDF
                ext = "pdf"
                content = r.content
                mode = "wb"
            elif ftype == "txt":
                out_dir = OUT_TXT
                ext = "txt"
                content = r.text
                mode = "w"
            else:
                out_dir = OUT_HTML
                ext = "html"
                content = r.text
                mode = "w"

            filename = safe_filename(title if title else "sin_titulo", doc_id, ext)
            out_path = out_dir / filename

            # escribir
            if mode == "wb":
                out_path.write_bytes(content)
            else:
                out_path.write_text(content, encoding="utf-8")

            df.at[i, "file_path"] = str(out_path).replace("\\", "/")
            df.at[i, "status"] = "downloaded"
            df.at[i, "error"] = ""

            downloaded += 1
            print(f"OK  {doc_id}  {ftype}  -> {out_path.name}")

            time.sleep(SLEEP_SEC)

        except Exception as e:
            df.at[i, "status"] = "failed"
            df.at[i, "error"] = f"{type(e).__name__}: {str(e)[:160]}"
            failed += 1
            print(f"ERR {doc_id} -> {e}")
            time.sleep(SLEEP_SEC)

    df.to_csv(MANIFEST, index=False, encoding="utf-8")

    print("\n=== RESUMEN ===")
    print(f"Downloaded: {downloaded}")
    print(f"Skipped (ya descargados): {skipped}")
    print(f"Failed: {failed}")
    print(f"Manifest actualizado: {MANIFEST}")


if __name__ == "__main__":
    main()
