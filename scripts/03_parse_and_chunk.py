from pathlib import Path
import re
import pandas as pd
from tqdm import tqdm
from bs4 import BeautifulSoup
from pypdf import PdfReader
import chardet

# =========================
# CONFIG
# =========================

MANIFEST = Path("data/manifest_filtered.csv")
OUT_DIR = Path("data/processed")
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

OUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# UTILIDADES
# =========================

def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\x00", "", text)
    return text.strip()

def extract_title(soup) -> str:
    # Intentar <title>
    if soup.title and soup.title.get_text(strip=True):
        t = clean_text(soup.title.get_text())
        if len(t) > 5:
            return t
    # Intentar primer h1, h2, h3
    for tag in ["h1", "h2", "h3"]:
        el = soup.find(tag)
        if el and el.get_text(strip=True):
            return clean_text(el.get_text())
    return ""

def extract_articles(text: str):
    """
    Detecta 'Artículo 12', 'Art. 15', etc.
    """
    matches = re.findall(r"(Artículo\s+\d+[A-Za-z\-]*)", text, flags=re.IGNORECASE)
    return matches[0] if matches else None


def chunk_text(text: str, size=800, overlap=100):
    # Intentar dividir por artículos primero
    article_pattern = re.compile(
        r'(?=Art[ií]culo\s+\d+|Art\.\s*\d+|ARTÍCULO\s+\d+)',
        re.IGNORECASE
    )
    splits = article_pattern.split(text)
    
    # Si encontró artículos y son de tamaño razonable, usarlos
    if len(splits) > 1:
        chunks = []
        for split in splits:
            split = split.strip()
            if not split:
                continue
            # Si el artículo es muy largo, subdividirlo
            if len(split) > size:
                start = 0
                while start < len(split):
                    chunks.append(split[start:start + size])
                    start += size - overlap
            else:
                chunks.append(split)
        return chunks

    # Fallback: chunking por caracteres normal
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks


def parse_pdf(path: Path):
    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append((i + 1, clean_text(text)))
    return pages


def parse_html(path: Path):
    import chardet
    import requests

    raw = path.read_bytes()
    detected = chardet.detect(raw)
    enc = detected.get("encoding") or "utf-8"
    try:
        html = raw.decode(enc, errors="replace")
    except Exception:
        html = raw.decode("utf-8", errors="replace")

    soup = BeautifulSoup(html, "html.parser")

    # Detectar frameset y seguir al frame con contenido
    frames = soup.find_all("frame")
    if frames:
        HEADERS = {"User-Agent": "Mozilla/5.0"}
        texts = []
        for frame in frames:
            src = frame.get("src", "")
            if not src or src.startswith("javascript"):
                continue
            # Construir URL absoluta si es relativa
            if src.startswith("http"):
                frame_url = src
            else:
                # Intentar leer desde el mismo directorio del archivo
                frame_path = path.parent / src
                if frame_path.exists():
                    frame_raw = frame_path.read_bytes()
                    frame_enc = chardet.detect(frame_raw).get("encoding") or "utf-8"
                    frame_html = frame_raw.decode(frame_enc, errors="replace")
                    frame_soup = BeautifulSoup(frame_html, "html.parser")
                    for tag in frame_soup(["script", "style", "nav", "footer"]):
                        tag.decompose()
                    texts.append(clean_text(frame_soup.get_text(separator=" ")))
                    continue
                # Si no existe localmente, intentar descargarlo
                base_url = "https://www.sunat.gob.pe"
                frame_url = base_url + "/" + src.lstrip("/")

            try:
                r = requests.get(frame_url, headers=HEADERS, timeout=30)
                r.raise_for_status()
                frame_enc = chardet.detect(r.content).get("encoding") or "utf-8"
                frame_html = r.content.decode(frame_enc, errors="replace")
                frame_soup = BeautifulSoup(frame_html, "html.parser")
                for tag in frame_soup(["script", "style", "nav", "footer"]):
                    tag.decompose()
                texts.append(clean_text(frame_soup.get_text(separator=" ")))
            except Exception as e:
                print(f"  [frame skip] {frame_url}: {e}")

        combined = " ".join(texts)
        if combined.strip():
            return [(1, combined)]

    # HTML normal
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()
    text = soup.get_text(separator=" ")
    return [(1, clean_text(text))]


# =========================
# MAIN
# =========================

def main():
    df = pd.read_csv(MANIFEST)

    chunks_data = []
    docs_data = []

    chunk_id_counter = 0

    for _, row in tqdm(df.iterrows(), total=len(df)):

        if row.get("status") != "downloaded":
            continue

        file_path = Path(row["file_path"])
        doc_id = row["doc_id"]
        title = row.get("title", "")
        category = row.get("category", "")
        file_type = row.get("file_type", "")

        if not file_path.exists():
            continue

        try:
            if file_type == "pdf":
                pages = parse_pdf(file_path)
            else:
                pages = parse_html(file_path)
                # Extraer título si está vacío o es SIN_TITULO
                if not title or title == "SIN_TITULO":
                    raw = file_path.read_bytes()
                    import chardet
                    enc = chardet.detect(raw).get("encoding") or "utf-8"
                    html = raw.decode(enc, errors="replace")
                    from bs4 import BeautifulSoup
                    soup_tmp = BeautifulSoup(html, "html.parser")
                    extracted = extract_title(soup_tmp)
                    if extracted:
                        title = extracted

            docs_data.append({
                "doc_id": doc_id,
                "title": title,
                "category": category,
                "file_path": str(file_path),
                "n_pages": len(pages)
            })

            for page_num, page_text in pages:
                if not page_text:
                    continue

                article = extract_articles(page_text)

                chunks = chunk_text(
                    page_text,
                    size=CHUNK_SIZE,
                    overlap=CHUNK_OVERLAP
                )

                for chunk in chunks:
                    if len(chunk.strip()) < 50:
                        continue

                    chunk_id_counter += 1

                    chunks_data.append({
                        "chunk_id": f"chunk_{chunk_id_counter:07d}",
                        "doc_id": doc_id,
                        "title": title,
                        "category": category,
                        "page": page_num,
                        "article": article,
                        "text": chunk
                    })

        except Exception as e:
            print(f"Error procesando {doc_id}: {e}")

    chunks_df = pd.DataFrame(chunks_data)
    docs_df = pd.DataFrame(docs_data)

    chunks_df.to_parquet(OUT_DIR / "chunks.parquet", index=False)
    docs_df.to_parquet(OUT_DIR / "documents.parquet", index=False)

    print("\n=== DONE ===")
    print(f"Documents: {len(docs_df)}")
    print(f"Chunks: {len(chunks_df)}")
    print("Saved to data/processed/")


if __name__ == "__main__":
    main()
