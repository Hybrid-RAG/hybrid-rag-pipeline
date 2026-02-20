from pathlib import Path
import re
import pandas as pd
from tqdm import tqdm
from bs4 import BeautifulSoup
from pypdf import PdfReader

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


def extract_articles(text: str):
    """
    Detecta 'Artículo 12', 'Art. 15', etc.
    """
    matches = re.findall(r"(Artículo\s+\d+[A-Za-z\-]*)", text, flags=re.IGNORECASE)
    return matches[0] if matches else None


def chunk_text(text: str, size=800, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunk = text[start:end]
        chunks.append(chunk)
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
    html = path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "html.parser")

    # eliminar scripts y estilos
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
