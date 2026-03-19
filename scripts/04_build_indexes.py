from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


# =========================
# CONFIG
# =========================
CHUNKS_PATH = Path("data/processed/chunks.parquet")
OUT_DIR = Path("data/index")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Embedding recomendado para español (legal): mejor calidad que MiniLM
EMBED_MODEL = "intfloat/multilingual-e5-large"
BATCH_SIZE = 64

# FAISS HNSW config
HNSW_M = 32
EF_CONSTRUCTION = 200
EF_SEARCH_DEFAULT = 64


# =========================
# BM25 tokenization simple
# =========================
STOPWORDS_ES = {
    "de", "la", "el", "en", "y", "a", "los", "del", "las", "un", "una",
    "por", "con", "para", "se", "que", "es", "al", "lo", "como", "más",
    "su", "sus", "son", "pero", "o", "si", "fue", "ha", "le", "ya",
    "este", "esta", "estos", "estas", "ese", "esa", "esos", "esas",
    "ser", "no", "han", "hay", "también", "cuando", "sobre", "entre",
    "hasta", "desde", "ante", "bajo", "sin", "tras", "durante",
}

def bm25_tokenize(text: str):
    tokens = text.lower().split()
    return [t for t in tokens if len(t) > 2 and t not in STOPWORDS_ES]


# =========================
# Main
# =========================
def main():
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(f"No existe: {CHUNKS_PATH}")

    chunks = pd.read_parquet(CHUNKS_PATH)

    # aseguramos columnas mínimas
    required = {"chunk_id", "doc_id", "page", "text"}
    if not required.issubset(set(chunks.columns)):
        raise ValueError(f"chunks.parquet debe incluir: {required}")

    # ---- metadata (id->row)
    chunks = chunks.reset_index(drop=True)
    meta = chunks[["chunk_id", "doc_id", "title", "category", "page", "article", "text"]].copy()
    meta_path = OUT_DIR / "meta.parquet"
    meta.to_parquet(meta_path, index=False)

    # =========================
    # EMBEDDINGS
    # =========================
    print(f"Loading embedding model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)

    texts = chunks["text"].astype(str).tolist()

    # e5 recomienda prefijo "query:" / "passage:" (mejora performance)
    passages = [f"passage: {t}" for t in texts]

    print(f"Encoding {len(passages)} passages...")
    emb = model.encode(
        passages,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,  # important para cosine
    ).astype("float32")

    emb_path = OUT_DIR / "embeddings.npy"
    np.save(emb_path, emb)

    # =========================
    # FAISS HNSW (cosine)
    # =========================
    d = emb.shape[1]
    print(f"Building FAISS HNSW index: dim={d}, M={HNSW_M}")

    index = faiss.IndexHNSWFlat(d, HNSW_M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = EF_CONSTRUCTION
    index.hnsw.efSearch = EF_SEARCH_DEFAULT

    index.add(emb)

    faiss_path = OUT_DIR / "faiss_hnsw.index"
    faiss.write_index(index, str(faiss_path))

    # =========================
    # BM25
    # =========================
    print("Building BM25...")
    tokenized = [bm25_tokenize(t) for t in texts]
    bm25 = BM25Okapi(tokenized)

    bm25_path = OUT_DIR / "bm25.pkl"
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25, f)

    # =========================
    # Config snapshot
    # =========================
    cfg = {
        "embed_model": EMBED_MODEL,
        "batch_size": BATCH_SIZE,
        "n_chunks": len(chunks),
        "embedding_dim": int(d),
        "faiss": {
            "type": "IndexHNSWFlat",
            "metric": "INNER_PRODUCT (cosine with normalized vectors)",
            "M": HNSW_M,
            "efConstruction": EF_CONSTRUCTION,
            "efSearch_default": EF_SEARCH_DEFAULT,
        },
        "bm25": {
            "tokenizer": "lower().split() with len>2 filter",
        },
        "paths": {
            "chunks": str(CHUNKS_PATH),
            "meta": str(meta_path),
            "embeddings": str(emb_path),
            "faiss": str(faiss_path),
            "bm25": str(bm25_path),
        }
    }

    with open(OUT_DIR / "index_config.pkl", "wb") as f:
        pickle.dump(cfg, f)

    print("\n=== DONE ===")
    print(f"Meta: {meta_path}")
    print(f"Embeddings: {emb_path}")
    print(f"FAISS: {faiss_path}")
    print(f"BM25: {bm25_path}")
    print(f"Chunks: {len(chunks)}")


if __name__ == "__main__":
    main()