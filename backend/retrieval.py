from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional

import faiss
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


@dataclass
class Evidence:
    chunk_id: str
    doc_id: str
    title: str
    category: str
    page: int
    article: str | None
    text: str
    score: float
    source: str  # "faiss" | "bm25" | "hybrid" | "rerank"


class HybridRetriever:
    """
    Carga índices (FAISS + BM25) + meta y permite:
    - faiss_topk(query)
    - bm25_topk(query)
    - hybrid_topk(query) -> merge + dedup
    - rerank(query, evidences) -> CrossEncoder opcional
    """

    def __init__(
        self,
        index_dir: str = "data/index",
        embed_model: str = "intfloat/multilingual-e5-large",
        ef_search: int = 64,
        rerank_model: Optional[str] = None,
        rerank_device: str = "cpu",
    ):
        self.index_dir = Path(index_dir)

        self.meta = pd.read_parquet(self.index_dir / "meta.parquet").reset_index(drop=True)

        self.index = faiss.read_index(str(self.index_dir / "faiss_hnsw.index"))
        if hasattr(self.index, "hnsw"):
            self.index.hnsw.efSearch = ef_search

        with open(self.index_dir / "bm25.pkl", "rb") as f:
            self.bm25: BM25Okapi = pickle.load(f)

        self.model = SentenceTransformer(embed_model)

        # Cross-encoder reranker opcional
        self.reranker = None
        if rerank_model:
            from sentence_transformers import CrossEncoder
            self.reranker = CrossEncoder(rerank_model, device=rerank_device)

        self._query_cache: dict = {}
        self.n = len(self.meta)

    STOPWORDS_ES = {
        "de", "la", "el", "en", "y", "a", "los", "del", "las", "un", "una",
        "por", "con", "para", "se", "que", "es", "al", "lo", "como", "más",
        "su", "sus", "son", "pero", "o", "si", "fue", "ha", "le", "ya",
        "este", "esta", "estos", "estas", "ese", "esa", "esos", "esas",
        "ser", "no", "han", "hay", "también", "cuando", "sobre", "entre",
        "hasta", "desde", "ante", "bajo", "sin", "tras", "durante",
    }

    @staticmethod
    def _tokenize_bm25(text: str) -> List[str]:
        tokens = text.lower().split()
        return [t for t in tokens if len(t) > 2 and t not in HybridRetriever.STOPWORDS_ES]
    

    def _row_to_evidence(self, idx: int, score: float, source: str) -> Evidence:
        r = self.meta.iloc[int(idx)]
        return Evidence(
            chunk_id=str(r["chunk_id"]),
            doc_id=str(r["doc_id"]),
            title=str(r.get("title", "")),
            category=str(r.get("category", "")),
            page=int(r.get("page", 0) or 0),
            article=(None if pd.isna(r.get("article")) else str(r.get("article"))),
            text=str(r.get("text", "")),
            score=float(score),
            source=source,
        )

    def faiss_topk(self, query: str, top_k: int = 5) -> List[Evidence]:
        if query not in self._query_cache:
            self._query_cache[query] = self.model.encode(
                [f"query: {query}"], normalize_embeddings=True
            ).astype("float32")
        q_emb = self._query_cache[query]
        scores, idxs = self.index.search(q_emb, top_k)
        idxs = idxs[0]
        scores = scores[0]

        out: List[Evidence] = []
        for i, s in zip(idxs, scores):
            if i < 0:
                continue
            out.append(self._row_to_evidence(int(i), float(s), "faiss"))
        return out

    def bm25_topk(self, query: str, top_k: int = 5) -> List[Evidence]:
        toks = self._tokenize_bm25(query)
        scores = self.bm25.get_scores(toks)
        idxs = np.argsort(scores)[::-1][:top_k]

        out: List[Evidence] = []
        for i in idxs:
            out.append(self._row_to_evidence(int(i), float(scores[int(i)]), "bm25"))
        return out

    def hybrid_topk(
        self,
        query: str,
        top_k_faiss: int = 6,
        top_k_bm25: int = 6,
        final_k: int = 6,
        w_faiss: float = 0.6,
        w_bm25: float = 0.4,
    ) -> List[Evidence]:
        fa = self.faiss_topk(query, top_k_faiss)
        bm = self.bm25_topk(query, top_k_bm25)

        def norm_scores(items: List[Evidence]) -> Dict[str, float]:
            if not items:
                return {}
            vals = np.array([e.score for e in items], dtype="float32")
            vmin, vmax = float(vals.min()), float(vals.max())
            if vmax - vmin < 1e-8:
                return {e.chunk_id: 1.0 for e in items}
            return {e.chunk_id: float((e.score - vmin) / (vmax - vmin)) for e in items}

        nfa = norm_scores(fa)
        nbm = norm_scores(bm)

        pool: Dict[str, Evidence] = {}
        for e in fa:
            pool[e.chunk_id] = e
        for e in bm:
            pool.setdefault(e.chunk_id, e)

        merged: List[Evidence] = []
        for cid, e in pool.items():
            s = w_faiss * nfa.get(cid, 0.0) + w_bm25 * nbm.get(cid, 0.0)
            if cid in nfa and cid in nbm:
                src = "hybrid"
            elif cid in nfa:
                src = "faiss"
            else:
                src = "bm25"
            merged.append(Evidence(**{**e.__dict__, "score": float(s), "source": src}))

        merged.sort(key=lambda x: x.score, reverse=True)
        return merged[:final_k]

    def rerank(self, query: str, evidences: List[Evidence], top_n: int = 8) -> List[Evidence]:
        if not evidences:
            return []
        if self.reranker is None:
            return evidences[:top_n]

        pairs = [(query, e.text[:1200]) for e in evidences]
        scores = self.reranker.predict(pairs)

        out: List[Evidence] = []
        for e, s in zip(evidences, scores):
            out.append(Evidence(**{**e.__dict__, "score": float(s), "source": "rerank"}))

        out.sort(key=lambda x: x.score, reverse=True)
        return out[:top_n]