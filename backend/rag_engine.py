from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Tuple

import numpy as np
from huggingface_hub import InferenceClient

from backend.retrieval import HybridRetriever, Evidence


# -----------------------------
# Data structures
# -----------------------------
@dataclass
class SentenceCitation:
    sentence: str
    citations: List[str]  # "doc_id:pX"
    grounding: float      # grounding por oración (0-1)


@dataclass
class RAGResult:
    answer: str
    grounding_score: float
    hallucination_flag: bool
    citations_per_sentence: List[SentenceCitation]
    evidences: List[Evidence]
    debug: Dict


# -----------------------------
# Retriever singleton
# -----------------------------
_RETRIEVER: Optional[HybridRetriever] = None


def _default_index_dir() -> str:
    # 1) prioridad: variable de entorno
    env = os.getenv("RAG_INDEX_DIR")
    if env:
        return env

    # 2) fallback: relativo al archivo (rag_pipeline/backend/rag_engine.py)
    here = Path(__file__).resolve()
    base = here.parents[1]  # .../rag_pipeline/backend -> .../rag_pipeline
    return str(base / "data" / "index")


def get_retriever() -> HybridRetriever:
    global _RETRIEVER
    if _RETRIEVER is None:
        # Si quieres activar rerank, descomenta:
        # _RETRIEVER = HybridRetriever(index_dir=_default_index_dir(), rerank_model="BAAI/bge-reranker-base", rerank_device="cpu")
        _RETRIEVER = HybridRetriever(index_dir=_default_index_dir())
    return _RETRIEVER


# -----------------------------
# Helpers
# -----------------------------
_SENT_SPLIT = re.compile(r"(?<=[\.\!\?])\s+")


def split_sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text.strip()) if s.strip()]


def build_prompt(question: str, evidences: List[Evidence]) -> str:
    ctx = []
    for i, e in enumerate(evidences, 1):
        cite = f"{e.doc_id}:p{e.page}"
        ctx.append(f"[{i}] ({cite}) {e.text[:900]}")
    context_block = "\n\n".join(ctx)

    return f"""Eres un asistente experto en legislación aduanera peruana (SUNAT).
Reglas:
- Responde SOLO usando la evidencia proporcionada.
- Si falta sustento, dilo explícitamente.
- Escribe en español técnico, claro.
- No inventes artículos, números, ni resoluciones.
- Estructura tu respuesta en oraciones cortas.

PREGUNTA:
{question}

EVIDENCIA:
{context_block}

RESPUESTA:
"""


def normalize_retrieval_mode(retrieval_mode: str) -> str:
    """
    Acepta modos internos ('hybrid', 'faiss', 'bm25', 'hybrid+rerank')
    y también los labels bonitos del dashboard.
    """
    if not retrieval_mode:
        return "hybrid"

    # Si el dashboard manda labels bonitos:
    label_map = {
        "Hybrid (BM25 + FAISS)": "hybrid",
        "FAISS only (HNSW)": "faiss",
        "BM25 only": "bm25",
        "Hybrid + Rerank (CrossEncoder)": "hybrid+rerank",
    }
    if retrieval_mode in label_map:
        return label_map[retrieval_mode]

    # Si ya viene interno:
    mode = retrieval_mode.strip().lower()
    aliases = {
        "hybrid (bm25 + faiss)": "hybrid",
        "faiss only (hnsw)": "faiss",
        "bm25 only": "bm25",
    }
    return aliases.get(mode, mode)


def call_hf_llm(
    prompt: str,
    model_name: str,
    max_new_tokens: int = 320,
    temperature: float = 0.0,
    retries: int = 2,
) -> str:
    """
    Llama a HF Inference API. Manejo simple de 401/429 con retry.
    """
    token = os.getenv("HF_TOKEN")  # desde .env o variable de entorno
    client = InferenceClient(model=model_name, token=token)

    last_err: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            # Chat completions si el backend lo soporta
            try:
                resp = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": "Eres un asistente experto en legislación aduanera peruana (SUNAT)."},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                )
                return (resp.choices[0].message.content or "").strip()
            except Exception:
                out = client.text_generation(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=(temperature > 0.0),
                    return_full_text=False,
                )
                return (out or "").strip()

        except Exception as e:
            last_err = e
            msg = str(e).lower()

            # Rate limit / overload: backoff simple
            if ("429" in msg or "rate limit" in msg or "too many requests" in msg) and attempt < retries:
                time.sleep(1.2 * (attempt + 1))
                continue

            # Auth error
            if ("401" in msg or "unauthorized" in msg or "forbidden" in msg):
                raise RuntimeError(
                    "Error de autenticación con Hugging Face (401/403). "
                    "Revisa que HF_TOKEN esté configurado y que tengas acceso al modelo."
                ) from e

            break

    raise RuntimeError(f"No se pudo generar respuesta desde HF Inference API: {last_err}") from last_err


def assign_evidence_per_sentence(
    retriever: HybridRetriever,
    sentences: List[str],
    evidences: List[Evidence],
) -> Tuple[List[SentenceCitation], float]:
    if not sentences or not evidences:
        return [], 0.0

    sent_emb = retriever.model.encode(
        [f"query: {s}" for s in sentences],
        normalize_embeddings=True,
    )
    ev_emb = retriever.model.encode(
        [f"passage: {e.text[:1200]}" for e in evidences],
        normalize_embeddings=True,
    )

    sims = sent_emb @ ev_emb.T

    out: List[SentenceCitation] = []
    per_sent_scores: List[float] = []

    for i, sent in enumerate(sentences):
        j = int(np.argmax(sims[i]))
        score = float(sims[i, j])
        per_sent_scores.append(score)
        e = evidences[j]
        cite = f"{e.doc_id}:p{e.page}"
        out.append(SentenceCitation(sentence=sent, citations=[cite], grounding=score))

    grounding = float(np.mean(per_sent_scores)) if per_sent_scores else 0.0
    grounding = max(0.0, min(1.0, grounding))
    return out, grounding


# -----------------------------
# Main RAG
# -----------------------------
def run_rag(
    question: str,
    retrieval_mode: str,
    llm_name: str,
    top_k: int,
    grounding_threshold: float,
    chat_context: Optional[list] = None,
) -> RAGResult:
    t0 = time.time()
    retriever = get_retriever()

    mode = normalize_retrieval_mode(retrieval_mode)

    # 1) Retrieval
    if mode == "faiss":
        evidences = retriever.faiss_topk(question, top_k=top_k)

    elif mode == "bm25":
        evidences = retriever.bm25_topk(question, top_k=top_k)

    else:
        # hybrid y hybrid+rerank entran acá
        evidences = retriever.hybrid_topk(
            question,
            top_k_faiss=top_k,
            top_k_bm25=top_k,
            final_k=top_k,
        )

    # 2) Filtro anti-boilerplate
    evidences = [
        e for e in evidences
        if ":: sunat ::" not in e.text.lower()
        and "inicio legislación" not in e.text.lower()
    ]

    # 3) Rerank opcional
    rerank_used = False
    if mode == "hybrid+rerank":
        rerank_used = True
        evidences = retriever.rerank(question, evidences, top_n=top_k)

    # 4) Prompt + LLM remoto HF
    prompt = build_prompt(question, evidences)

    if evidences:
        answer = call_hf_llm(prompt, model_name=llm_name, max_new_tokens=320, temperature=0.0)
        if not answer:
            answer = "No pude generar una respuesta en este momento. Intenta nuevamente."
    else:
        answer = "No encontré evidencia suficiente en el corpus para responder con seguridad."

    # 5) Citas por oración + grounding real
    sentences = split_sentences(answer)
    citations_per_sentence, grounding_score = assign_evidence_per_sentence(retriever, sentences, evidences)

    # 6) Hallucination guard
    hallucination_flag = grounding_score < grounding_threshold

    if hallucination_flag and evidences:
        answer = (
            "No encontré sustento suficiente para responder con alta confianza. "
            "Estos son los fragmentos más relevantes hallados:\n\n"
            + "\n\n".join([f"- ({e.doc_id}:p{e.page}) {e.text[:350]}" for e in evidences[: min(5, len(evidences))]])
        )
        sentences = split_sentences(answer)
        citations_per_sentence, grounding_score = assign_evidence_per_sentence(retriever, sentences, evidences)
        hallucination_flag = grounding_score < grounding_threshold

    debug = {
        "latency_ms": int((time.time() - t0) * 1000),
        "retrieval_mode_raw": retrieval_mode,
        "retrieval_mode_norm": mode,
        "rerank_used": rerank_used,
        "llm": llm_name,
        "top_k": top_k,
        "grounding_threshold": grounding_threshold,
        "prompt_chars": len(prompt),
        "n_evidences": len(evidences),
        "n_sentences": len(citations_per_sentence),
    }

    return RAGResult(
        answer=answer,
        grounding_score=grounding_score,
        hallucination_flag=hallucination_flag,
        citations_per_sentence=citations_per_sentence,
        evidences=evidences,
        debug=debug,
    )