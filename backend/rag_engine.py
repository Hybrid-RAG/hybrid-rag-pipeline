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


@dataclass
class RAGRuntimeError(RuntimeError):
    stage: str
    code: str
    message: str
    user_hint: str
    retryable: bool = False
    raw_type: str = "Exception"

    def __post_init__(self):
        super().__init__(self.__str__())

    def __str__(self) -> str:
        return (
            f"[{self.stage}/{self.code}] {self.message} "
            f"Hint: {self.user_hint}"
        )

    def to_debug_dict(self) -> Dict:
        return {
            "stage": self.stage,
            "code": self.code,
            "message": self.message,
            "user_hint": self.user_hint,
            "retryable": self.retryable,
            "raw_type": self.raw_type,
        }


def _runtime_error(
    *,
    stage: str,
    code: str,
    message: str,
    user_hint: str,
    exc: Exception,
    retryable: bool = False,
) -> RAGRuntimeError:
    return RAGRuntimeError(
        stage=stage,
        code=code,
        message=message,
        user_hint=user_hint,
        retryable=retryable,
        raw_type=type(exc).__name__,
    )


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
        index_dir = _default_index_dir()
        try:
            _RETRIEVER = HybridRetriever(
                index_dir=index_dir,
                rerank_model="BAAI/bge-reranker-base",
                rerank_device="cpu"
            )
        except FileNotFoundError as e:
            missing_name = Path(getattr(e, "filename", "") or "").name.lower()
            code = "retriever_artifact_missing"
            message = f"No se encontraron artefactos de retrieval en '{index_dir}'."

            if missing_name == "meta.parquet":
                code = "retriever_meta_missing"
                message = f"Falta meta.parquet en '{index_dir}'."
            elif missing_name == "faiss_hnsw.index":
                code = "retriever_index_missing"
                message = f"Falta faiss_hnsw.index en '{index_dir}'."
            elif missing_name == "bm25.pkl":
                code = "retriever_bm25_missing"
                message = f"Falta bm25.pkl en '{index_dir}'."

            raise _runtime_error(
                stage="retriever_init",
                code=code,
                message=message,
                user_hint="Revisa RAG_INDEX_DIR o reconstruye los artefactos del indice.",
                exc=e,
            ) from e
        except RuntimeError as e:
            if "dimension mismatch" in str(e).lower():
                raise _runtime_error(
                    stage="retriever_init",
                    code="retriever_dim_mismatch",
                    message="La dimension de embeddings no coincide con el indice FAISS cargado.",
                    user_hint="Alinea embed_model con index_config.pkl o reconstruye data/index.",
                    exc=e,
                ) from e
            raise _runtime_error(
                stage="retriever_init",
                code="retriever_init_failed",
                message=f"No se pudo inicializar el retriever desde '{index_dir}'.",
                user_hint="Revisa artefactos locales, modelos de retrieval y configuracion del entorno.",
                exc=e,
            ) from e
        except Exception as e:
            raise _runtime_error(
                stage="retriever_init",
                code="retriever_init_failed",
                message=f"No se pudo inicializar el retriever desde '{index_dir}'.",
                user_hint="Revisa artefactos locales, modelos de retrieval y configuracion del entorno.",
                exc=e,
            ) from e
    return _RETRIEVER


# -----------------------------
# Helpers
# -----------------------------
_SENT_SPLIT = re.compile(r"(?<=[\.\!\?])\s+")


def split_sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text.strip()) if s.strip()]


def build_prompt(question: str, evidences: List[Evidence], chat_context: Optional[list] = None) -> str:
    ctx = []
    for i, e in enumerate(evidences, 1):
        cite = f"{e.doc_id}:p{e.page}"
        ctx.append(f"[{i}] ({cite}) {e.text[:900]}")
    context_block = "\n\n".join(ctx)

    # Historial de conversación
    history_block = ""
    if chat_context:
        history_lines = []
        for msg in chat_context[-6:]:  # últimos 3 turnos
            role = "Usuario" if msg["role"] == "user" else "Asistente"
            history_lines.append(f"{role}: {msg['content'][:300]}")
        history_block = "\n\nHISTORIAL PREVIO:\n" + "\n".join(history_lines)

    return f"""Eres un asistente experto en legislación aduanera peruana (SUNAT).
Reglas:
- Responde SOLO usando la evidencia proporcionada.
- Si el historial previo es relevante, úsalo para dar contexto.
- Si falta sustento, dilo explícitamente.
- Escribe en español técnico, claro.
- No inventes artículos, números, ni resoluciones.
- Estructura tu respuesta en oraciones cortas.
{history_block}

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
    token = os.getenv("HF_TOKEN")
    provider = os.getenv("HF_PROVIDER") or os.getenv("HF_INFERENCE_PROVIDER")
    client = InferenceClient(api_key=token, provider=provider)

    last_err: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "Eres un asistente experto en legislacion aduanera peruana (SUNAT)."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_new_tokens,
                temperature=max(temperature, 0.01),
            )
            return (resp.choices[0].message.content or "").strip()

        except Exception as e:
            last_err = e
            msg = str(e).lower()
            if ("429" in msg or "rate limit" in msg) and attempt < retries:
                time.sleep(1.2 * (attempt + 1))
                continue
            if "429" in msg or "rate limit" in msg:
                raise _runtime_error(
                    stage="generation",
                    code="hf_rate_limit",
                    message="Hugging Face rechazo la solicitud por limite de tasa.",
                    user_hint="Espera unos segundos e intenta de nuevo.",
                    exc=e,
                    retryable=True,
                ) from e
            if "must provide an api_key" in msg or "hf auth login" in msg:
                raise _runtime_error(
                    stage="generation",
                    code="hf_auth_missing",
                    message="No se encontro autenticacion de Hugging Face.",
                    user_hint="Define HF_TOKEN o inicia sesion con `hf auth login`.",
                    exc=e,
                ) from e
            if "401" in msg or "unauthorized" in msg:
                raise _runtime_error(
                    stage="generation",
                    code="hf_auth_invalid",
                    message="Hugging Face rechazo la autenticacion actual.",
                    user_hint="Revisa HF_TOKEN y los permisos del modelo.",
                    exc=e,
                ) from e
            if "model_not_supported" in msg or "not supported by any provider" in msg:
                provider_name = provider or "auto"
                raise _runtime_error(
                    stage="generation",
                    code="hf_model_not_supported",
                    message=(
                        f"El modelo '{model_name}' no esta disponible para el provider actual "
                        f"de Hugging Face ('{provider_name}')."
                    ),
                    user_hint="Prueba otro LLM en la UI o configura HF_PROVIDER.",
                    exc=e,
                ) from e
            break

    if last_err is None:
        last_err = RuntimeError("Unknown HF generation failure")

    raise _runtime_error(
        stage="generation",
        code="hf_generation_failed",
        message="No se pudo generar respuesta con Hugging Face.",
        user_hint="Revisa provider, modelo y conectividad antes de reintentar.",
        exc=last_err,
    ) from last_err
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
    try:
        if mode == "faiss":
            evidences = retriever.faiss_topk(question, top_k=top_k)

        elif mode == "bm25":
            evidences = retriever.bm25_topk(question, top_k=top_k)

        else:
            # hybrid y hybrid+rerank entran aca
            evidences = retriever.hybrid_topk(
                question,
                top_k_faiss=top_k,
                top_k_bm25=top_k,
                final_k=top_k,
            )
    except RAGRuntimeError:
        raise
    except Exception as e:
        code_by_mode = {
            "faiss": "retrieval_faiss_failed",
            "bm25": "retrieval_bm25_failed",
            "hybrid": "retrieval_hybrid_failed",
            "hybrid+rerank": "retrieval_hybrid_failed",
        }
        raise _runtime_error(
            stage="retrieval",
            code=code_by_mode.get(mode, "retrieval_failed"),
            message=f"Fallo el retrieval en modo '{mode}'.",
            user_hint="Revisa artefactos del indice y el modelo de embeddings antes de reintentar.",
            exc=e,
        ) from e

    # 2) Filtro anti-boilerplate
    BOILERPLATE = [
        ":: sunat ::",
        "inicio legislación",
        "tamaño de texto",
        "esta página usa marcos",
        "su explorador no los admite",
        "javascript",
        "stylesheet",
        "bannersup",
        "document.write",
        "window.print",
        "skip to content",
        "ir al contenido",
    ]
    evidences = [
        e for e in evidences
        if not any(b in e.text.lower() for b in BOILERPLATE)
        and len(e.text.strip()) > 80
    ]

    # 3) Rerank opcional
    rerank_used = False
    if mode == "hybrid+rerank":
        rerank_used = True
        evidences = retriever.rerank(question, evidences, top_n=top_k)

    # 4) Prompt + LLM remoto HF
    prompt = build_prompt(question, evidences, chat_context=chat_context)

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
