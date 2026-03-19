import sys, os
from pathlib import Path
from dotenv import load_dotenv

# Cargar HF_TOKEN desde .env
load_dotenv()

# Agregar raíz al path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.rag_engine import run_rag

# Preguntas de prueba
QUERIES = [
    "¿Qué dice la legislación aduanera sobre la valoración en aduana?",
    "¿Cuáles son los regímenes aduaneros contemplados en la ley?",
    "¿Qué es el despacho aduanero y cuáles son sus requisitos?",
]

print("=" * 60)
print("TEST RAG COMPLETO")
print("=" * 60)

for q in QUERIES:
    print(f"\n📌 PREGUNTA: {q}")
    print("-" * 60)

    result = run_rag(
        question=q,
        retrieval_mode="hybrid+rerank",
        llm_name="meta-llama/Llama-3.2-1B-Instruct",
        top_k=5,
        grounding_threshold=0.25,
    )

    print(f"✅ Grounding score: {result.grounding_score:.3f}")
    print(f"🚨 Hallucination flag: {result.hallucination_flag}")
    print(f"⏱  Latencia: {result.debug['latency_ms']} ms")
    print(f"\n💬 RESPUESTA:\n{result.answer}")
    print(f"\n📚 CITAS POR ORACIÓN:")
    for sc in result.citations_per_sentence:
        print(f"  [{sc.citations[0]}] (grounding={sc.grounding:.2f}) {sc.sentence[:100]}...")
    print("=" * 60)