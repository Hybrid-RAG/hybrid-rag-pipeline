"""
Evaluación automática del sistema RAG usando BLEU, ROUGE y Grounding.
Ejecutar: python -m scripts.07_evaluate_bleu_rouge
"""

import sys
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import nltk
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

from backend.rag_engine import run_rag

# =============================================
# DATASET DE EVALUACIÓN
# Preguntas + respuestas de referencia escritas
# manualmente basadas en la legislación SUNAT
# =============================================
EVAL_SET = [
    {
        "question": "¿Qué es la valoración en aduana?",
        "reference": (
            "La valoración en aduana es el proceso para determinar el valor de las mercancías "
            "importadas con fines de aplicación de derechos arancelarios. Se basa principalmente "
            "en el valor de transacción, es decir, el precio realmente pagado o por pagar por "
            "las mercancías cuando se venden para su exportación. El sistema debe ser equitativo, "
            "uniforme y neutro, excluyendo valores arbitrarios o ficticios."
        ),
    },
    {
        "question": "¿Qué es el despacho aduanero?",
        "reference": (
            "El despacho aduanero es el conjunto de actos y formalidades relativos a la entrada "
            "y salida de mercancías al territorio aduanero que deben realizarse ante la aduana. "
            "Implica la presentación de documentos, pago de tributos y verificación de la carga. "
            "El importador o su agente de aduana debe cumplir con los requisitos establecidos "
            "por SUNAT para la regularización del despacho."
        ),
    },
    {
        "question": "¿Cuáles son las infracciones aduaneras más comunes?",
        "reference": (
            "Las infracciones aduaneras incluyen la declaración incorrecta de mercancías, "
            "el contrabando, la subvaluación y la clasificación arancelaria incorrecta. "
            "Las sanciones pueden ser multas, comiso de mercancías o inhabilitación. "
            "SUNAT aplica estas sanciones de acuerdo con la Ley General de Aduanas."
        ),
    },
    {
        "question": "¿Qué documentos se requieren para importar mercancías?",
        "reference": (
            "Para importar mercancías se requiere la declaración aduanera de mercancías, "
            "factura comercial, documento de transporte, lista de empaque y certificado de origen "
            "cuando corresponda. El agente de aduana transmite electrónicamente estos documentos "
            "al sistema SUNAT para el canal de control asignado."
        ),
    },
    {
        "question": "¿Qué es el régimen de importación para el consumo?",
        "reference": (
            "El régimen de importación para el consumo permite el ingreso de mercancías al "
            "territorio aduanero para su uso o consumo definitivo luego del pago de los "
            "derechos arancelarios y demás tributos aplicables. La mercancía queda en libre "
            "disposición una vez numerada y cancelada la deuda tributaria aduanera."
        ),
    },
    {
        "question": "¿Cuándo se asigna el canal de control en el régimen de importación para el consumo?",
        "reference": (
            "La asignación del canal de control determina el tipo de control al que se sujetan "
            "las mercancías. En el despacho anticipado se asigna con la numeración de la declaración; "
            "para la vía terrestre, adicionalmente debe contar con el registro de la llegada del "
            "medio de transporte. En el despacho diferido y en el despacho urgente se asigna cuando "
            "la declaración cuente con la deuda tributaria aduanera y recargos cancelados o "
            "garantizados. Para la vía terrestre en despacho urgente, también debe contar con "
            "el registro de la llegada del medio de transporte."
        ),
    },
{
        "question": "¿Qué es el drawback y cuáles son sus requisitos?",
        "reference": (
            "El drawback es un régimen aduanero que permite la restitución total o parcial de "
            "los derechos arancelarios pagados por la importación de insumos utilizados en la "
            "producción de bienes exportados. Para acogerse al drawback se requiere presentar "
            "la solicitud dentro del plazo establecido, acreditar la exportación definitiva "
            "y demostrar que los insumos importados fueron utilizados en la producción."
        ),
    },
    {
        "question": "¿Qué es la admisión temporal para perfeccionamiento activo?",
        "reference": (
            "La admisión temporal para perfeccionamiento activo es un régimen que permite el "
            "ingreso de mercancías extranjeras al territorio aduanero con suspensión del pago "
            "de derechos arancelarios y demás impuestos, para ser sometidas a operaciones de "
            "perfeccionamiento y posterior reexportación en forma de productos compensadores. "
            "El plazo máximo es de veinticuatro meses."
        ),
    },
    {
        "question": "¿Cuáles son los canales de control en el despacho aduanero?",
        "reference": (
            "Los canales de control en el despacho aduanero son tres: canal verde, canal naranja "
            "y canal rojo. El canal verde otorga el levante automático sin revisión documentaria "
            "ni reconocimiento físico. El canal naranja implica revisión documentaria sin "
            "reconocimiento físico. El canal rojo implica reconocimiento físico de la mercancía "
            "y revisión documentaria."
        ),
    },
]


# =============================================
# FUNCIONES DE EVALUACIÓN
# =============================================

def compute_bleu(reference: str, hypothesis: str) -> float:
    """BLEU a nivel de oraciones con suavizado."""
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()
    smoothie = SmoothingFunction().method4
    score = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothie)
    return round(float(score), 4)


def compute_rouge(reference: str, hypothesis: str) -> dict:
    """ROUGE-1, ROUGE-2 y ROUGE-L."""
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
    scores = scorer.score(reference, hypothesis)
    return {
        "rouge1_f": round(scores["rouge1"].fmeasure, 4),
        "rouge2_f": round(scores["rouge2"].fmeasure, 4),
        "rougeL_f": round(scores["rougeL"].fmeasure, 4),
    }


# =============================================
# MAIN
# =============================================

def main():
    nltk.download("punkt", quiet=True)

    print("=" * 65)
    print("EVALUACIÓN AUTOMÁTICA: BLEU + ROUGE + GROUNDING")
    print("=" * 65)

    results = []

    for i, item in enumerate(EVAL_SET, 1):
        q = item["question"]
        ref = item["reference"]

        print(f"\n[{i}/{len(EVAL_SET)}] {q}")

        rag_result = run_rag(
            question=q,
            retrieval_mode="hybrid",
            llm_name="meta-llama/Llama-3.2-1B-Instruct",
            top_k=5,
            grounding_threshold=0.25,
        )

        hyp = rag_result.answer
        grounding = rag_result.grounding_score

        bleu = compute_bleu(ref, hyp)
        rouge = compute_rouge(ref, hyp)

        print(f"  BLEU:      {bleu:.4f}")
        print(f"  ROUGE-1:   {rouge['rouge1_f']:.4f}")
        print(f"  ROUGE-2:   {rouge['rouge2_f']:.4f}")
        print(f"  ROUGE-L:   {rouge['rougeL_f']:.4f}")
        print(f"  Grounding: {grounding:.4f}")
        print(f"  Hallucin.: {'⚠ SÍ' if rag_result.hallucination_flag else '✅ NO'}")

        results.append({
            "question": q,
            "bleu": bleu,
            **rouge,
            "grounding": grounding,
            "hallucination": rag_result.hallucination_flag,
        })

    # ---- Promedios finales ----
    print("\n" + "=" * 65)
    print("RESUMEN FINAL (promedios sobre 9 preguntas)")
    print("=" * 65)

    avg_bleu      = np.mean([r["bleu"] for r in results])
    avg_rouge1    = np.mean([r["rouge1_f"] for r in results])
    avg_rouge2    = np.mean([r["rouge2_f"] for r in results])
    avg_rougeL    = np.mean([r["rougeL_f"] for r in results])
    avg_grounding = np.mean([r["grounding"] for r in results])
    n_hallucin    = sum(1 for r in results if r["hallucination"])

    print(f"  BLEU medio:       {avg_bleu:.4f}")
    print(f"  ROUGE-1 medio:    {avg_rouge1:.4f}")
    print(f"  ROUGE-2 medio:    {avg_rouge2:.4f}")
    print(f"  ROUGE-L medio:    {avg_rougeL:.4f}")
    print(f"  Grounding medio:  {avg_grounding:.4f}")
    print(f"  Alucinaciones:    {n_hallucin}/{len(results)}")
    print("=" * 65)


if __name__ == "__main__":
    main()