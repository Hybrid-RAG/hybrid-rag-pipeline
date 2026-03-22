# Hybrid RAG Pipeline

Pipeline modular para construir, ejecutar y evaluar un sistema Hybrid RAG sobre legislacion aduanera SUNAT.

## Que hace este repo

Este repositorio cubre el backend completo del proyecto:

- ingesta y descarga del corpus legal
- parsing por pagina
- chunking con overlap
- embeddings
- indexacion hibrida con `FAISS HNSW + BM25`
- retrieval hibrido
- reranking con cross-encoder
- generacion con LLM
- citation per sentence
- grounding score y guard de alucinacion
- evaluacion automatica con `BLEU / ROUGE + grounding`

## Requisitos que cubre

Este repo contiene la implementacion tecnica de los requisitos obligatorios del proyecto:

- dataset externo real
- chunking + overlap
- FAISS HNSW
- BM25 + FAISS
- reranker
- citation per sentence
- grounding + hallucination guard
- evaluacion BLEU / ROUGE + grounding

## Estructura

- `backend/`
  - motor reutilizable consumido por el dashboard
- `scripts/`
  - pipeline paso a paso
- `data/`
  - corpus local, chunks, metadatos e indices
- `artifacts/`
  - salidas auxiliares
- `README.md`
  - guia operativa del backend
- `TRADEOFFS.md`
  - nota tecnica de decisiones y tradeoffs

## Flujo del pipeline

Orden principal de construccion:

1. `scripts/01_crawl_sunat.py`
2. `scripts/01b_filter_manifest.py`
3. `scripts/02_download_docs.py`
4. `scripts/03_parse_and_chunk.py`
5. `scripts/04_build_indexes.py`

Validacion y pruebas:

6. `scripts/05_test_retrieval.py`
7. `scripts/06_test_rag.py`
8. `scripts/07_evaluate_bleu_rouge.py`

## Instalacion

### Opcion 1: entorno local del pipeline

Desde `rag_pipeline/`:

```bash
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

### Opcion 2: instalacion editable para el dashboard

Este repo incluye `setup.py`, por lo que puede instalarse en modo editable desde `rag_dashboard/`.

```bash
pip install --upgrade pip setuptools wheel
pip install -e ../rag_pipeline
```

Si la carpeta local usa guion:

```bash
pip install -e ../rag-pipeline
```

## Configuracion

Se recomienda crear `rag_pipeline/.env` con al menos:

```env
HF_TOKEN=tu_token_de_hugging_face
```

Notas:

- `HF_TOKEN` es el token normal de Hugging Face.
- El backend resuelve provider por modelo para:
  - `Qwen/Qwen2.5-7B-Instruct`
  - `meta-llama/Llama-3.2-1B-Instruct`
  - `mistralai/Mistral-7B-Instruct-v0.2`
- Si defines `HF_PROVIDER` o `HF_INFERENCE_PROVIDER`, ese valor global tiene prioridad.

## Uso desde Python

El punto de entrada principal del backend es:

```python
from backend.rag_engine import run_rag
```

Ejemplo:

```python
from backend.rag_engine import run_rag

result = run_rag(
    question="Que dice la normativa sobre valoracion en aduana?",
    retrieval_mode="hybrid",
    llm_name="Qwen/Qwen2.5-7B-Instruct",
    top_k=6,
    grounding_threshold=0.7,
)
```

## Evaluacion reproducible

Script principal:

- `scripts/07_evaluate_bleu_rouge.py`

Comando:

```bash
python -m scripts.07_evaluate_bleu_rouge
```

### Prerrequisitos

1. Estar ubicado en `rag_pipeline/`
2. Usar el Python correcto del proyecto
3. Tener `requirements.txt` instalado
4. Tener `HF_TOKEN` disponible en `rag_pipeline/.env`

Ejemplo en Windows:

```powershell
cd D:\MAESTRIA\CarpetaCodex\IA_Generativa\rag_pipeline
venv\Scripts\activate
python -m pip install -r requirements.txt
python -m scripts.07_evaluate_bleu_rouge
```

### Evidencia validada

Corrida validada manualmente en este proyecto:

- `BLEU medio = 0.1068`
- `ROUGE-1 medio = 0.3910`
- `ROUGE-2 medio = 0.1868`
- `ROUGE-L medio = 0.3005`
- `Grounding medio = 0.8736`
- `Alucinaciones = 0/9`

## Problemas comunes

1. `ModuleNotFoundError: dotenv` o `ModuleNotFoundError: nltk`
   - estas usando un Python distinto al entorno donde instalaste dependencias
2. `generation/hf_auth_missing`
   - falta `HF_TOKEN` en `rag_pipeline/.env`
3. `generation/hf_model_not_supported`
   - el provider no soporta el modelo configurado
4. Mensajes `UNEXPECTED` al cargar `intfloat/multilingual-e5-base` o `BAAI/bge-reranker-base`
   - en este proyecto no bloquearon la ejecucion validada

## Archivos recomendados para entrega

- `TRADEOFFS.md`
- `../docs_ai/architecture/project_architecture_technical.svg`
- `../docs_ai/architecture/project_architecture_executive.svg`

## Alcance respecto al dashboard

Este repo implementa el backend y la evaluacion offline.

La experiencia interactiva principal vive en:

- `../rag_dashboard/`

Ese dashboard consume localmente este backend sin API intermedia.
