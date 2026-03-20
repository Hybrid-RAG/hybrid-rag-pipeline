# Hybrid RAG Pipeline

Pipeline reproducible para construir un sistema RAG hibrido:

- Dataset externo (>=100 documentos PDF/Wiki/TXT)
- Parsing por pagina
- Chunking + overlap
- Embeddings
- Indice hibrido: FAISS (HNSW) + BM25
- Retrieval hibrido + Cross-Encoder Reranker
- Evaluacion: ROUGE / BLEU + grounding

## Estructura actual

- `backend/`: modulo reutilizable consumido por el dashboard
- `scripts/`: pipeline paso a paso (crawl, parse, index, pruebas, evaluacion)
- `data/`: dataset local, chunks e indices
- `artifacts/`: salidas auxiliares

## Instalacion local editable

Este repo incluye `setup.py`, por lo que puede instalarse en modo editable.

Desde `rag_dashboard/`:

```bash
pip install --upgrade pip setuptools wheel
pip install -e ../rag-pipeline
```

Si tu carpeta local se llama con guion bajo:

```bash
pip install -e ../rag_pipeline
```

## Uso del modulo desde Python

```python
from backend.rag_engine import run_rag
```

Nota: este import requiere que `rag_pipeline` este instalado en el entorno activo (editable o normal).
