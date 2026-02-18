# Hybrid RAG Pipeline

Pipeline reproducible para construir un sistema RAG híbrido:

- Dataset externo (≥100 documentos PDF/Wiki/TXT)
- Parsing por página
- Chunking + overlap
- Embeddings
- Índice híbrido: FAISS (HNSW) + BM25
- Retrieval híbrido + Cross-Encoder Reranker
- Evaluación: ROUGE / BLEU + Grounding mean

## Estructura

- `notebooks/`: notebooks de Colab (pipeline paso a paso)
- `src/`: módulos reutilizables (chunking, embeddings, indexing, retrieval, reranking, metrics)
- `artifacts/`: salida generada (no se sube al repo; se guarda en storage externo)
- `data/`: dataset local (no se sube al repo)
