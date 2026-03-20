# Tradeoffs y Evidencia Tecnica

Este documento resume las decisiones tecnicas principales del proyecto RAG y las vincula con evidencia concreta del codigo.

## Mapa de evidencia

| Tema requerido | Decision implementada | Evidencia de codigo | Tradeoff observado |
|---|---|---|---|
| Chunking + overlap | Se usa chunking por caracteres con intento previo de corte por articulos. Configuracion actual: `CHUNK_SIZE = 800`, `CHUNK_OVERLAP = 120`. | `scripts/03_parse_and_chunk.py:15`, `scripts/03_parse_and_chunk.py:16`, `scripts/03_parse_and_chunk.py:41`, `scripts/03_parse_and_chunk.py:218` | Chunks mas grandes preservan contexto legal, pero reducen granularidad. El overlap reduce cortes bruscos entre fragmentos, a costa de duplicar contenido y aumentar el indice. |
| FAISS HNSW | Se usa `IndexHNSWFlat` con metrica de producto interno sobre embeddings normalizados. | `scripts/04_build_indexes.py:98`, `scripts/04_build_indexes.py:99`, `scripts/04_build_indexes.py:100`, `scripts/04_build_indexes.py:127` | HNSW acelera busqueda aproximada sobre miles de chunks. La contrapartida es mas complejidad de configuracion (`M`, `efConstruction`, `efSearch`) y posible perdida marginal frente a busqueda exacta. |
| Hybrid BM25 + FAISS | Se combinan scores lexicales y vectoriales con normalizacion y pesos `0.6 / 0.4`. | `backend/retrieval.py:153`, `backend/retrieval.py:160`, `backend/retrieval.py:176`, `backend/retrieval.py:184`, `backend/rag_engine.py:376` | BM25 ayuda cuando la consulta comparte terminos exactos con la norma; FAISS ayuda con similitud semantica. La fusion mejora cobertura, pero requiere calibrar pesos y deduplicacion. |
| Reranker Cross-Encoder | Se usa `CrossEncoder` opcional para reordenar evidencias despues del retrieval inicial. | `backend/retrieval.py:68`, `backend/retrieval.py:69`, `backend/retrieval.py:191`, `backend/rag_engine.py:220`, `backend/rag_engine.py:400` | El reranker mejora precision en el top final, pero agrega latencia y costo computacional. Por eso se aplica como paso opcional posterior al retrieval base. |
| Citation per sentence | Cada oracion de la respuesta se vincula a una evidencia principal mediante similitud entre embeddings de oracion y pasaje. | `backend/rag_engine.py:315`, `backend/rag_engine.py:333`, `backend/rag_engine.py:347`, `backend/rag_engine.py:438` | Este enfoque mejora trazabilidad explicita de la respuesta. La limitacion es que asigna una cita principal por oracion y puede simplificar casos donde una idea dependa de multiples fragmentos. |
| Grounding + hallucination guard | Se calcula un `grounding_score` promedio por oracion y se marca alucinacion si queda por debajo del umbral. | `backend/rag_engine.py:438`, `backend/rag_engine.py:441`, `backend/rag_engine.py:443`, `backend/rag_engine.py:450`, `backend/rag_engine.py:468` | El guard reduce respuestas con bajo sustento y fuerza una salida mas conservadora. La contrapartida es que un umbral alto puede volver el sistema demasiado restrictivo. |
| BLEU / ROUGE | La evaluacion automatica compara respuesta generada contra referencias manuales con `BLEU`, `ROUGE-1`, `ROUGE-2` y `ROUGE-L`. | `scripts/07_evaluate_bleu_rouge.py:123`, `scripts/07_evaluate_bleu_rouge.py:132`, `scripts/07_evaluate_bleu_rouge.py:173`, `scripts/07_evaluate_bleu_rouge.py:174` | Son utiles para comparar parecido textual con una referencia, pero no capturan completamente correccion juridica ni utilidad practica. Por eso se complementan con `grounding` y bandera de alucinacion. |
| Evaluacion con grounding | La evaluacion tambien registra `grounding_score` y `hallucination_flag` junto a las metricas de texto. | `scripts/07_evaluate_bleu_rouge.py:171`, `scripts/07_evaluate_bleu_rouge.py:181`, `scripts/07_evaluate_bleu_rouge.py:188` | Esto compensa parte de la debilidad de BLEU/ROUGE en QA generativa, porque agrega una senal sobre sustento documental real. |

## Resumen tecnico

- El proyecto prioriza trazabilidad y sustento documental sobre respuestas libres sin control.
- La arquitectura combina recuperacion semantica, recuperacion lexical y reranking para mejorar precision.
- La capa de grounding busca evitar que una respuesta aparente sea aceptada sin evidencia suficiente.
- La evaluacion automatica no se limita a similitud textual; tambien mide si la respuesta se mantiene anclada al corpus.
