# Tradeoffs y Soporte Tecnico del Proyecto RAG

Este documento resume las decisiones tecnicas del proyecto y las conecta con evidencia concreta del codigo. El objetivo no es repetir toda la implementacion, sino dejar claro por que se eligio cada componente y que tradeoff introduce.

## 1. Criterio general de diseno

El sistema prioriza tres objetivos:

1. recuperar evidencia juridica relevante desde un corpus real de SUNAT
2. generar respuestas trazables y conservadoras
3. medir no solo parecido textual, sino tambien sustento documental

Por eso la arquitectura no usa un solo mecanismo de retrieval o una sola metrica de evaluacion. Se combinaron retrieval hibrido, reranking, citas por oracion, grounding y una evaluacion automatica con BLEU / ROUGE.

## 2. Mapa rapido de requisitos obligatorios

| Requisito                | Implementacion actual                                             | Evidencia                                                                                                       |
| ------------------------ | ----------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| Dataset externo real     | Crawl, filtro y descarga de fuentes SUNAT                         | `scripts/01_crawl_sunat.py`, `scripts/01b_filter_manifest.py`, `scripts/02_download_docs.py`                    |
| Chunking con overlap     | Chunking por caracteres con intento previo por articulos          | `scripts/03_parse_and_chunk.py`                                                                                 |
| FAISS HNSW               | Indice `IndexHNSWFlat` sobre embeddings normalizados              | `scripts/04_build_indexes.py`                                                                                   |
| Pipeline completo        | De documentos a respuesta RAG con retrieval, LLM y guardrails     | `scripts/03_parse_and_chunk.py`, `scripts/04_build_indexes.py`, `backend/retrieval.py`, `backend/rag_engine.py` |
| Reranker cross-encoder   | Cross-encoder opcional sobre evidencias recuperadas               | `backend/retrieval.py`, `backend/rag_engine.py`                                                                 |
| Citation per sentence    | Asignacion de evidencia principal por oracion                     | `backend/rag_engine.py`                                                                                         |
| Grounding + guard        | Grounding score promedio y bandera de alucinacion por umbral      | `backend/rag_engine.py`                                                                                         |
| BLEU / ROUGE + grounding | Script de evaluacion automatica con metricas de texto y grounding | `scripts/07_evaluate_bleu_rouge.py`                                                                             |
| Hybrid BM25 + FAISS      | Fusion de retrieval lexical y semantico                           | `backend/retrieval.py`                                                                                          |

## 3. Decisiones principales y tradeoffs

### 3.1 Chunking con overlap

**Decision implementada**

- configuracion actual:
  - `CHUNK_SIZE = 800`
  - `CHUNK_OVERLAP = 120`
- antes de aplicar ventanas por caracteres, el parser intenta dividir por articulos legales cuando detecta patrones como `Articulo 12` o `Art. 15`

**Evidencia**

- `scripts/03_parse_and_chunk.py:12`
- `scripts/03_parse_and_chunk.py:13`
- `scripts/03_parse_and_chunk.py:38`
- `scripts/03_parse_and_chunk.py:44`
- `scripts/03_parse_and_chunk.py:187`

**Justificacion**

- En normativa legal, un chunk demasiado pequeno puede romper la unidad semantica de una disposicion.
- Un chunk demasiado grande aumenta ruido, reduce precision de retrieval y hace mas costoso el reranking.
- El overlap amortigua cortes bruscos entre ventanas y preserva continuidad entre fragmentos cercanos.

**Tradeoff**

- Mas overlap mejora continuidad, pero duplica contenido y agranda el indice.
- Mas tamano preserva contexto, pero empeora granularidad.

### 3.2 FAISS HNSW

**Decision implementada**

- uso de `faiss.IndexHNSWFlat`
- metrica: producto interno
- embeddings normalizados para aproximar similitud coseno
- parametros:
  - `M = 32`
  - `efConstruction = 200`
  - `efSearch = 64`

**Evidencia**

- `scripts/04_build_indexes.py:18`
- `scripts/04_build_indexes.py:22`
- `scripts/04_build_indexes.py:86`
- `scripts/04_build_indexes.py:88`
- `scripts/04_build_indexes.py:89`
- `scripts/04_build_indexes.py:90`

**Justificacion**

- HNSW permite retrieval aproximado rapido sobre miles de chunks.
- Es mas adecuado para demo interactiva que una busqueda exacta mas lenta.

**Tradeoff**

- Gana velocidad, pero requiere calibracion de parametros y puede perder algo de exactitud frente a un indice exacto.
- La calidad depende del equilibrio entre `M`, `efConstruction` y `efSearch`.

### 3.3 Retrieval hibrido BM25 + FAISS

**Decision implementada**

- FAISS aporta similitud semantica
- BM25 aporta coincidencia lexical
- ambos scores se normalizan y se fusionan con pesos:
  - `w_faiss = 0.6`
  - `w_bm25 = 0.4`

**Evidencia**

- `backend/retrieval.py:131`
- `backend/retrieval.py:137`
- `backend/retrieval.py:153`
- `backend/retrieval.py:167`
- `backend/retrieval.py:176`
- `backend/rag_engine.py:374`

**Justificacion**

- En consultas legales, algunos casos dependen de terminos exactos y otros de parafrasis semantica.
- La fusion reduce el riesgo de depender solo del matching textual o solo del embedding.

**Tradeoff**

- Mejora cobertura, pero introduce mas complejidad:
  - normalizacion de scores
  - pesos de fusion
  - deduplicacion de chunks

### 3.4 Reranker cross-encoder

**Decision implementada**

- se usa `CrossEncoder` opcional con `BAAI/bge-reranker-base`
- el reranker opera sobre las evidencias ya recuperadas
- se expone como modo visible de demo: `Hybrid + Rerank (CrossEncoder)`

**Evidencia**

- `backend/retrieval.py:57`
- `backend/retrieval.py:62`
- `backend/retrieval.py:187`
- `backend/rag_engine.py:414`

**Justificacion**

- El retrieval inicial busca recall.
- El reranker mejora precision en el top final antes de construir el prompt.

**Tradeoff**

- Mejora relevancia, pero agrega latencia y costo computacional.
- Por eso se aplica despues del retrieval base y no sobre todo el corpus.

### 3.5 Citation per sentence

**Decision implementada**

- la respuesta se divide en oraciones
- cada oracion se compara contra las evidencias recuperadas
- a cada oracion se le asigna la evidencia mas similar como cita principal

**Evidencia**

- `backend/rag_engine.py:170`
- `backend/rag_engine.py:323`
- `backend/rag_engine.py:338`
- `backend/rag_engine.py:343`

**Justificacion**

- Este mecanismo mejora trazabilidad: cada parte de la respuesta queda asociada a sustento documental.

**Tradeoff**

- Es interpretable y simple de mostrar en UI, pero simplifica casos donde una afirmacion depende de varias evidencias.
- No representa relaciones mas complejas entre multiples fragmentos.

### 3.6 Grounding score y hallucination guard

**Decision implementada**

- se calcula similitud embedding entre oraciones de respuesta y evidencias
- se promedia para obtener `grounding_score`
- si el score queda por debajo del umbral, se activa `hallucination_flag`
- cuando el guard se activa, el sistema reemplaza la respuesta libre por una salida mas conservadora basada en fragmentos recuperados

**Evidencia**

- `backend/rag_engine.py:343`
- `backend/rag_engine.py:347`
- `backend/rag_engine.py:430`
- `backend/rag_engine.py:433`
- `backend/rag_engine.py:435`

**Justificacion**

- En un dominio legal, es preferible una salida conservadora a una respuesta inventada pero fluida.

**Tradeoff**

- Un umbral muy bajo deja pasar respuestas con poco sustento.
- Un umbral muy alto puede volver el sistema demasiado restrictivo.

### 3.7 BLEU / ROUGE como evaluacion automatica

**Decision implementada**

- el script de evaluacion compara respuestas generadas contra referencias manuales
- calcula:
  - `BLEU`
  - `ROUGE-1`
  - `ROUGE-2`
  - `ROUGE-L`
- ademas registra grounding y bandera de alucinacion

**Evidencia**

- `scripts/07_evaluate_bleu_rouge.py:120`
- `scripts/07_evaluate_bleu_rouge.py:129`
- `scripts/07_evaluate_bleu_rouge.py:170`
- `scripts/07_evaluate_bleu_rouge.py:180`
- `scripts/07_evaluate_bleu_rouge.py:187`

**Justificacion**

- BLEU / ROUGE permiten una comparacion automatica reproducible contra respuestas de referencia.

**Tradeoff**

- Son utiles como senal automatica, pero no capturan completamente correccion juridica ni utilidad real.
- Por eso se complementan con grounding y `hallucination_flag`.

## 4. Resultado actual de evaluacion

Corrida validada manualmente en este proyecto:

- `BLEU medio = 0.0790`
- `ROUGE-1 medio = 0.3630`
- `ROUGE-2 medio = 0.1868`
- `ROUGE-L medio = 0.3005`
- `Grounding medio = 0.8640`
- `Alucinaciones = 0/9`

## 5. Limitaciones honestas del sistema

1. El reranker mejora precision, pero aumenta latencia.
2. Citation per sentence usa una evidencia principal por oracion, no razonamiento multi-cita.
3. BLEU / ROUGE no sustituyen evaluacion experta del contenido legal.
4. La disponibilidad de LLMs depende del provider de inferencia configurado en Hugging Face.

## 6. Conclusion tecnica

La arquitectura actual no busca maximizar solo fluidez de respuesta. Busca equilibrio entre:

- recall y precision en retrieval
- trazabilidad por cita
- grounding cuantificable
- control de alucinacion
- evaluacion reproducible

Ese equilibrio es coherente con el objetivo del proyecto: debido a que responde sobre legislacion aduanera con sustento documental real y con mecanismos visibles de control de calidad.
