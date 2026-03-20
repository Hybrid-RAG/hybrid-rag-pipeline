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

## Evaluacion reproducible BLEU / ROUGE / Grounding

El script de evaluacion del proyecto es:

- `scripts/07_evaluate_bleu_rouge.py`

Comando de ejecucion:

```bash
python -m scripts.07_evaluate_bleu_rouge
```

### Prerrequisitos

1. Estar ubicado en la carpeta `rag_pipeline/`
2. Usar el entorno Python correcto del proyecto
3. Tener instaladas las dependencias de `requirements.txt`
4. Tener `HF_TOKEN` disponible en `rag_pipeline/.env`

Ejemplo en Windows usando el `venv` actual del dashboard:

```powershell
cd D:\MAESTRIA\CarpetaCodex\IA_Generativa\rag_dashboard
venv\Scripts\activate
cd ..\rag_pipeline
python -m pip install -r requirements.txt
python -m scripts.07_evaluate_bleu_rouge
```

Si prefieres no activar el entorno:

```powershell
D:\MAESTRIA\CarpetaCodex\IA_Generativa\rag_dashboard\venv\Scripts\python.exe -m pip install -r D:\MAESTRIA\CarpetaCodex\IA_Generativa\rag_pipeline\requirements.txt
cd D:\MAESTRIA\CarpetaCodex\IA_Generativa\rag_pipeline
D:\MAESTRIA\CarpetaCodex\IA_Generativa\rag_dashboard\venv\Scripts\python.exe -m scripts.07_evaluate_bleu_rouge
```

### Archivo `.env`

La evaluacion espera encontrar las credenciales en:

- `rag_pipeline/.env`

Contenido minimo:

```env
HF_TOKEN=tu_token_de_hugging_face
```

Notas:

- Tambien puede funcionar `hf auth login`, pero para este repo se recomienda `rag_pipeline/.env` porque deja el flujo mas reproducible.
- El script usa actualmente el modelo `meta-llama/Llama-3.2-1B-Instruct` para la evaluacion automatica.
- Tu cuenta/token debe tener acceso al modelo y al provider que Hugging Face resuelva para esa inferencia.

### Problemas comunes

1. `ModuleNotFoundError: dotenv` o `ModuleNotFoundError: nltk`
   - estas usando un Python distinto al entorno donde instalaste `requirements.txt`
2. `generation/hf_auth_missing`
   - falta `HF_TOKEN` en `rag_pipeline/.env` o no se cargo correctamente
3. `model_not_supported`
   - el provider actual no soporta el modelo configurado
4. Mensajes `UNEXPECTED` al cargar `intfloat/multilingual-e5-base` o `BAAI/bge-reranker-base`
   - en la corrida actual del proyecto estos mensajes no bloquearon la evaluacion

### Evidencia actual del proyecto

Corrida validada manualmente en este proyecto:

- `BLEU medio = 0.1068`
- `ROUGE-1 medio = 0.3910`
- `ROUGE-2 medio = 0.1868`
- `ROUGE-L medio = 0.3005`
- `Grounding medio = 0.8736`
- `Alucinaciones = 0/9`
