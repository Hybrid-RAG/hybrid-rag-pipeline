from backend.retrieval import HybridRetriever

r = HybridRetriever(index_dir="data/index")

q = "¿cual es el plazo para numerar una declaración aduanera de mercancías al régimen de importación para el consumo?"
evid = r.hybrid_topk(q, top_k_faiss=6, top_k_bm25=6, final_k=5)

print("QUERY:", q)
for i, e in enumerate(evid, 1):
    print(f"\n[{i}] {e.source} score={e.score:.3f}  doc={e.doc_id}  page={e.page}")
    print("title:", e.title)
    print("text:", e.text[:300], "...")