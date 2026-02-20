# scripts/01b_filter_manifest.py
import os, csv, re
from urllib.parse import urlparse

IN_MANIFEST = "data/manifest.csv"
OUT_MANIFEST = "data/manifest_filtered.csv"

TARGET_N = 250          # deja 200–300 para asegurar >=100 descargados
MIN_PDF = 120           # mínimo de PDFs deseados (aprox). Si no alcanza, completa con HTML.

# 1) Palabras clave “aduaneras” (en URL y/o título/categoría)
KEYWORDS = [
    "aduan", "aduanera", "aduanas",
    "proced", "procedimiento",
    "resol", "resolucion",
    "valoracion", "consulta",
    "import", "export",
    "arancel", "clasif", "partida",
    "regimen", "despacho",
    "infraccion", "sancion",
    "ley", "decreto", "suprema",
    "superintendencia",
]

# 2) Rutas preferidas (SUNAT suele colgar lo aduanero aquí)
PATH_HINTS = [
    "/legislacion/aduanera/",
    "/legislacion/",
    "/orientacionaduanera/",
]

# 3) Extensiones descartadas (assets)
BAD_EXT = (".jpg",".jpeg",".png",".gif",".css",".js",".zip",".rar",".mp3",".mp4",".svg",".ico")

def norm(s: str) -> str:
    return (s or "").strip().lower()

def contains_any(text: str, words) -> bool:
    t = norm(text)
    return any(w in t for w in words)

def is_good(row) -> bool:
    url = norm(row.get("source_url",""))
    title = norm(row.get("title",""))
    category = norm(row.get("category",""))
    ft = norm(row.get("file_type",""))

    if not url or url.startswith("mailto:") or url.startswith("javascript:"):
        return False
    if "#" in url:
        return False
    if url.split("?")[0].endswith(BAD_EXT):
        return False

    p = urlparse(url)

    # dominio SUNAT
    if not p.netloc.endswith("sunat.gob.pe"):
        return False

    # Debe parecer “aduanero” por:
    #  - ruta sugerente O
    #  - keywords en url/title/category
    path_ok = any(h in p.path.lower() for h in PATH_HINTS)
    keyword_ok = (
        contains_any(url, KEYWORDS) or
        contains_any(title, KEYWORDS) or
        contains_any(category, KEYWORDS)
    )

    if not (path_ok or keyword_ok):
        return False

    # Si es unknown, lo dejamos pasar solo si la ruta/keywords son fuertes
    if ft == "unknown" and not (path_ok and keyword_ok):
        return False

    return True

def score(row) -> int:
    url = norm(row.get("source_url",""))
    title = norm(row.get("title",""))
    category = norm(row.get("category",""))
    ft = norm(row.get("file_type",""))

    s = 0

    # PDF tiene más valor (citas/páginas), pero NO exclusivo
    if ft == "pdf":
        s += 80
    elif ft == "html":
        s += 35
    else:
        s += 10

    # Si está en carpeta aduanera, sube bastante
    path = urlparse(url).path.lower()
    if "/legislacion/aduanera/" in path:
        s += 40
    elif "/legislacion/" in path:
        s += 20

    # keywords fuertes
    strong = ["aduan", "aduanera", "proced", "resol", "valoracion", "arancel", "despacho", "infraccion", "sancion"]
    for k in strong:
        if k in url: s += 6
        if k in title: s += 4
        if k in category: s += 2

    # penaliza URLs demasiado largas (a veces son navegacionales)
    if len(url) > 220:
        s -= 10

    return s

def main():
    with open(IN_MANIFEST, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print("Manifest vacío.")
        return

    # Filtra
    good = [r for r in rows if is_good(r)]

    # Dedup por URL
    uniq = {}
    for r in good:
        uniq[r["source_url"]] = r
    good = list(uniq.values())

    # Ordena por score
    good.sort(key=score, reverse=True)

    # Estrategia: intentar MIN_PDF primero, luego completar con HTML
    pdfs = [r for r in good if norm(r.get("file_type")) == "pdf"]
    htmls = [r for r in good if norm(r.get("file_type")) == "html"]
    others = [r for r in good if norm(r.get("file_type")) not in ("pdf","html")]

    selected = []
    selected.extend(pdfs[:MIN_PDF])

    # completa hasta TARGET_N con HTML y luego otros
    remaining = TARGET_N - len(selected)
    if remaining > 0:
        selected.extend(htmls[:remaining])

    remaining = TARGET_N - len(selected)
    if remaining > 0:
        selected.extend(others[:remaining])

    # si aún no llegamos, completar con más PDFs/HTML si hay
    if len(selected) < TARGET_N:
        pool = [r for r in good if r not in selected]
        selected.extend(pool[: (TARGET_N - len(selected))])

    # Reset status “discovered” para re-descarga limpia si quieres
    for r in selected:
        if not r.get("status"):
            r["status"] = "discovered"

    os.makedirs("data", exist_ok=True)
    fieldnames = ["doc_id","title","category","source_url","file_type","file_path","status","pages","error"]
    with open(OUT_MANIFEST, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(selected)

    n_pdf = sum(1 for r in selected if norm(r.get("file_type")) == "pdf")
    n_html = sum(1 for r in selected if norm(r.get("file_type")) == "html")
    print(f"OK: {len(selected)} docs → {OUT_MANIFEST} | pdf={n_pdf} html={n_html}")
    print(f"Filtrados aduaneros candidatos: {len(good)} (de {len(rows)} totales)")

if __name__ == "__main__":
    main()
