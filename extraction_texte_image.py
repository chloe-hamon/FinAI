import os
import fitz
import ollama
from PyPDF2 import PdfReader
from langchain_core.documents import Document
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

CHROMA_PATH = "chroma_db/"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VISION_MODEL = "qwen2.5vl:3b"

output_folder = "Annual report"
images_folder = "pages_images"

os.makedirs(output_folder, exist_ok=True)
os.makedirs(images_folder, exist_ok=True)

# ==========================================
# CHARGEMENT
# ==========================================
def lancer_extraction(pdf_path: str, pages_filter: list[int], existing_ids: set = None,
    max_workers: int = 3) -> list[Document]:
    """
    Analyse les pages d'un PDF avec un modèle de vision (Ollama).

    Args:
        pdf_path     : chemin vers le PDF
        max_pages    : limite max de pages à analyser (optionnel)
        pages_filter : liste d'index (0-based) des pages à analyser.
                       Si fourni, max_pages est ignoré.
        existing_ids : ensemble d'IDs existants à ignorer.
        max_workers  : nombre maximum de workers pour l'analyse parallèle.

    Retourne une liste de Documents LangChain prêts pour ChromaDB.
    """
    nom_pdf = os.path.splitext(os.path.basename(pdf_path))[0]
    output_file = os.path.join(output_folder, f"{nom_pdf}_analyse_visuelle.txt")

    doc_fitz    = fitz.open(pdf_path)
    reader      = PdfReader(pdf_path)
    total_pages = len(reader.pages)

    # ── 1. Filtrer les index valides ─────────────────────────────
    pages_a_traiter = [i for i in pages_filter if 0 <= i < total_pages]

    # ── 2. Skip pages déjà indexées ──────────────────────────────
    if existing_ids:
        avant =len(pages_a_traiter)
        pages_a_traiter = [
            i for i in pages_a_traiter
            if f"{nom_pdf}_p{i + 1}_visual" not in existing_ids
        ]
        skipped = avant - len(pages_a_traiter)
        if skipped:
            print(f"   ⏭️  {nom_pdf} — {skipped} pages déjà indexées, ignorées")

    if not pages_a_traiter:
        print(f"   ✅ {nom_pdf} — toutes les pages déjà analysées")
        doc_fitz.close()
        return []

    print(f"📸 Analyse visuelle : {nom_pdf} "
          f"({len(pages_a_traiter)} nouvelles pages / {total_pages} total) "
          f"[{max_workers} workers]")

    # ── 3. Pré-générer toutes les images (séquentiel, thread-safe) 
    image_paths = {}
    for i in pages_a_traiter:
        page = doc_fitz.load_page(i)
        pix  = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
        image_path = os.path.join(images_folder, f"{nom_pdf}_page_{i + 1}.png")
        pix.save(image_path)
        image_paths[i] = image_path

    doc_fitz.close()

    # ── 4. Pré-extraire tous les textes (séquentiel, thread-safe)
    textes = {i: (reader.pages[i].extract_text() or "") for i in pages_a_traiter}

    # ── Verrou pour l'écriture dans le fichier debug ──────────────
    file_lock = threading.Lock()
    documents = []
    doc_lock  = threading.Lock()

    def traiter_page(i: int):
        """Traite une page : appel Ollama + création Document."""
        texte_page = textes[i]
        image_path = image_paths[i]


        prompt = f"""Analyse rapidement cette page de rapport financier.
Texte extrait:
{texte_page[:1000]}
Dis seulement :1. Type de contenu 2.Chiffres importants 3. Graphiques/tableaux visibles 4.Résumé utile pour un RAG
"""

        # ── Appel Ollama ────────────────────────────────────────
        try:
            response = ollama.chat(
                model=VISION_MODEL,
                messages=[{
                    "role"   : "user",
                    "content": prompt,
                    "images" : [image_path]
                }]
            )
            analyse = response["message"]["content"]

        except Exception as e:
            print(f"   ❌ Erreur vision page {i + 1} : {e} — ignorée")
            analyse = f"[Erreur analyse visuelle : {e}]"

        # ── Écriture fichier debug (thread-safe) ──────────────────
        with file_lock:
            with open(output_file, "a", encoding="utf-8") as f:
                f.write(f"\nPAGE {i + 1}\n")
                f.write("=" * 50 + "\n")
                f.write("TEXTE PDFREADER :\n")
                f.write(texte_page[:2000])
                f.write("\n\nANALYSE OLLAMA :\n")
                f.write(analyse)
                f.write("\n\n")

        # ── Document LangChain ────────────────────────────────────
        chunk_id = f"{nom_pdf}_p{i + 1}_visual"
        doc = Document(
            page_content=analyse,
            metadata={
                "source"   : os.path.basename(pdf_path),
                "pdf_path" : pdf_path,
                "page"     : i + 1,
                "type"     : "visual_analysis",
                "chunk_id" : chunk_id
            }
        )
        with doc_lock:
            documents.append(doc)

        print(f"   ✅ Page {i + 1} analysée ({len(analyse)} chars)")
        return i  # retourner l'index pour le suivi

    # ── 5. Parallélisme ───────────────────────────────────────────
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(traiter_page, i): i for i in pages_a_traiter}
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"   ❌ Future échouée page {futures[future] + 1} : {e}")

    # ── 6. Trier par numéro de page (ordre cohérent pour ChromaDB)
    documents.sort(key=lambda d: d.metadata["page"])

    print(f"✅ {len(documents)} analyses visuelles prêtes | Debug : {output_file}")
    return documents


if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))
    pdf_path = os.path.join(base, "Annual report", "2025_AnnualReport_Microsoft.pdf")

    docs = lancer_extraction(pdf_path, pages_filter=[0, 1, 2])

    print(f"\n📋 {len(docs)} documents retournés")
    for doc in docs:
        print(f"  Page {doc.metadata['page']} — {len(doc.page_content)} caractères")