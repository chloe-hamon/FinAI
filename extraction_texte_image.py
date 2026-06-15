import os
import fitz
import ollama
from PyPDF2 import PdfReader
from langchain_core.documents import Document


CHROMA_PATH = "chroma_db/"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VISION_MODEL = "qwen2.5vl:7b"

output_folder = "Annual report"
images_folder = "pages_images"

os.makedirs(output_folder, exist_ok=True)
os.makedirs(images_folder, exist_ok=True)


# ==========================================
# CHARGEMENT
# ==========================================
def lancer_extraction(pdf_path, max_pages=None):
    """
    Analyse chaque page d'un PDF avec un modèle de vision (Ollama).
    
    Retourne une liste de Documents LangChain prêts à être ajoutés dans ChromaDB.
    """
    nom_pdf = os.path.splitext(os.path.basename(pdf_path))[0]
    output_file = os.path.join(output_folder, f"{nom_pdf}_analyse_visuelle.txt")

    reader = PdfReader(pdf_path)
    doc_fitz = fitz.open(pdf_path)
    total_pages = len(reader.pages)
    
    if max_pages is None:
        max_pages = total_pages
    else:
        max_pages = min(max_pages, total_pages)

    print(f"📸 Extraction visuelle : {nom_pdf} ({max_pages}/{total_pages} pages)")
    
    # ==========================================
    # BOUCLE PRINCIPALE
    # ==========================================

    documents_a_ajouter = []    

    with open(output_file, "w", encoding="utf-8") as f:
        for i in range(max_pages):
            print(f"Page {i+1}/{max_pages}...")

            texte_page = reader.pages[i].extract_text() or ""

            page = doc_fitz.load_page(i)
            # Image moins lourde
            pix = page.get_pixmap(matrix=fitz.Matrix(1, 1))

            image_path = os.path.join(images_folder, f"{nom_pdf}_page_{i + 1}.png")
            pix.save(image_path)

            prompt = f"""
Analyse rapidement cette page de rapport financier.

Contexte texte :
{texte_page[:1000]}

Dis seulement :
1. Type de contenu
2. Chiffres importants
3. Graphiques/tableaux visibles
4. Résumé utile pour un RAG
"""

            try:
                response = ollama.chat(
                    model=VISION_MODEL,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                            "images": [image_path]
                        }
                    ]
                )
                analyse = response["message"]["content"]

            except Exception as e:
                print(f"   ❌ Erreur vision page {i + 1} : {e} — page ignorée")
                analyse = f"[Erreur d'analyse visuelle : {e}]"

            f.write(f"\nPAGE {i+1}\n")
            f.write("=" * 50 + "\n")
            f.write("TEXTE PDFREADER :\n")
            f.write(texte_page[:2000])
            f.write("\n\nANALYSE OLLAMA :\n")
            f.write(analyse)
            f.write("\n\n")

            document = Document(
                page_content=analyse,
                metadata={
                    "source": nom_pdf,
                    "page": i + 1,
                    "type": "image_analysis",
                    "pdf_path": pdf_path
                }
            )
            documents_a_ajouter.append(document)

    doc_fitz.close()  

    print(f"✅ {len(documents_a_ajouter)} analyses prêtes | Fichier debug : {output_file}")
    return documents_a_ajouter


if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))
    pdf_path = os.path.join(base, "Annual report", "2025_AnnualReport_Microsoft.pdf")

     # Test sur 3 pages seulement en mode standalone
    docs = lancer_extraction(pdf_path, max_pages=3)

    print(f"\n📋 {len(docs)} documents retournés")
    for doc in docs:
        print(f"  Page {doc.metadata['page']} — {len(doc.page_content)} caractères")
