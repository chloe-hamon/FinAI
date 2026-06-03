import os
import fitz
import ollama
from PyPDF2 import PdfReader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

CHROMA_PATH = "chroma_db/"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VISION_MODEL = "qwen2.5vl:3b"

base = os.path.dirname(os.path.abspath(__file__))
pdf_path = os.path.join(base, "Annual report", "2025_AnnualReport_Microsoft.pdf")

output_folder = "Annual report"
images_folder = "pages_images"

os.makedirs(output_folder, exist_ok=True)
os.makedirs(images_folder, exist_ok=True)


# ==========================================
# CHARGEMENT
# ==========================================
def lancer_extraction(pdf_path, max_pages=None):

    nom_pdf = os.path.splitext(os.path.basename(pdf_path))[0]
    output_file = os.path.join(output_folder, nom_pdf + "_test_3_pages.txt")

    reader = PdfReader(pdf_path)
    doc = fitz.open(pdf_path)

    if max_pages is None:
        max_pages = min(3, len(reader.pages))
    else:
        max_pages = min(max_pages, len(reader.pages))

    # ==========================================
    # INITIALISATION CHROMADB                        
    # ==========================================

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_store = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )


    # ==========================================
    # BOUCLE PRINCIPALE
    # ==========================================

    documents_a_ajouter = []    

    with open(output_file, "w", encoding="utf-8") as f:
        for i in range(max_pages):
            print(f"Page {i+1}/{max_pages}...")

            texte_page = reader.pages[i].extract_text() or ""

            page = doc.load_page(i)

            # Image moins lourde
            pix = page.get_pixmap(matrix=fitz.Matrix(1, 1))

            image_path = os.path.join(images_folder, f"page_{i+1}.png")
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
                    "type": "image_analysis"
                }
            )
            documents_a_ajouter.append(document)

# ==========================================
# AJOUT DANS CHROMADB 
# ==========================================

    if documents_a_ajouter:
        vector_store.add_documents(documents_a_ajouter)
        print(f"✅ {len(documents_a_ajouter)} analyses ajoutées dans ChromaDB")

    print("Terminé :", output_file)

if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))
    pdf_path = os.path.join(base, "Annual report", "2025_AnnualReport_Microsoft.pdf")
    lancer_extraction(pdf_path)