import os
import fitz
import ollama
from PyPDF2 import PdfReader

pdf_path = r"C:\Users\mache\Annual repport\2025_AnnualReport_Microsoft.pdf"

output_folder = "analyse_resultats"
images_folder = "pages_images"

os.makedirs(output_folder, exist_ok=True)
os.makedirs(images_folder, exist_ok=True)

nom_pdf = os.path.splitext(os.path.basename(pdf_path))[0]
output_file = os.path.join(output_folder, nom_pdf + "_test_3_pages.txt")

reader = PdfReader(pdf_path)
doc = fitz.open(pdf_path)

max_pages = min(3, len(reader.pages))

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
            model="qwen2.5vl:3b",
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

print("Terminé :", output_file)
