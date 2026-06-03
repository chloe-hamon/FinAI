from pypdf import PdfReader
import os

dossier = "Annual repport"

for fichier in os.listdir(dossier):
    if fichier.endswith(".pdf"):
        chemin = os.path.join(dossier, fichier)

        print("\n" + "="*50)
        print("TEST :", fichier)
        print("="*50)

        try:
            reader = PdfReader(chemin)

            texte = ""

            for page in reader.pages[:3]:
                contenu = page.extract_text()
                if contenu:
                    texte += contenu

            print(texte[:1000])

        except Exception as e:
            print("Erreur :", e)
