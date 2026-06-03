import os
import re
import json
import pandas as pd
import ollama
from PyPDF2 import PdfReader

DOSSIER_PDF = "Annual report"
FICHIER_SORTIE = "donnees_financieres_structurees.csv"


def nettoyer_texte(texte):
    if texte is None:
        return ""

    texte = texte.replace("\n", " ")
    texte = re.sub(r"\s+", " ", texte)
    return texte.strip()


def analyser_metadata_avec_ollama(nom_fichier, texte_debut):
    prompt = f"""
Tu analyses le début d'un rapport financier.

Nom du fichier :
{nom_fichier}

Début du document :
{texte_debut[:2000]}

Retourne uniquement un JSON valide avec cette structure :
{{
  "entreprise": "...",
  "type_rapport": "Annuel ou Trimestriel ou Inconnu",
  "annee": "..."
}}

Ne mets aucun commentaire.
"""

    try:
        response = ollama.chat(
            model="qwen2.5vl:3b",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        contenu = response["message"]["content"].strip()

        # Nettoyage si Ollama ajoute des ```json
        contenu = contenu.replace("```json", "").replace("```", "").strip()

        data = json.loads(contenu)

        entreprise = data.get("entreprise", "Inconnue")
        type_rapport = data.get("type_rapport", "Inconnu")
        annee = data.get("annee", "Inconnue")

        return entreprise, type_rapport, annee

    except Exception as e:
        print("Erreur Ollama metadata :", e)
        return "Inconnue", "Inconnu", "Inconnue"


def detecter_type_contenu(texte):
    texte_min = texte.lower()

    mots_tableau = [
        "table", "statement", "balance sheet", "cash flow",
        "income", "revenue", "assets", "liabilities",
        "compte de résultat", "bilan", "flux de trésorerie"
    ]

    mots_graphique = [
        "graph", "chart", "performance", "growth", "index",
        "trend", "croissance", "évolution", "résultats"
    ]

    if len(texte) < 300:
        return "visuel possible"

    if any(mot in texte_min for mot in mots_graphique):
        return "texte + graphique possible"

    if any(mot in texte_min for mot in mots_tableau):
        return "texte + tableau possible"

    return "texte"


def evaluer_qualite(texte):
    if len(texte) == 0:
        return "nulle"
    elif len(texte) < 300:
        return "faible"
    elif len(texte) < 1000:
        return "moyenne"
    else:
        return "bonne"


def traiter_pdf(chemin_pdf):
    lignes = []
    nom_fichier = os.path.basename(chemin_pdf)

    print(f"\nTraitement : {nom_fichier}")

    try:
        reader = PdfReader(chemin_pdf)

        # Texte du début du document pour détecter les métadonnées
        texte_debut = ""

        for i in range(min(3, len(reader.pages))):
            texte_page = reader.pages[i].extract_text() or ""
            texte_debut += texte_page + "\n"

        entreprise, type_rapport, annee = analyser_metadata_avec_ollama(
            nom_fichier,
            texte_debut
        )

        print("Entreprise détectée :", entreprise)
        print("Type rapport :", type_rapport)
        print("Année :", annee)

        for numero_page, page in enumerate(reader.pages, start=1):
            try:
                texte_brut = page.extract_text()
            except:
                texte_brut = ""

            texte = nettoyer_texte(texte_brut)

            ligne = {
                "entreprise": entreprise,
                "nom_fichier": nom_fichier,
                "type_rapport": type_rapport,
                "annee": annee,
                "numero_page": numero_page,
                "type_contenu": detecter_type_contenu(texte),
                "qualite_extraction": evaluer_qualite(texte),
                "longueur_texte": len(texte),
                "contenu": texte
            }

            lignes.append(ligne)

    except Exception as e:
        print(f"Erreur avec {nom_fichier} : {e}")

    return lignes


def main():
    toutes_les_lignes = []

    for fichier in os.listdir(DOSSIER_PDF):
        if fichier.lower().endswith(".pdf"):
            chemin_pdf = os.path.join(DOSSIER_PDF, fichier)
            lignes_pdf = traiter_pdf(chemin_pdf)
            toutes_les_lignes.extend(lignes_pdf)

    df = pd.DataFrame(toutes_les_lignes)

    df.to_csv(FICHIER_SORTIE, index=False, encoding="utf-8-sig")

    print("\nDataFrame créé avec succès")
    print("Nombre de lignes :", len(df))
    print("Fichier créé :", FICHIER_SORTIE)

    print("\nAperçu :")
    print(df.head())


if __name__ == "__main__":
    main()
