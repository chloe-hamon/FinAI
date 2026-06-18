import pandas as pd
from langchain_ollama import ChatOllama
from vadim_eval import ask, charger_vector_store


# Modèle utilisé uniquement pour rédiger un commentaire
evaluateur = ChatOllama(
    model="llama3.2:1b",
    temperature=0
)


df_questions = pd.read_csv(
    "questions_evaluation_rag.csv",
    sep=";",
    encoding="utf-8-sig"
)


vector_store = charger_vector_store()


def noter_pertinence(reponse_rag):
    if not reponse_rag or "je ne trouve pas" in reponse_rag.lower():
        return 1
    if len(reponse_rag) < 80:
        return 2
    return 4


def noter_utilisation_rag(sources, contexte):
    if not sources or not contexte:
        return 1
    if len(contexte) < 300:
        return 2
    return 4


def noter_clarte(reponse_rag):
    if not reponse_rag:
        return 1

    longueur = len(reponse_rag)

    if longueur < 80:
        return 2
    elif longueur > 2500:
        return 3
    else:
        return 4


def noter_coherence(reponse_rag):
    texte = reponse_rag.lower()

    mots_problemes = [
        "contradiction",
        "je ne sais pas",
        "aucune information",
        "impossible"
    ]

    if any(mot in texte for mot in mots_problemes):
        return 2

    if len(reponse_rag) < 80:
        return 2

    return 4


def noter_exactitude(reponse_rag, contexte):
    """
    Évaluation simple : si la réponse reprend des éléments du contexte,
    elle est probablement plus exacte.
    """
    if not reponse_rag or not contexte:
        return 1

    mots_reponse = set(reponse_rag.lower().split())
    mots_contexte = set(contexte.lower().split())

    intersection = mots_reponse.intersection(mots_contexte)

    if len(intersection) < 10:
        return 2
    elif len(intersection) < 30:
        return 3
    else:
        return 4


def generer_commentaire(question, reponse_rag, sources, notes):
    prompt = f"""
Tu es un évaluateur de réponses RAG pour un projet d'analyse financière.

Question :
{question}

Réponse générée :
{reponse_rag}

Sources utilisées :
{sources}

Notes attribuées automatiquement :
{notes}

Rédige un commentaire court en français expliquant la qualité de la réponse.
Mentionne si la réponse est pertinente, cohérente, claire, et si elle utilise bien les sources.
Ne donne pas de nouvelles notes.
"""

    try:
        response = evaluateur.invoke(prompt)
        return response.content.strip()
    except Exception as e:
        return f"Commentaire non généré : {e}"


resultats = []

for _, ligne in df_questions.iterrows():
    question = ligne["question"]
    reponse_attendue = ligne["reponse_attendue"]

    print("\nQuestion :", question)

    try:
        resultat_rag = ask(
            question,
            vector_store=vector_store,
            verbose=False
        )

        reponse_rag = resultat_rag["reponse"]
        sources = ", ".join(resultat_rag["sources"])
        contexte = resultat_rag["contexte"]

    except Exception as e:
        reponse_rag = f"Erreur RAG : {e}"
        sources = ""
        contexte = ""

    pertinence = noter_pertinence(reponse_rag)
    coherence = noter_coherence(reponse_rag)
    exactitude = noter_exactitude(reponse_rag, contexte)
    utilisation_rag = noter_utilisation_rag(sources, contexte)
    clarte = noter_clarte(reponse_rag)

    note_totale = pertinence + coherence + exactitude + utilisation_rag + clarte

    notes = {
        "pertinence": pertinence,
        "coherence": coherence,
        "exactitude": exactitude,
        "utilisation_rag": utilisation_rag,
        "clarte": clarte,
        "note_totale": note_totale
    }

    commentaire = generer_commentaire(
        question,
        reponse_rag,
        sources,
        notes
    )

    resultats.append({
        "id": ligne["id"],
        "question": question,
        "reponse_attendue": reponse_attendue,
        "reponse_rag": reponse_rag,
        "sources": sources,
        "pertinence": pertinence,
        "coherence": coherence,
        "exactitude": exactitude,
        "utilisation_rag": utilisation_rag,
        "clarte": clarte,
        "note_totale": note_totale,
        "commentaire": commentaire
    })


df_resultats = pd.DataFrame(resultats)

# Ajout d'une ligne de moyenne
ligne_moyenne = {
    "id": "MOYENNE",
    "question": "",
    "reponse_attendue": "",
    "reponse_rag": "",
    "sources": "",
    "pertinence": round(df_resultats["pertinence"].mean(), 2),
    "coherence": round(df_resultats["coherence"].mean(), 2),
    "exactitude": round(df_resultats["exactitude"].mean(), 2),
    "utilisation_rag": round(df_resultats["utilisation_rag"].mean(), 2),
    "clarte": round(df_resultats["clarte"].mean(), 2),
    "note_totale": round(df_resultats["note_totale"].mean(), 2),
    "commentaire": "Moyenne générale des évaluations RAG."
}

df_resultats = pd.concat(
    [df_resultats, pd.DataFrame([ligne_moyenne])],
    ignore_index=True
)

df_resultats.to_csv(
    "tableau_evaluation_rag_auto.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nTableau d'évaluation créé : tableau_evaluation_rag_auto.csv")
print(df_resultats)