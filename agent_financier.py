

from langchain.tools import tool
from langchain_ollama import ChatOllama

# -----------------------
# TOOLS
# -----------------------

@tool
def calcul_marge_nette(resultat_net: float, chiffre_affaires: float) -> float:
    """Calcule la marge nette."""
    return round((resultat_net / chiffre_affaires) * 100, 2)


@tool
def calcul_croissance(valeur_actuelle: float, valeur_precedente: float) -> float:
    """Calcule le taux de croissance."""
    return round(
        ((valeur_actuelle - valeur_precedente)
         / valeur_precedente) * 100,
        2
    )


# -----------------------
# MODELE OLLAMA
# -----------------------

llm = ChatOllama(
    model="llama3.2:1b",
    temperature=0
)

# -----------------------
# TEST SIMPLE
# -----------------------

question = """
Calcule le taux de croissance
entre 100000 et 120000.
"""

reponse = llm.invoke(question)

print(reponse.content)
