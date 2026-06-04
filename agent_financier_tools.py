

from langchain.tools import tool
from langchain_ollama import ChatOllama


# -----------------------
# TOOLS
# -----------------------

@tool
def calcul_croissance(valeur_actuelle: float, valeur_precedente: float) -> float:
    """
    Calcule le taux de croissance.
    """
    return round(
        ((valeur_actuelle - valeur_precedente)
         / valeur_precedente) * 100,
        2
    )


# -----------------------
# MODELE
# -----------------------

llm = ChatOllama(
    model="llama3.2:1b",
    temperature=0
)

# On donne les tools au modèle
llm_with_tools = llm.bind_tools([calcul_croissance])

# -----------------------
# QUESTION
# -----------------------

question = """
Calcule la croissance entre 100000 et 120000.
"""

response = llm_with_tools.invoke(question)

print(response)
