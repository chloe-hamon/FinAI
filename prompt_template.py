from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder
)
SYSTEM_PROMPT = """Tu es FinAI, un assistant financier expert. Tu analyses :
- Rapports annuels et trimestriels
- Données boursières (actions, indices, crypto, devises)
- Ratios financiers (P/E, ROE, EBITDA, etc.)
- Tendances macro-économiques

Règles :
- Réponds toujours en français
- Sois précis et concis
- Cite tes sources si disponibles
- Si tu n'as pas l'information, dis-le clairement
"""

HUMAN_PROMPT = """CONTEXTE EXTRAIT DES DOCUMENTS :
{contexte}

---
HISTORIQUE DE LA CONVERSATION :
{historique}

---
QUESTION : {question}"""

# ==========================================
# CONSTRUCTION DU TEMPLATE
# ==========================================

def creer_prompt_template() -> ChatPromptTemplate:
    """
    Crée et retourne le ChatPromptTemplate complet pour FinAI.
    """
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT),
        HumanMessagePromptTemplate.from_template(HUMAN_PROMPT)
    ])
    return prompt

def formater_prompt(contexte: str, question: str, historique: list = None) -> str:
    """
    Formate le prompt final à envoyer à Ollama.

    Paramètres :
    ------------
    - contexte   : Chunks récupérés depuis ChromaDB
    - question   : Question de l'utilisateur
    - historique : Liste de string ou dict [{"role": "user/assistant", "content": "..."}]

    Retourne :
    ----------
    Liste de dicts [{"role": ..., "content": ...}]  
    """

    historique_str = formater_historique(historique)

    prompt_template = creer_prompt_template()

    messages = prompt_template.format_messages(
        contexte=contexte,
        historique=historique_str,
        question=question
    )

    # Conversion en format Ollama
    messages_ollama = []
    for msg in messages:
        if msg.type == "system":
            messages_ollama.append({"role": "system", "content": msg.content})
        elif msg.type == "human":
            messages_ollama.append({"role": "user", "content": msg.content})

    return messages_ollama


def formater_historique(historique) -> str:
    """
    Convertit l'historique en string lisible.
    Accepte :
    - None ou liste vide
    - Liste de strings : ["Humain: ...", "Assistant: ..."]
    - Liste de dicts   : [{"role": "user", "content": "..."}]
    """
    if not historique:
        return "Aucun historique — début de conversation."

    blocs = []

    for msg in historique[-6:]:  # 6 derniers messages max
        if isinstance(msg, dict):
            role = "Utilisateur" if msg.get("role") == "user" else "FinAI"
            contenu = msg.get("content", "")
        elif isinstance(msg, str):
            contenu = msg
            # Détecter le rôle depuis le préfixe
            if msg.startswith("Humain:") or msg.startswith("Human:"):
                role = "Utilisateur"
            elif msg.startswith("Assistant:") or msg.startswith("FinAI:"):
                role = "FinAI"
            else:
                role = "Message"
        else:
            continue

        blocs.append(f"{role} : {contenu}")

    return "\n".join(blocs) if blocs else "Aucun historique — début de conversation."
