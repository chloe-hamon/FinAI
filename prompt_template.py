from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

# ==========================================
# SYSTEM PROMPT — Personnalité de l'agent
# ==========================================

SYSTEM_PROMPT = """Tu es FinAI, un assistant financier expert spécialisé dans :
- L'analyse de rapports annuels et trimestriels
- L'interprétation de données boursières (actions, indices, crypto, taux de change)
- Le calcul et l'explication de ratios financiers (P/E, ROE, EBITDA, etc.)
- La détection d'anomalies et d'alertes dans les données financières

Règles strictes :
1. Tu te bases UNIQUEMENT sur le contexte fourni pour répondre
2. Si une information est absente du contexte, tu le dis clairement : "Je n'ai pas cette information dans les documents disponibles."
3. Tu cites toujours tes sources (nom du fichier + page)
4. Tu structures tes réponses avec des titres clairs quand c'est pertinent
5. Tu donnes les chiffres avec leurs unités (millions $, %, etc.)
6. Tu restes factuel et neutre — tu ne fais pas de recommandations d'investissement

Langue : Tu réponds dans la langue de la question posée.
"""

# ==========================================
# HUMAN PROMPT — Structure de la question
# ==========================================

HUMAN_PROMPT = """
CONTEXTE EXTRAIT DES DOCUMENTS :
{contexte}

---
HISTORIQUE DE LA CONVERSATION :
{historique}

---
QUESTION :
{question}

---
RÉPONSE :
"""

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
    - historique : Liste de dict [{"role": "user/assistant", "content": "..."}]

    Retourne :
    ----------
    - Le prompt formaté sous forme de string
    """

    # Formatage de l'historique
    if historique:
        blocs_historique = []
        for msg in historique[-6:]:  # On garde les 6 derniers messages max
            role = "Utilisateur" if msg["role"] == "user" else "FinAI"
            blocs_historique.append(f"{role} : {msg['content']}")
        historique_str = "\n".join(blocs_historique)
    else:
        historique_str = "Aucun historique — début de conversation."

    # Construction du prompt final
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


# ==========================================
# TEST STANDALONE
# ==========================================

if __name__ == "__main__":
    print("Test du prompt template...\n")

    contexte_test = """--- Source 1 : 2025_AnnualReport_Microsoft.pdf, page 42 ---
    Total Revenue: $211.9 billion, up 16% year-over-year.
    Cloud revenue grew 23% to $135.7 billion."""

    historique_test = [
        {"role": "user", "content": "Bonjour, parle-moi de Microsoft"},
        {"role": "assistant", "content": "Bonjour ! Je suis prêt à analyser les données Microsoft."}
    ]

    messages = formater_prompt(
        contexte=contexte_test,
        question="Quel est le chiffre d'affaires cloud de Microsoft ?",
        historique=historique_test
    )

    print(f"Nombre de messages générés : {len(messages)}\n")
    for msg in messages:
        print(f"[{msg['role'].upper()}]")
        print(msg['content'][:300])
        print("...\n")

    print("✅ Prompt template opérationnel !")