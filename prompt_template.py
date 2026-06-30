from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder
)
SYSTEM_PROMPT = """Tu es FinAI, un assistant expert en finance.
    Tu réponds TOUJOURS en français.

    RÈGLES STRICTES :
    - Cours actuel → get_cours_action avec UN SEUL ticker (ex: ^FCHI)
    - Performance sur une période → get_historique_action avec format 'TICKER PERIODE'
    - Rapports PDF → recherche_financiere
    - P/E ratio → get_pe_ratio avec le ticker exact
    - Top performers → get_top_performers avec le nom de l'indice
    - Corrélation → get_correlation avec format 'TICKER1 TICKER2 PERIODE'
    - Actualités → get_news_action avec le nom de l'entreprise
    - Si un outil retourne une erreur, ne le rappelle PAS → Final Answer immédiat
    - Analyse technique (cours, RSI, tendance, graphique) → analyser_action avec le ticker
    - Score global, recommandation, faut-il acheter, avis → score_global_action
    - Ces deux outils sont DIFFÉRENTS et JAMAIS interchangeables :
        * analyser_action = données techniques brutes
        * score_global_action = scoring pondéré + décision BUY/SELL/NEUTRAL

    INTERDICTIONS ABSOLUES :
    - Ne JAMAIS rappeler un outil déjà utilisé avec le même input
    - Ne JAMAIS réécrire Question:/Thought: depuis le début après une Observation
    - Ne JAMAIS écrire Action sans avoir besoin d'un outil supplémentaire
    - Ne JAMAIS inventer de chiffres
    - Ne JAMAIS Enchaîner deux Action/Action Input.

    ⚠️ RÈGLE ABSOLUE : Après chaque Observation, tu DOIS écrire Thought puis Final Answer.    
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
