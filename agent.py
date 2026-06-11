import os
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import tool
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
# Importer nos fonctions existantes
from vadim import ask, retriever, charger_vector_store, construire_contexte

from financial_tools_langchain import (
    calcul_marge_nette,
    calcul_croissance,
    calcul_ratio_endettement,
    verifier_alerte_seuil
)

# ==========================================
# CONFIGURATION
# ==========================================
OLLAMA_MODEL = "llama3.2:1b"

model = ChatOllama(
    model=OLLAMA_MODEL,
    temperature=0
)

print("🔌 Chargement de la base vectorielle...")
vector_store = charger_vector_store()
print("✅ Base prête.\n")

# ==========================================
# OUTILS (TOOLS)
# ==========================================
@tool
def recherche_rag(question: str) -> str:
    """Recherche des informations dans les rapports financiers."""
    resultat = ask(question, vector_store=vector_store)
    return resultat["reponse"]

@tool
def get_cours_action(ticker: str) -> str:
    """Récupère le cours actuel d'une action via yfinance."""
    import yfinance as yf
    try:
        action = yf.Ticker(ticker)
        prix = action.fast_info.last_price
        variation = action.fast_info.get("regularMarketChangePercent", None)
        if variation:
            return f"{ticker} : {prix:.2f} ({variation:+.2f}%)"
        return f"{ticker} : {prix:.2f}"
    except Exception as e:
        return f"❌ Impossible de récupérer {ticker} : {e}"
# ==========================================
# CRÉATION DE L'AGENT
# ==========================================

tools = [
    recherche_rag,
    get_cours_action,
    calcul_marge_nette,
    calcul_croissance,
    calcul_ratio_endettement,
    verifier_alerte_seuil
]

# Prompt ReAct
prompt = PromptTemplate.from_template("""Tu es FinAI, un assistant expert en finance et marchés boursiers.
Tu réponds toujours en français, de façon claire et précise.

Tu as accès aux outils suivants :
{tools}

Noms des outils disponibles : {tool_names}

Pour répondre, utilise ce format :
Question: la question posée
Thought: réfléchis à ce que tu dois faire
Action: le nom de l'outil à utiliser
Action Input: l'entrée pour l'outil
Observation: le résultat de l'outil
... (répète Thought/Action/Observation si nécessaire)
Thought: j'ai maintenant la réponse finale
Final Answer: ta réponse complète en français

Historique de la conversation :
{chat_history}

Question: {input}
Thought: {agent_scratchpad}""")

# Agent
agent = create_react_agent(
    llm=model,
    tools=tools,
    prompt=prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=5
)
# ==========================================
# BOUCLE DE CHAT
# ==========================================

if __name__ == "__main__":
    print("🤖 FinAI Agent — Assistant Financier")
    print("=" * 50)
    print(f"🛠️  Outils disponibles : {', '.join([t.name for t in tools])}")
    print("📊 Actions disponibles : Apple, Tesla, NVIDIA, LVMH...")
    print("📈 Indices disponibles : CAC40, S&P500, NASDAQ, DAX...")
    print("💬 Tapez 'quit' pour quitter\n")

    chat_history = []


    while True:
        try:
            user_input = input("Vous : ").strip()
            
            if not user_input:
                continue
            if user_input.lower() in ["quit", "exit", "q"]:
                print("👋 Au revoir !")
                break

            print("\n🤔 Analyse en cours...\n")
            
            # Formater l'historique
            historique = "\n".join(chat_history) if chat_history else "Aucun historique."


            result = agent_executor.invoke({
                "input": user_input,
                "chat_history": historique
            })


            reponse = result["output"]
            print(f"\n🤖 FinAI : {reponse}")
            print("-" * 50 + "\n")

            # Sauvegarder dans l'historique
            chat_history.append(f"Humain: {user_input}")
            chat_history.append(f"Assistant: {reponse}")

            # Limiter l'historique aux 10 derniers échanges
            if len(chat_history) > 20:
                chat_history = chat_history[-20:]

        except KeyboardInterrupt:
            print("\n👋 Au revoir !")
            break
        except Exception as e:
            print(f"❌ Erreur : {e}\n")