import os
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import tool
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from vadim import ask, retriever, charger_vector_store, construire_contexte

from financial_tools_langchain import (
    calcul_marge_nette,
    calcul_croissance,
    calcul_ratio_endettement,
    verifier_alerte
)

# ==========================================
# CONFIGURATION
# ==========================================
OLLAMA_MODEL = "qwen2.5vl:7b"

model = ChatOllama(
    model=OLLAMA_MODEL,
    temperature=0
)

print("📂 Chargement de la base vectorielle...")
vector_store = charger_vector_store()
print("✅ Base prête.\n")

# ==========================================
# MEMORY
# ==========================================
memory = ConversationBufferMemory(
    memory_key="chat_history",
    input_key="input",
    return_messages=False
)


# ==========================================
# OUTILS (TOOLS)
# ==========================================
@tool
def recherche_financiere(question: str) -> str:
    """Recherche des informations dans les documents financiers indexés (rapports annuels, bilans, analyses)."""
    # Récupérer l'historique depuis la memory sous forme de liste propre
    historique_raw = memory.load_memory_variables({}).get("chat_history", "")

    historique = [
        ligne for ligne in historique_raw.split("\n")
        if ligne.strip()
    ] if isinstance(historique_raw, str) and historique_raw.strip() else []

    resultat = ask(
        question=question,
        vector_store=vector_store,
        historique=historique,
        verbose=False
    )
    return resultat["reponse"]

@tool
def get_cours_action(ticker: str) -> str:
    """Récupère le cours actuel d'une action via yfinance. Utilise le symbole boursier ex: AAPL, TSLA, NVDA, MC.PA"""
    import yfinance as yf
    try:
        action = yf.Ticker(ticker.upper().strip())
        prix = action.fast_info.last_price
        if prix is None:
            return f"❌ Aucun cours trouvé pour {ticker}. Vérifiez le symbole."
        
        try:
            return f"{ticker.upper()} : {prix:.2f} USD"
        except Exception:
            return f"{ticker.upper()} : {prix:.2f} USD"
            
    except Exception as e:
        return f"❌ Impossible de récupérer {ticker} : {e}"
# ==========================================
# CRÉATION DE L'AGENT
# ==========================================

tools = [
    recherche_financiere,
    get_cours_action,
    calcul_marge_nette,
    calcul_croissance,
    calcul_ratio_endettement,
    verifier_alerte
]

# Prompt ReAct
prompt = PromptTemplate.from_template("""Tu es FinAI, un assistant expert en finance et marchés boursiers.
Tu réponds toujours en français, de façon claire et précise.

Tu as accès aux outils suivants :
{tools}

Outils disponibles : {tool_names}

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
    memory=memory,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=5
)
# ==========================================
# BOUCLE DE CHAT
# ==========================================

if __name__ == "__main__":
    print("🤖 FinAI Agent — Assistant Financier")
    print(f"🛠️  Outils disponibles : {', '.join([t.name for t in tools])}")
    print("📊 Actions disponibles : Apple, Tesla, NVIDIA, LVMH...")
    print("📈 Indices disponibles : CAC40, S&P500, NASDAQ, DAX...")
    print("💬 Tapez 'quit' pour quitter\n")

    
    while True:
        try:
            user_input = input("Vous : ").strip()
            
            if not user_input:
                continue
            if user_input.lower() in ["quit", "exit", "q"]:
                print("👋 Au revoir !")
                break

            print("\n🤔 Analyse en cours...\n")
            
            

            result = agent_executor.invoke({
                "input": user_input
            })


            reponse = result["output"]
            print(f"\n🤖 FinAI : {reponse}")
            print("-" * 50 + "\n")

        except KeyboardInterrupt:
            print("\n👋 Au revoir !")
            break
        except Exception as e:
            print(f"❌ Erreur : {e}\n")