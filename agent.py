import os
import re
from langchain_core.prompts import PromptTemplate, MessagesPlaceholder
from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.tools import tool
from langchain_ollama import ChatOllama
from vadim import ask, retriever, charger_vector_store, construire_contexte
from prompt_template import formater_historique, creer_prompt_template, formater_prompt
from prompt_template import SYSTEM_PROMPT, HUMAN_PROMPT
from financial_tools_langchain import TICKERS,ENTREPRISES_CONNUES


from financial_tools_langchain import (
    calcul_marge_nette,
    calcul_croissance,
    calcul_ratio_endettement,
    verifier_alerte,
    get_donnees_financieres,
    get_historique_action,
    get_pe_ratio,
    get_top_performers,
    get_correlation,
    get_cours_action,
    recherche_financiere,
    detecter_ticker,
    detecter_entreprise,
    analyser_action
)

historique = [] 

# ==========================================
# CONFIGURATION
# ==========================================
OLLAMA_MODEL = "qwen2.5:7b"

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
# Outils
# ==========================================

@tool
def get_news_action(entreprise: str) -> str:
    """Recherche les dernières actualités sur une entreprise ou un sujet financier.
    Input: nom de l'entreprise ou sujet ex: 'Airbus', 'inflation', 'BCE', 'Bitcoin'"""
    resultats = vector_store.similarity_search_with_score(
        f"actualités news {entreprise}",
        k=5,
        filter={"type": {"$in": ["news", "media_article"]}}
    )
    
    if not resultats:
        return f"❌ Aucune actualité trouvée pour '{entreprise}'. Relance chloe.py pour mettre à jour les news."
    
    SEUIL = 1.2
    filtres = [(doc, score) for doc, score in resultats if score < SEUIL]
    
    if not filtres:
        return f"❌ Aucune actualité récente pertinente pour '{entreprise}'."
    
    blocs = []
    for doc, score in filtres[:3]:
        blocs.append(doc.page_content.strip())
    
    return "\n\n".join(blocs)

# ==========================================
# CRÉATION DE L'AGENT
# ==========================================

tools = [
    recherche_financiere,
    get_cours_action,
    calcul_marge_nette,
    calcul_croissance,
    calcul_ratio_endettement,
    verifier_alerte,
    get_donnees_financieres,
    get_historique_action,
    get_pe_ratio,
    get_top_performers,
    get_correlation,
    get_news_action
]

# Prompt ReAct
prompt = PromptTemplate.from_template("""Tu es FinAI, un assistant expert en finance.
Tu réponds TOUJOURS en français.

Outils disponibles :
{tools}

Noms : {tool_names}

RÈGLES STRICTES :
- Cours actuel → get_cours_action avec UN SEUL ticker (ex: ^FCHI)
- Performance sur une période → get_historique_action avec format 'TICKER PERIODE' (ex: '^FCHI 1mo', '^GSPC 6mo')
- Pour comparer 2 actifs → appelle get_historique_action DEUX FOIS, une par ticker
- Données fondamentales → get_donnees_financieres
- Rapports PDF → recherche_financiere
- Après chaque Observation utile → Final Answer immédiatement
- Ne JAMAIS passer plusieurs tickers dans un seul appel
- P/E ratio → get_pe_ratio avec le ticker exact (LVMH=MC.PA, Sanofi=SAN.PA)
- Top performers d'un indice → get_top_performers avec le nom de l'indice (ex: CAC40, DAX)
- Corrélation entre deux actifs → get_correlation avec format 'TICKER1 TICKER2 PERIODE'
- Dès que tu as toutes les données nécessaires pour répondre, passe IMMÉDIATEMENT à Final Answer sans rappeler d'outil
- Si tu as déjà appelé le même outil avec le même input → STOP, passe à Final Answer
- Pour COMPARER deux actifs : fais DEUX appels séparés, un par un :
  1er appel → Action: get_historique_action / Action Input: TSLA 3mo
  Attends l'Observation
  2ème appel → Action: get_historique_action / Action Input: NVDA 3mo  
  Attends l'Observation
  Puis Final Answer
- Il est INTERDIT d'écrire deux "Action Input:" dans le même bloc
- Analyse d'opportunité d'achat ou tendance d'un actif → analyser_action avec le ticker, puis Final Answer avec ton interprétation
- Actualités sur une entreprise ou sujet → get_news_action avec le nom (ex: 'Airbus', 'inflation BCE')
- Ne jamais inventer de chiffres économiques si l'outil ne retourne rien

Format OBLIGATOIRE :

Question: la question posée
Thought: ce que je dois faire
Action: nom_de_loutil
Action Input: paramètre
Observation: résultat de l'outil
Thought: J'ai suffisamment d'informations pour répondre
Final Answer: ma réponse complète en français

---
Historique : {chat_history}

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
    max_iterations=10,
    max_execution_time=60,
    return_intermediate_steps=False,
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

    historique = []

    while True:
        try:
            user_input = input("Vous : ").strip()
            
            if not user_input:
                continue
            if user_input.lower() in ["quit", "exit", "q"]:
                print("👋 Au revoir !")
                break
           
            ticker = detecter_ticker(user_input)
            if ticker:
                user_input = f"{user_input} (ticker: {ticker})"

            print("\n🤔 Analyse en cours...\n")

            historique_formate = formater_historique(historique)

            result = agent_executor.invoke({
                "input": user_input,
                "chat_history": historique_formate 
            })

            reponse = result["output"]

            historique.append({"role": "user", "content": user_input})
            historique.append({"role": "assistant", "content": reponse})

            print(f"\n🤖 FinAI : {reponse}")
            print("-" * 50 + "\n")

        except KeyboardInterrupt:
            print("\n👋 Au revoir !")
            break
        except Exception as e:
            print(f"❌ Erreur : {e}\n")