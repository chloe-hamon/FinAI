import os
import re
from langchain_core.prompts import PromptTemplate, MessagesPlaceholder
from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_classic.memory import ConversationBufferWindowMemory
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
def creer_agent():
    model = ChatOllama(model="qwen2.5:7b", temperature=0)

    print("📂 Chargement de la base vectorielle...")
    vector_store = charger_vector_store()
    print("✅ Base prête.")

    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        input_key="input",
        k=3,
        return_messages=True
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
            return (f"❌ Aucune actualité trouvée pour '{entreprise}'."
                   f"STOP. Donne Final Answer maintenant : les actualités ne sont pas disponibles.")


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
        get_news_action,
        analyser_action,
    ]

    # Prompt ReAct
    prompt = PromptTemplate.from_template("""Tu es FinAI, un assistant expert en finance.
    Tu réponds TOUJOURS en français.

    Outils disponibles :
    {tools}

    Noms : {tool_names}

    RÈGLES STRICTES :
    - Cours actuel → get_cours_action avec UN SEUL ticker (ex: ^FCHI)
    - Performance sur une période → get_historique_action avec format 'TICKER PERIODE'
    - Données fondamentales → get_donnees_financieres
    - Rapports PDF → recherche_financiere
    - P/E ratio → get_pe_ratio avec le ticker exact
    - Top performers → get_top_performers avec le nom de l'indice
    - Corrélation → get_correlation avec format 'TICKER1 TICKER2 PERIODE'
    - Actualités → get_news_action avec le nom de l'entreprise
    - Analyse d'opportunité → analyser_action avec le ticker
    - Si un outil retourne '❌ Aucun document trouvé' ne rappelle PAS le même outil avec la même entrée passe directement à Final Answer avec ce que tu sais
    - Pour comparer deux entreprises, appelle recherche_financiere UNE FOIS pour chaque entreprise séparément, puis synthétise

    INTERDICTIONS ABSOLUES :
    Ne JAMAIS écrire Action Input sans Action avant
    Ne JAMAIS appeler le même outil deux fois avec le même input
    Ne JAMAIS inventer de chiffres si l'outil ne retourne rien
    Ne JAMAIS écrire deux Action/Action Input dans le même bloc

    OBLIGATION :
    Dès que tu as les données → Final Answer IMMÉDIATEMENT
    Si aucun document trouvé → Final Answer avec "Information non disponible"
    Toujours citer la source exacte (nom du PDF, page)
    
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


    agent = create_react_agent(llm=model,tools=tools,prompt=prompt)

    def handle_error(error) -> str:
        error_str = str(error)
        # Détection boucle (même outil répété)
        if "get_news_action" in error_str or "get_cours_action" in error_str:
            return (
                "Tu boucles sur le même outil. "
                "Thought: Je n'ai pas les données.\n"
                "Final Answer: Information non disponible pour cette entreprise."
            )
            
        if "Invalid Format" in error_str or "Missing 'Action:'" in error_str:
            return (
                "ERREUR DE FORMAT DÉTECTÉE.\n"
                "Tu as les informations nécessaires. "
                "Réponds MAINTENANT avec :\n"
                "Thought: J'ai suffisamment d'informations.\n"
                "Final Answer: [ta réponse complète en français]"
            )
        if "Aucun document trouvé" in error_str or "Aucune actualité" in error_str:
            return (
                "Cet outil ne contient pas les données demandées. "
                "Ne rappelle plus cet outil. "
                "Donne immédiatement une Final Answer en expliquant "
                "que les données ne sont pas disponibles."
            )
    
        return (
            "Erreur inconnue. Ne rappelle plus le dernier outil. "
            "Donne ta Final Answer maintenant."
        )

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        handle_parsing_errors= handle_error,
        max_iterations=10,
        max_execution_time=120,
        return_intermediate_steps=False,
    )
    return agent_executor, tools

# ==========================================
# BOUCLE DE CHAT
# ==========================================

if __name__ == "__main__":
    agent_executor, tools = creer_agent()

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