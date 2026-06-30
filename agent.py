import os
import re
from langchain_core.prompts import PromptTemplate, MessagesPlaceholder
from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_classic.memory import ConversationBufferWindowMemory
from langchain_classic.tools import tool
from langchain_ollama import ChatOllama
from rag import ask, retriever, charger_vector_store, construire_contexte
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
    analyser_action,
    score_global_action,
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

        SEUIL = 1.2

        resultats = vector_store.similarity_search_with_score(
            f"actualités news {entreprise}",
            k=5,
            filter={"type": {"$in": ["news", "media_article"]}}
        )

        if not resultats:
            resultats = vector_store.similarity_search_with_score(
                f"actualités news {entreprise}", k=5
            )

        filtres = [(doc, score) for doc, score in resultats if score < SEUIL]

        if not filtres:
            return f"❌ Aucune actualité récente pertinente pour '{entreprise}'."

        return ("Voici les actualités récentes :\n\n"
                + "\n\n".join(doc.page_content.strip() for doc, _ in filtres[:3])
                + "\n\nObservation terminée. Passe directement à Final Answer."
                )  

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
        score_global_action,
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

    FORMAT STRICT — chaque étape UNE SEULE FOIS :

    Question: {input}
    Thought: [ce que je vais faire]
    Action: [nom_outil]
    Action Input: [paramètre]
    Observation: [résultat — NE PAS RÉÉCRIRE, c'est fourni automatiquement]
    Thought: J'ai les informations nécessaires. Je rédige ma réponse finale.
    Final Answer: [réponse complète en français]

    ⚠️ RÈGLE ABSOLUE : Après chaque Observation, tu DOIS écrire Thought puis Final Answer.
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
        max_iterations=5,
        max_execution_time=60,
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
    
    MOTS_SCORE = ["score", "recommandation", "acheter", "vendre", "avis", "décision", "faut-il"]
    MOTS_ANALYSE = ["analyse", "technique", "rsi", "tendance", "graphique", "performance"]

    while True:
        try:
            user_input = input("Vous : ").strip()
            
            if not user_input:
                continue
            if user_input.lower() in ["quit", "exit", "q"]:
                print("👋 Au revoir !")
                break
           
            ticker = detecter_ticker(user_input)  
            question_lower = user_input.lower()
            
            if ticker:
                if any(mot in question_lower for mot in MOTS_SCORE):
                    print("\n🎯 Score global en cours...\n")
                    reponse = score_global_action.invoke(ticker)
                    print(f"\n🤖 FinAI : {reponse}")
                    print("-" * 50 + "\n")
                    historique.append({"role": "user",      "content": user_input})
                    historique.append({"role": "assistant", "content": reponse})
                    continue

                elif any(mot in question_lower for mot in MOTS_ANALYSE):
                    print("\n📊 Analyse technique en cours...\n")
                    reponse = analyser_action.invoke(ticker)
                    print(f"\n🤖 FinAI : {reponse}")
                    print("-" * 50 + "\n")
                    historique.append({"role": "user",      "content": user_input})
                    historique.append({"role": "assistant", "content": reponse})
                    continue

                else:
                    # Cas général avec ticker → on l'injecte dans l'input
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
