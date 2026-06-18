import os
from langchain_core.prompts import PromptTemplate, MessagesPlaceholder
from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.tools import tool
from langchain_ollama import ChatOllama
from vadim import ask, retriever, charger_vector_store, construire_contexte
from prompt_template import formater_historique, creer_prompt_template, formater_prompt
from prompt_template import SYSTEM_PROMPT, HUMAN_PROMPT


from financial_tools_langchain import (
    calcul_marge_nette,
    calcul_croissance,
    calcul_ratio_endettement,
    verifier_alerte,
    get_donnees_financieres
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
# OUTILS (TOOLS)
# ==========================================
TICKERS = { 
    "Apple":         "AAPL",
    "Tesla":         "TSLA",
    "Microsoft":     "MSFT",
    "Google":        "GOOGL",
    "Amazon":        "AMZN",
    # NASDAQ
    "Nvidia":        "NVDA",
    "Meta":          "META",
    "Netflix":       "NFLX",
    "AMD":           "AMD",
    "Intel":         "INTC",
    # CAC40
    "Airbus":        "AIR.PA",
    "TotalEnergies": "TTE.PA",
    "LVMH":          "MC.PA",
    "BNP Paribas":   "BNP.PA",
    "Sanofi":        "SAN.PA",
    # DAX
    "SAP":           "SAP.DE",
    "Siemens":       "SIE.DE",
    "BMW":           "BMW.DE",
    "Volkswagen":    "VOW3.DE",
    "Adidas":        "ADS.DE",
    # FTSE100
    "HSBC":          "HSBA.L",
    "BP":            "BP.L",
    "Shell":         "SHEL.L",
    "Unilever":      "ULVR.L",
    "AstraZeneca":   "AZN.L",
    # Nikkei225
    "Toyota":        "7203.T",
    "Sony":          "6758.T",
    "SoftBank":      "9984.T",
    "Nintendo":      "7974.T",
    "Mitsubishi":    "8058.T",
    "Honda":         "7267.T",
}
ENTREPRISES_CONNUES = {
    "apple":          "Apple",
    "tesla":          "Tesla",
    "microsoft":      "Microsoft",
    "google":         "Google",
    "alphabet":       "Google",
    "amazon":         "Amazon",
    "nvidia":         "Nvidia",
    "meta":           "Meta",
    "facebook":       "Meta",
    "netflix":        "Netflix",
    "intel":          "Intel",
    "amd":            "AMD",
    "airbus":         "Airbus",
    "totalenergies":  "TotalEnergies",
    "total":          "TotalEnergies",
    "lvmh":           "LVMH",
    "bnp paribas":    "BNP Paribas",
    "bnp":            "BNP Paribas",
    "sanofi":         "Sanofi",
    "sap":            "SAP",
    "siemens":        "Siemens",
    "bmw":            "BMW",
    "volkswagen":     "Volkswagen",
    "vw":             "Volkswagen",
    "adidas":         "Adidas",
    "hsbc":           "HSBC",
    "bp":             "BP",
    "shell":          "Shell",
    "unilever":       "Unilever",
    "astrazeneca":    "AstraZeneca",
    "toyota":         "Toyota",
    "sony":           "Sony",
    "softbank":       "SoftBank",
    "nintendo":       "Nintendo",
    "mitsubishi":     "Mitsubishi",
    "honda":          "Honda",
}

def detecter_entreprise(question: str) -> str | None:
    """Détecte le nom canonique de l'entreprise dans la question."""
    question_lower = question.lower()
    # Trier par longueur décroissante → "bnp paribas" matché avant "bnp"
    for variante in sorted(ENTREPRISES_CONNUES, key=len, reverse=True):
        if variante in question_lower:
            entreprise = ENTREPRISES_CONNUES[variante]
            print(f"   🏢 Entreprise détectée : '{entreprise}'")
            return entreprise
    print("   🏢 Aucune entreprise détectée — recherche globale")
    return None

def detecter_ticker(question: str) -> str | None:
    """
    Détecte le ticker boursier depuis la question.
    Utilisé pour enrichir l'input de get_cours_action.
    """
    question_lower = question.lower()
    for nom, ticker in sorted(TICKERS.items(), key=lambda x: len(x[0]), reverse=True):
        if nom.lower() in question_lower:
            return ticker
    return None

@tool
def recherche_financiere(question: str) -> str:
    """Recherche des informations financières dans les documents.
    UTILISE CET OUTIL EN PREMIER pour trouver les chiffres bruts.
    Input : une question en langage naturel."""

    # ── Détection entreprise + filtre Chroma
    entreprise = detecter_entreprise(question)
    if entreprise:
        # Recherche partielle : "Microsoft" dans "Microsoft_rapportannuel"
        filtre = {"fichier": {"$contains": entreprise}}
    else:
        filtre = None
    resultats = vector_store.similarity_search_with_score(
        question,
        k=10,
        filter=filtre
    )

    print(f"\n   → {len(resultats)} chunks avant filtrage"
          + (f" (filtrés sur '{entreprise}')" if entreprise else " (global)") + " :")
    for doc, score in resultats:
        source = os.path.basename(doc.metadata.get("source", "inconnue"))
        page   = doc.metadata.get("page", "?")
        print(f"     Score: {score:.4f} | {source} p.{page}")

    # ── Filtre 1 : score de similarité
    SEUIL_SCORE = 0.85
    resultats_filtres = [
        (doc, score) for doc, score in resultats
        if score < SEUIL_SCORE
    ]

    # ── Filtre 2 : disclaimers
    MOTS_EXCLUS = [
        "éléments de projection",
        "facteurs d'incertitudes",
        "aucunement à mettre à jour",
        "obligations légales"
    ]
    resultats_filtres = [
        (doc, score) for doc, score in resultats_filtres
        if not any(mot in doc.page_content for mot in MOTS_EXCLUS)
    ]

    print(f"   → {len(resultats_filtres)} chunks après filtrage (seuil={SEUIL_SCORE})")

    if not resultats_filtres:
        if entreprise:
            return (
                f"❌ Aucun document trouvé pour '{entreprise}'. "
                f"Le rapport annuel de {entreprise} n'est peut-être pas "
                f"dans la base de données."
            )
        return (
            "❌ Aucun document pertinent trouvé dans la base de données. "
            "Le rapport demandé n'a probablement pas été ingéré."
        )

    # ── Construction réponse
    blocs = []
    for i, (doc, score) in enumerate(resultats_filtres[:5]):
        source          = doc.metadata.get("source", "inconnue")
        page            = doc.metadata.get("page", "?")
        entreprise_doc  = doc.metadata.get("entreprise", "?")
        blocs.append(
            f"[Source {i+1} : {entreprise_doc} | "
            f"{os.path.basename(source)}, p.{page} | score={score:.4f}]\n"
            f"{doc.page_content.strip()}"
        )

    return "\n\n".join(blocs)

@tool
def get_cours_action(ticker: str) -> str:
    """Récupère le cours actuel d'une action via yfinance. Utilise le symbole boursier ex: AAPL, TSLA, NVDA, MC.PA"""
    import yfinance as yf
    try:
        ticker = ticker.replace("ticker:", "").replace("Ticker:", "").replace("TICKER:", "").strip().upper()
        
        action = yf.Ticker(ticker)
        prix = action.fast_info.last_price
        if prix is None:
            return f"❌ Aucun cours trouvé pour {ticker}. Vérifiez le symbole."
        
        return f"{ticker} : {prix:.2f} USD"
            
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
    verifier_alerte,
    get_donnees_financieres,
]

# Prompt ReAct
prompt = PromptTemplate.from_template("""Tu es FinAI, un assistant expert en finance.
Tu réponds TOUJOURS en français.

Tu as accès aux outils suivants :
{tools}

Noms des outils : {tool_names}

RÈGLES STRICTES :
1. Utilise UN SEUL outil par question si possible
2. Après l'Observation, passe DIRECTEMENT à Final Answer
3. Ne répète JAMAIS le même appel d'outil
4. Si l'Observation contient la réponse, n'appelle plus d'outil

Format OBLIGATOIRE — respecte-le à la lettre :

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
    max_iterations=3,
    agent_kwargs={"stop": ["Observation:"]},
    early_stopping_method="force",
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