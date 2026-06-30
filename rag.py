import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from prompt_template import formater_prompt
from collecte import CHROMA_PATH, EMBEDDING_MODEL

# ==========================================
# CONFIGURATION 
# ==========================================
OLLAMA_MODEL = "qwen2.5:7b"

# Nombre de chunks récupérés depuis ChromaDB
K_RESULTS = 8

# ==========================================
# INITIALISATION (chargée une seule fois)
# ==========================================
llm = ChatOllama(model=OLLAMA_MODEL)
parser = StrOutputParser()


def charger_vector_store():
    """
    Charge la base vectorielle ChromaDB existante.
    
    """
    if not os.path.exists(CHROMA_PATH):
        raise FileNotFoundError(
            f"❌ La base ChromaDB '{CHROMA_PATH}' est introuvable.\n"
            f"   Lance d'abord pipeline_pdf.py pour ingérer les documents."
        )

    print(f"🔌 Chargement de la base vectorielle depuis '{CHROMA_PATH}'...")

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    vector_store = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )

    print("✅ Base vectorielle chargée.")
    return vector_store


# ==========================================
# ÉTAPE 1 : RETRIEVAL
# ==========================================

def retriever(vector_store, question: str, k: int = K_RESULTS) -> list:
    """
    Étape Retrieval : Recherche les chunks les plus pertinents
    dans ChromaDB en fonction de la question posée.

    Retourne une liste de Documents LangChain.
    """
    resultats = vector_store.similarity_search_with_score(
        query=question,
        k=k * 2  # On prend plus pour pouvoir filtrer
    )

    # Filtrer les pages de définitions (>= 65 = annexes/glossaire)
    resultats_filtres = [
        (doc, score) for doc, score in resultats
        if doc.metadata.get("page", 0) < 200
    ]

    # Reprendre les k meilleurs après filtrage
    resultats_filtres = resultats_filtres[:k]

    print(f"   → {len(resultats_filtres)} chunks après filtrage :")
    for i, (doc, score) in enumerate(resultats_filtres):
        source = doc.metadata.get("source", "inconnue")
        page   = doc.metadata.get("page", "?")
        print(f"     [{i+1}] Score: {score:.4f} | {os.path.basename(source)} p.{page}")

    return [doc for doc, score in resultats_filtres]


# ==========================================
# ÉTAPE 2 : CONSTRUCTION DU CONTEXTE
# ==========================================

def construire_contexte(documents: list) -> str:
    """
    Assemble les chunks récupérés en un seul bloc de contexte
    à injecter dans le prompt.
    """
    blocs = []

    for i, doc in enumerate(documents):
        source = doc.metadata.get("source", "inconnue")
        page   = doc.metadata.get("page", "?")

        bloc = (
            f"--- Source {i+1} : {os.path.basename(source)}, page {page} ---\n"
            f"{doc.page_content.strip()}"
        )
        blocs.append(bloc)

    return "\n\n".join(blocs)


# ==========================================
# ÉTAPE 3 : GÉNÉRATION
# ==========================================

def generer_reponse(question: str, contexte: str, historique: list = None) -> str:
    print("\n🤖 Génération de la réponse avec Ollama...")

    if historique is None:
        historique = []
    
    historique_filtre = []
    for h in historique:
        if isinstance(h, dict) and h.get("content", "").strip():
            historique_filtre.append(h)
        elif isinstance(h, str) and h.strip():
            historique_filtre.append(h)

    messages = formater_prompt(contexte, question, historique_filtre)

    reponse = llm.invoke(messages)
    return parser.invoke(reponse)


# ==========================================
# FONCTION PRINCIPALE : ask()
# ==========================================

def ask(question: str, vector_store=None, historique: list = None,verbose: bool = True) -> dict:
    """
    Fonction principale RAG : Retrieval + Generation.

    Paramètres :
    ------------
    - question    : La question posée par l'utilisateur
    - vector_store: Instance ChromaDB (optionnel, rechargée si None)
    - verbose     : Affiche les étapes dans le terminal

    Retourne :
    ----------
    Un dictionnaire contenant :
    {
        "question"  : str,
        "reponse"   : str,
        "sources"   : list[str],   # fichiers + pages utilisés
        "contexte"  : str          # texte brut injecté dans le prompt
    }
    """
    if verbose:
        print("\n" + "=" * 60)
        print(f"❓ QUESTION : {question}")
        print("=" * 60)

    # Chargement du vector store si non fourni
    if vector_store is None:
        vector_store = charger_vector_store()

    # --- RETRIEVAL ---
    documents = retriever(vector_store, question, k=10)

    if not documents:
        return {
            "question": question,
            "reponse": "❌ Aucun document pertinent trouvé dans la base.",
            "sources": [],
            "contexte": ""
        }

    # --- CONSTRUCTION DU CONTEXTE ---
    contexte = construire_contexte(documents)

    # --- GENERATION ---
    reponse = generer_reponse(question, contexte, historique)

    # --- SOURCES utilisées ---
    sources = []
    for doc in documents:
        source = doc.metadata.get("source", "inconnue")
        page   = doc.metadata.get("page", "?")
        entree = f"{os.path.basename(source)} (p.{page})"
        if entree not in sources:
            sources.append(entree)

    if verbose:
        print("\n💬 RÉPONSE :")
        print(reponse)
        print("\n📚 SOURCES UTILISÉES :")
        for s in sources:
            print(f"   - {s}")
        print("=" * 60)

    return {
        "question" : question,
        "reponse"  : reponse,
        "sources"  : sources,
        "contexte" : contexte
    }


# ==========================================
# POINT D'ENTRÉE - TESTS
# ==========================================

if __name__ == "__main__":

    # Chargement unique du vector store pour enchaîner plusieurs questions
    vs = charger_vector_store()

    questions_test = [
        "Quel est le chiffre d'affaires de Microsoft en 2025 ?",
        "Quels sont les principaux risques mentionnés dans le rapport annuel ?",
        "Quelle est la stratégie de croissance de l'entreprise ?",
        "Quels sont les résultats nets par segment d'activité ?",
    ]

    for question in questions_test:
        resultat = ask(question, vector_store=vs)
        print()
