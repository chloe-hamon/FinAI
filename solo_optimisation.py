# ==========================================
# OPTIMISATION PROMPTS + CHROMADB - Solo
# ==========================================

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# --- PARAMÈTRES CHROMADB À TESTER ---
# Valeurs originales du pipeline
CHUNK_SIZE_ORIGINAL = 1000
CHUNK_OVERLAP_ORIGINAL = 200

# Valeurs optimisées 
CHUNK_SIZE_OPTIMISE = 500       # chunks plus petits = plus précis
CHUNK_OVERLAP_OPTIMISE = 100

CHROMA_PATH = "chroma_db/"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# --- PROMPTS OPTIMISÉS ---
SYSTEM_PROMPT_ORIGINAL = "Tu es un assistant financier."

SYSTEM_PROMPT_OPTIMISE = """Tu es un analyste financier expert et rigoureux.
Réponds uniquement en te basant sur le contexte fourni.
Si l'information n'est pas dans le contexte, dis-le clairement.
Structure ta réponse en 3 parties : Analyse, Risques, Recommandation.
Sois concis et factuel."""

def tester_recherche_avec_seuil(query: str, k: int = 3, seuil: float = 1.40):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    
    resultats = db.similarity_search_with_score(query, k=k)
    
    print(f"\n🔍 Requête : '{query}' | k={k} | seuil={seuil}")
    print(f"{'='*50}")
    filtres = [(doc, score) for doc, score in resultats if score <= seuil]
    print(f"Résultats après filtrage : {len(filtres)}/{len(resultats)}")
    for i, (doc, score) in enumerate(filtres):
        print(f"\n📄 Résultat {i+1} | Score : {score:.4f}")
        print(f"Source : {doc.metadata.get('source', 'N/A')}")
        print(f"Contenu : {doc.page_content[:200]}...")

# --- TEST DE RECHERCHE CHROMADB ---
def tester_recherche(query: str, k: int = 3):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    
    resultats = db.similarity_search_with_score(query, k=k)
    
    print(f"\n🔍 Requête : '{query}' | k={k}")
    print(f"{'='*50}")
    for i, (doc, score) in enumerate(resultats):
        print(f"\n📄 Résultat {i+1} | Score : {score:.4f}")
        print(f"Source : {doc.metadata.get('source', 'N/A')}")
        print(f"Contenu : {doc.page_content[:200]}...")

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

def tester_prompt(label: str, prompt_system: str, question: str):
    print(f"\n{'='*50}")
    print(f"🧪 TEST : {label}")
    print(f"{'='*50}")
    llm = ChatOllama(model="llama3.2:3b", temperature=0.1)
    message = HumanMessage(content=f"{prompt_system}\n\nQuestion : {question}")
    response = llm.invoke([message])
    print(response.content)

if __name__ == "__main__":
    # Test avec différentes valeurs de k
    tester_recherche("chiffre d'affaires Apple", k=3)
    tester_recherche("chiffre d'affaires Apple", k=5)
    # Test avec score threshold pour filtrer les mauvais résultats
    print("\n=== TEST AVEC SCORE THRESHOLD ===")
    tester_recherche_avec_seuil("chiffre d'affaires Apple", k=3, seuil=1.40)
    # Test prompts
    question = "Quel est le chiffre d'affaires d'Apple ?"
    tester_prompt("PROMPT ORIGINAL", SYSTEM_PROMPT_ORIGINAL, question)
    tester_prompt("PROMPT OPTIMISÉ", SYSTEM_PROMPT_OPTIMISE, question)