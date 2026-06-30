import os
import shutil
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import gc
import time

from extraction_texte_image import lancer_extraction

# ==========================================
# CONFIGURATION
# ==========================================
DATA_PATH = "Annual report"
CHROMA_PATH = "chroma_db/"

# Paramètres de découpage (Chunking)
# Pour des rapports financiers, 1000 caractères permettent de capturer un paragraphe 
# entier (ou un petit tableau) sans perdre le contexte. 
# L'overlap de 200 assure que les phrases coupées à la frontière de deux chunks restent compréhensibles.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Modèle d'embedding (Gratuit, open-source, tourne en local)
# "all-MiniLM-L6-v2" est un standard très rapide et efficace pour l'anglais.
# Si tes rapports sont en français, tu peux utiliser "dangvantuan/sentence-camembert-base"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ==========================================
# FONCTIONS DU PIPELINE
# ==========================================

def load_documents(directory_path: str):
    """
    Étape 1 : Chargement des documents (Loading)
    Parcourt le dossier spécifié et charge tous les fichiers PDF.
    """
    print(f"📄 Chargement des PDF depuis le dossier '{directory_path}'...")
    # On utilise DirectoryLoader qui va appliquer PyPDFLoader sur chaque fichier ".pdf"
    loader = DirectoryLoader(
        directory_path, 
        glob="**/*.pdf", 
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    print(f"✅ {len(documents)} pages chargées au total.")
    return documents

def split_documents(documents):
    """
    Étape 2 : Découpage du texte (Chunking)
    Divise les pages en petits morceaux (chunks) pour faciliter la recherche vectorielle.
    """
    print("✂️ Découpage des documents en chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True, # Garde la trace d'où vient le texte dans la page
    )
    chunks = text_splitter.split_documents(documents)
    print(f"✅ Documents découpés en {len(chunks)} chunks.")
    return chunks

def save_to_chroma(chunks, reset: bool = True):
    """
    Étape 3 & 4 : Vectorisation (Embeddings) et Stockage (Vector Store)
    Transforme le texte en vecteurs mathématiques et les sauvegarde dans ChromaDB.
    """
    
    print(f"🧠 Initialisation du modèle d'embedding : {EMBEDDING_MODEL}...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    if reset and os.path.exists(CHROMA_PATH):
        print("   🗑️ Suppression de l'ancienne base pour repartir proprement...")
        gc.collect()
        time.sleep(1)
        shutil.rmtree(CHROMA_PATH, ignore_errors=True)


    print(f"💾 Création de la base de données vectorielle ChromaDB dans '{CHROMA_PATH}'...")

    # Création et sauvegarde persistante de la base
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )
    total = vector_store._collection.count()
    print(f"✅ {total} chunks stockés dans ChromaDB.")
    print("🚀 Pipeline terminé avec succès ! Les données sont prêtes à être interrogées.")

# ==========================================
# EXÉCUTION PRINCIPALE
# ==========================================

def lancer_ingestion(dossier_donnees: str, reset: bool = True):
    if not os.path.exists(dossier_donnees):
        print(f"❌ Erreur : Le dossier '{dossier_donnees}' n'existe pas.")
        return

    # Étape 1 : Chunking texte (reset=True ici pour vider l'ancienne base)
    docs = load_documents(dossier_donnees)
    if len(docs) > 0:
        chunks = split_documents(docs)
        save_to_chroma(chunks, reset=reset)
    else:
        print(f"⚠️ Aucun PDF trouvé dans '{dossier_donnees}'.")
        return

    # Étape 2 : Analyse visuelle APRÈS (ChromaDB existe déjà, on ajoute dedans)
    print("📸 Analyse visuelle des pages...")
    for fichier in os.listdir(dossier_donnees):
        if fichier.endswith(".pdf"):
            pdf_path = os.path.join(dossier_donnees, fichier)
            lancer_extraction(pdf_path)

if __name__ == "__main__":
    lancer_ingestion(DATA_PATH, reset=True)
