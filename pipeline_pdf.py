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
    all_docs = []
    pdf_files = [f for f in os.listdir(directory_path) if f.endswith(".pdf")]

    if not pdf_files:
        print(f"⚠️ Aucun PDF trouvé dans '{directory_path}'.")
        return []

    for fichier in pdf_files:
        path = os.path.join(directory_path, fichier)
        try:
            loader = PyPDFLoader(path)
            docs = loader.load()
            print(f"   ✅ {fichier} — {len(docs)} pages")
            all_docs.extend(docs)
        except Exception as e:
            print(f"   ❌ Erreur sur {fichier} : {e} — ignoré")

    print(f"✅ {len(all_docs)} pages chargées au total.")
    return all_docs

def evaluer_qualite(texte):
    if len(texte) == 0:
        return "nulle"
    elif len(texte) < 300:
        return "faible"
    elif len(texte) < 1000:
        return "moyenne"
    else:
        return "bonne"

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
    print(f"   → {len(chunks)} chunks bruts générés")
    
    # ── FILTRE QUALITÉ ──────────────────────────────────────────
    chunks_filtres = []
    ignores = 0

    for chunk in chunks:
        qualite = evaluer_qualite(chunk.page_content)
        if qualite in ("nulle", "faible"):
            ignores += 1
        else:
            chunk.metadata["qualite"] = qualite   # utile pour debug
            chunks_filtres.append(chunk)

    print(f"   → {ignores} chunks ignorés (qualité nulle/faible)")
    print(f"✅ {len(chunks_filtres)} chunks conservés après filtrage.")
    return chunks_filtres

def save_to_chroma(chunks, reset: bool = False):
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

    BATCH_SIZE = 500
    vector_store = None

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        print(f"   → Batch {i // BATCH_SIZE + 1} : {len(batch)} chunks...")

        if vector_store is None:
            vector_store = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                persist_directory=CHROMA_PATH
            )
        else:
            vector_store.add_documents(batch)

    total = vector_store._collection.count()
    print(f"✅ {total} chunks stockés dans ChromaDB.")
    return vector_store


def ajouter_extractions_visuelles(dossier: str, vector_store: Chroma):
    """
    Étape 5 : Extraction visuelle (images, tableaux) et ajout dans ChromaDB.
    """
    print("📸 Analyse visuelle des pages (images/tableaux)...")

    pdf_files = [f for f in os.listdir(dossier) if f.endswith(".pdf")]

    for fichier in pdf_files:
        pdf_path = os.path.join(dossier, fichier)
        try:
            # lancer_extraction doit retourner une liste de Documents LangChain
            docs_visuels = lancer_extraction(pdf_path)

            if docs_visuels:
                vector_store.add_documents(docs_visuels)
                print(f"   ✅ {fichier} — {len(docs_visuels)} éléments visuels ajoutés")
            else:
                print(f"   ⚠️ {fichier} — aucun élément visuel extrait")

        except Exception as e:
            print(f"   ❌ Erreur visuelle sur {fichier} : {e} — ignoré")

# ==========================================
# EXÉCUTION PRINCIPALE
# ==========================================

def lancer_ingestion(dossier_donnees: str, reset: bool = False):
    if not os.path.exists(dossier_donnees):
        print(f"❌ Erreur : Le dossier '{dossier_donnees}' n'existe pas.")
        return

    docs = load_documents(dossier_donnees)
    if not docs:
        return

    chunks = split_documents(docs)
    vector_store = save_to_chroma(chunks, reset=reset)

    # Étape 4 : Visuel — ajout dans la même base
    ajouter_extractions_visuelles(dossier_donnees, vector_store)

    print("\n🚀 Pipeline terminé ! La base est prête à être interrogée.")

if __name__ == "__main__":
    lancer_ingestion(DATA_PATH, reset=True)