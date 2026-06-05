import os
import fitz  # PyMuPDF : L'outil ultime pour manipuler les PDF
import base64
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage

# ==========================================
# CONFIGURATION
# ==========================================
DATA_PATH = "data_pdf/"
CHROMA_PATH = "chroma_db/"
IMG_TMP_PATH = "tmp_images/" # Dossier temporaire pour stocker les graphiques extraits

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# Modèle de vision local (nécessite l'installation d'Ollama sur ta machine)
# bien installer ollama
VISION_MODEL = "llava" 

# ==========================================
# FONCTIONS DE VISION
# ==========================================

def image_to_base64(image_path):
    """Convertit une image en base64 pour l'envoyer au LLM de vision."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def describe_image(image_path):
    """Demande au modèle de vision (LLaVA) de décrire le graphique financier."""
    print(f"   👁️ Analyse de l'image {os.path.basename(image_path)} par l'IA...")
    
    llm = ChatOllama(model=VISION_MODEL, temperature=0.1)
    image_b64 = image_to_base64(image_path)
    
    # Le prompt pour guider l'IA de vision
    prompt = "Tu es un analyste financier expert. Décris ce graphique ou tableau en détail, en incluant les chiffres clés, les tendances et les axes. Sois précis."
    
    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
        ]
    )
    
    try:
        response = llm.invoke([message])
        return response.content
    except Exception as e:
        print(f"   ⚠️ Erreur avec le modèle de vision : {e}")
        return "Image non analysée."

def extract_and_describe_images(pdf_path):
    """Extrait les images d'un PDF et génère des Documents LangChain avec leurs descriptions."""
    os.makedirs(IMG_TMP_PATH, exist_ok=True)
    pdf_document = fitz.open(pdf_path)
    image_documents = []
    
    print(f"🖼️ Recherche de graphiques dans {os.path.basename(pdf_path)}...")
    
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        image_list = page.get_images(full=True)
        
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image["image"]
            
            # Sauvegarde temporaire de l'image
            image_filename = f"{IMG_TMP_PATH}page{page_num+1}_img{img_index}.jpg"
            with open(image_filename, "wb") as f:
                f.write(image_bytes)
                
            # 1. On décrit l'image avec l'IA
            description = describe_image(image_filename)
            
            # 2. On transforme cette description en un Document compatible LangChain
            doc = Document(
                page_content=f"Ceci est la description d'un graphique (Page {page_num+1}) : \n{description}",
                metadata={"source": pdf_path, "page": page_num + 1, "type": "image_description"}
            )
            image_documents.append(doc)
            
    return image_documents

# ==========================================
# FONCTIONS DU PIPELINE
# ==========================================

def load_documents(directory_path: str):
    """Charge le texte classique ET ajoute les descriptions des images."""
    print(f"📄 Chargement des textes depuis '{directory_path}'...")
    
    # 1. Chargement du texte classique
    loader = DirectoryLoader(directory_path, glob="**/*.pdf", loader_cls=PyPDFLoader)
    text_documents = loader.load()
    print(f"✅ {len(text_documents)} pages de texte chargées.")
    
    # 2. Chargement et analyse des graphiques/images
    all_image_documents = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(directory_path, filename)
            img_docs = extract_and_describe_images(pdf_path)
            all_image_documents.extend(img_docs)
            
    print(f"✅ {len(all_image_documents)} graphiques/images analysés et convertis en texte.")
    
    # On fusionne le texte brut et les descriptions d'images !
    return text_documents + all_image_documents

def split_documents(documents):
    print("✂️ Découpage des documents en chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"✅ Documents découpés en {len(chunks)} chunks.")
    return chunks

def save_to_chroma(chunks):
    print(f"🧠 Initialisation de l'embedding : {EMBEDDING_MODEL}...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print(f"💾 Sauvegarde dans ChromaDB ('{CHROMA_PATH}')...")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )
    print("🚀 Pipeline d'ingestion Multimodal terminé avec succès !")

def lancer_ingestion(dossier_donnees):
    if not os.path.exists(dossier_donnees):
        print(f"❌ Erreur : Le dossier '{dossier_donnees}' n'existe pas.")
        return
        
    docs = load_documents(dossier_donnees)
    if len(docs) > 0:
        chunks = split_documents(docs)
        save_to_chroma(chunks)
    else:
        print(f"⚠️ Aucun PDF trouvé dans '{dossier_donnees}'.")

if __name__ == "__main__":
    lancer_ingestion(DATA_PATH)

"""
====================================================================
PIPELINE D'INGESTION RAG 
====================================================================
Ce script est une boîte à outils. Il ne se lance pas tout seul.

UTILISATION :

1. Placez tous vos documents financiers (.pdf) dans le dossier 'data_pdf/'.
2. Dans votre script principal, importez la fonction :
   from pipeline_pdf import lancer_ingestion
3. Lancez la fonction avec le chemin vers le dossier :
   lancer_ingestion("data_pdf/")
   
Le script s'occupera de tout découper et de peupler la base ChromaDB !
====================================================================
"""# 