import os
import shutil
import gc
import fitz
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from extraction_texte_image import lancer_extraction

# ==========================================
# CONFIGURATION
# ==========================================
DATA_PATH = "Annual report"
CHROMA_PATH = "chroma_db/" 
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ==========================================
# MOTS-CLÉS FINANCIERS
# ==========================================
FINANCIAL_KEYWORDS = [
    "revenue", "net income", "operating income", "ebitda", "gross profit",
    "cash flow", "free cash flow", "capital expenditure", "capex",
    "balance sheet", "total assets", "total liabilities", "equity",
    "earnings per share", "eps", "dividend", "guidance", "outlook",
    "key metrics", "highlights", "financial summary", "income statement",
    "return on equity", "roe", "debt", "liquidity",
    "chiffre d'affaires", "résultat net", "résultat opérationnel",
    "bilan", "trésorerie", "flux de trésorerie", "capitaux propres",
    "perspectives", "endettement", "marge", "résumé financier",
    "compte de résultat", "actif", "passif"
]

# ==========================================
# DÉTECTION
# ==========================================

def a_keywords_financiers(texte: str, seuil: int = 2) -> bool:
    texte_lower = texte.lower()
    return sum(1 for kw in FINANCIAL_KEYWORDS if kw in texte_lower) >= seuil


def a_des_images(page_fitz) -> bool:
    return len(page_fitz.get_images()) > 0


# ==========================================
# CHARGEMENT + FILTRAGE UNIFIÉ
# ==========================================

def load_documents(directory_path: str, seuil_keywords: int = 2):
    print(f"📄 Chargement des PDF depuis '{directory_path}'...")

    pdf_files = sorted([f for f in os.listdir(directory_path) if f.endswith(".pdf")])
    if not pdf_files:
        print(f"⚠️ Aucun PDF trouvé dans '{directory_path}'.")
        return [], {}

    pages_texte   = []
    pages_par_pdf = {}
    total_files   = len(pdf_files)

    for idx, fichier in enumerate(pdf_files, 1):
        path = os.path.join(directory_path, fichier)
        print(f"   [{idx:02d}/{total_files}] {fichier}...")

        # ── Extraction du nom d'entreprise depuis le nom de fichier
        nom_entreprise = fichier.replace(".pdf", "") \
                                .replace("_rapportannuel", "") \
                                .replace("_annualreport", "") \
                                .replace("_rapport_annuel", "") \
                                .replace("_annual_report", "") \
                                .strip("_") \
                                .strip()

        try:
            loader  = PyPDFLoader(path)
            docs    = loader.load()
            doc_fitz = fitz.open(path)

            index_visuels = set()
            n_texte       = 0
            n_image_only  = 0
            n_ignores     = 0

            for i, doc in enumerate(docs):
                texte      = doc.page_content
                page_fitz  = doc_fitz.load_page(i)

                est_financiere    = a_keywords_financiers(texte, seuil=seuil_keywords)
                a_texte_suffisant = len(texte.strip()) >= 100
                est_visuelle      = a_des_images(page_fitz)

                # ── Métadonnées enrichies sur chaque page
                doc.metadata["fichier"]     = fichier
                doc.metadata["entreprise"]  = nom_entreprise
                doc.metadata["page_index"]  = i          # index 0-based
                doc.metadata["page"]        = i + 1      # numéro lisible

                if a_texte_suffisant and est_financiere:
                    doc.metadata["type"] = "financial_key_page"
                    pages_texte.append(doc)
                    n_texte += 1
                    if est_visuelle:
                        index_visuels.add(i)

                elif not a_texte_suffisant:
                    index_visuels.add(i)
                    n_image_only += 1

                elif est_visuelle and not est_financiere:
                    index_visuels.add(i)
                    n_image_only += 1

                else:
                    n_ignores += 1  # ← compter les pages vraiment ignorées

            doc_fitz.close()

            pages_par_pdf[fichier] = sorted(index_visuels)
            print(f"      → {n_texte} pages texte | "
                  f"{n_image_only} pages image only | "
                  f"{len(index_visuels)} pages visuelles | "
                  f"{n_ignores} ignorées")

        except Exception as e:
            print(f"   ❌ Erreur sur {fichier} : {e} — ignoré")
            pages_par_pdf[fichier] = []  # ← éviter KeyError plus tard

    print(f"\n✅ {len(pages_texte)} pages texte | "
          f"{sum(len(v) for v in pages_par_pdf.values())} pages visuelles")

    return pages_texte, pages_par_pdf


# ==========================================
# DÉCOUPAGE
# ==========================================

def evaluer_qualite(texte: str) -> str:
    texte = texte.strip()
    longueur = len(texte)
    
    if longueur == 0:
        return "nulle"
    
    # Garder les chunks courts MAIS financiers
    if longueur < 100:
        return "nulle"
    elif longueur < 300:
        # Sauver si contenu financier
        if a_keywords_financiers(texte, seuil=1):
            return "moyenne"
        return "faible"
    elif longueur < 1000:
        return "moyenne"
    else:
        return "bonne"


def split_documents(documents):
    print("✂️ Découpage des documents en chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True,
    )
    chunks = splitter.split_documents(documents)
    print(f"   → {len(chunks)} chunks bruts générés")

    chunks_filtres = []
    ignores = 0
    for chunk in chunks:
        qualite = evaluer_qualite(chunk.page_content)
        if qualite in ("nulle", "faible"):
            ignores += 1
        else:
            chunk.metadata["qualite"] = qualite
            chunks_filtres.append(chunk)

    print(f"   → {ignores} chunks ignorés (qualité nulle/faible)")
    print(f"✅ {len(chunks_filtres)} chunks conservés.")
    return chunks_filtres


# ==========================================
# SAUVEGARDE CHROMA
# ==========================================

def save_to_chroma(chunks, reset: bool = False) -> Chroma:
    print(f"🧠 Initialisation embedding : {EMBEDDING_MODEL}...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"batch_size": 512}
    )

    if reset and os.path.exists(CHROMA_PATH):
        print("   🗑️ Suppression ancienne base...")
        gc.collect()
        time.sleep(1)
        shutil.rmtree(CHROMA_PATH, ignore_errors=True)

    print(f"💾 Création ChromaDB dans '{CHROMA_PATH}'...")

    BATCH_SIZE = 500
    vector_store = None

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        print(f"   → Batch {i // BATCH_SIZE + 1} : {len(batch)} chunks...")

        if vector_store is None:
            vector_store = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                persist_directory=CHROMA_PATH,
                collection_metadata={"hnsw:space": "cosine"}  
            )
        else:
            vector_store.add_documents(batch)

    total = vector_store._collection.count()
    print(f"✅ {total} chunks stockés dans ChromaDB.")
    return vector_store


# ==========================================
# ANALYSE VISUELLE
# ==========================================

def ajouter_extractions_visuelles(
    dossier: str,
    vector_store: Chroma,
    pages_par_pdf: dict
):
    print("\n📸 Lancement des analyses visuelles...")

    existing_ids = set(vector_store._collection.get()["ids"])
    total_ajoutes = 0

    for fichier, index_pages in pages_par_pdf.items():
        if not index_pages:
            print(f"   ⏭️ {fichier} — aucune page visuelle")
            continue

        pdf_path = os.path.join(dossier, fichier)
        try:
            docs_visuels = lancer_extraction(pdf_path, index_pages, existing_ids, max_workers=6)
            if docs_visuels:
                # Dédoublonnage par chunk_id
                ids = [doc.metadata["chunk_id"] for doc in docs_visuels]
                vector_store.add_documents(docs_visuels, ids=ids)
                total_ajoutes += len(docs_visuels)
                print(f"   ✅ {fichier} — {len(docs_visuels)} analyses ajoutées")
            else:
                print(f"   ⚠️ {fichier} — aucun élément visuel extrait")

        except Exception as e:
            print(f"   ❌ Erreur visuelle {fichier} : {e} — ignoré")

    print(f"\n✅ {total_ajoutes} analyses visuelles ajoutées au total.")


# ==========================================
# PIPELINE PRINCIPAL
# ==========================================

def lancer_ingestion(dossier_donnees: str, reset: bool = False, seuil_keywords: int = 2):
    if not os.path.exists(dossier_donnees):
        print(f"❌ Dossier introuvable : '{dossier_donnees}'")
        return

    print(f"⚙️  Seuil filtrage : {seuil_keywords} mots-clés minimum\n")

    pages_texte, pages_par_pdf = load_documents(dossier_donnees, seuil_keywords)
    if not pages_texte and not any(pages_par_pdf.values()):
        print("❌ Rien à indexer.")
        return

    chunks = split_documents(pages_texte)
    vector_store = save_to_chroma(chunks, reset=reset)
    ajouter_extractions_visuelles(dossier_donnees, vector_store, pages_par_pdf)

    final = vector_store._collection.count()
    print(f"\n🚀 Pipeline terminé — {final} chunks total dans ChromaDB.")


if __name__ == "__main__":
    lancer_ingestion(DATA_PATH, reset=True, seuil_keywords=2)