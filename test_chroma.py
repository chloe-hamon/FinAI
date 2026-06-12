from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

CHROMA_PATH = "chroma_db/"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ==========================================
# CONNEXION
# ==========================================
print("🔌 Connexion à ChromaDB...")
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

# ← CORRIGÉ : API publique au lieu de _collection.count()
total = db._collection.count()
print(f"📦 Documents en base : {total}")

if total == 0:
    print("❌ La base est VIDE — Relancer pipeline_pdf.py")
    exit()

# ==========================================
# TEST 1 : Aperçu des sources disponibles
# ==========================================
print("\n" + "=" * 50)
print("TEST 1 — Sources disponibles dans la base")
print("=" * 50)

# ← CORRIGÉ : récupération via API publique
tous = db.get()  # Chroma LangChain expose .get() directement
sources = set()

for metadata in tous["metadatas"]:
    source = metadata.get("source", "inconnue")
    sources.add(source)

print(f"📁 {len(sources)} source(s) distincte(s) :")
for s in sorted(sources):
    print(f"   - {s}")

# ==========================================
# TEST 2 : Types de contenu
# ==========================================
print("\n" + "=" * 50)
print("TEST 2 — Types de contenu")
print("=" * 50)

types = {}
for metadata in tous["metadatas"]:
    t = metadata.get("type", "texte")
    types[t] = types.get(t, 0) + 1

for type_contenu, nb in types.items():
    print(f"   {type_contenu} : {nb} documents")

# ==========================================
# TEST 3 : Recherches financières types
# ==========================================
print("\n" + "=" * 50)
print("TEST 3 — Recherches financières types")
print("=" * 50)

requetes_test = [
    "Apple annual revenue 2024",
    "Tesla net income",
    "Microsoft cloud growth",
    "exchange rate EUR USD",
    "CAC40 performance",
]

for requete in requetes_test:
    print(f"\n🔍 Requête : '{requete}'")
    results = db.similarity_search(requete, k=2)

    if not results:
        print("   ⚠️ Aucun résultat")
        continue

    for j, doc in enumerate(results):
        print(f"   Résultat {j + 1} :")
        print(f"   Source : {doc.metadata.get('source', 'N/A')} "
              f"| Page : {doc.metadata.get('page', 'N/A')} "
              f"| Type : {doc.metadata.get('type', 'texte')}")
        print(f"   Extrait : {doc.page_content[:150]}...")

# ==========================================
# TEST 4 : Score de similarité
# ==========================================
print("\n" + "=" * 50)
print("TEST 4 — Score de similarité (pertinence)")
print("=" * 50)

requete_score = "revenue net income profit"
results_scores = db.similarity_search_with_score(requete_score, k=3)

print(f"Requête : '{requete_score}'")
print("(Score : plus il est BAS, plus c'est pertinent avec ChromaDB)\n")

for doc, score in results_scores:
    if score < 0.3:
        pertinence = "🟢 Très pertinent"
    elif score < 0.6:
        pertinence = "🟡 Pertinent"
    else:
        pertinence = "🔴 Peu pertinent"

    print(f"   {pertinence} | Score : {score:.4f}")
    print(f"   Source : {doc.metadata.get('source', 'N/A')}")
    print(f"   Extrait : {doc.page_content[:120]}...")
    print()

# ==========================================
# BILAN FINAL
# ==========================================
print("=" * 50)
print("BILAN")
print("=" * 50)
print(f"✅ Base ChromaDB opérationnelle")
print(f"   Documents totaux   : {total}")
print(f"   Sources distinctes : {len(sources)}")
print(f"   Types de contenu   : {types}")

if total < 100:
    print("\n⚠️ Moins de 100 documents — base incomplète")
    print("   → Vérifier que pipeline_pdf.py a bien tourné sur tous les PDFs")
elif total < 500:
    print("\n🟡 Base correcte mais limitée")
    print("   → Ajouter plus de rapports PDF si possible")
else:
    print("\n🟢 Base bien peuplée, prête pour la Phase 3 !")