import os
import subprocess
import sys
import threading

from pipeline_pdf import lancer_ingestion, DATA_PATH
from chloe import (
    get_market_indices,
    get_multiple_stocks,
    get_crypto_and_forex,
    nettoyer_tous,
    indexer_toutes_les_donnees,
    collecter_toutes_news,
    indexer_news_dans_chroma,
    collecter_rss,
    build_google_news_rss,
    indexer_articles_dans_chroma,
    ACTIONS,
    FLUX_RSS,
    GOOGLE_NEWS_QUERIES,
    PATHS,
)

# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────
RESET_PDF    = False  # True = réingère tous les PDFs depuis zéro
RESET_MARCHE = True   # True = re-collecte toutes les données marché
LANCER_APP   = True   # True = lance Streamlit à la fin

# ─────────────────────────────────────────
# ÉTAPE 1 — PIPELINE PDF
# ─────────────────────────────────────────
def etape_pdf():
    print("\n" + "="*60)
    print("📄 ÉTAPE 1 — Ingestion des rapports PDF")
    print("="*60)

    chroma_existe = os.path.exists("chroma_db") and os.listdir("chroma_db")

    if chroma_existe and not RESET_PDF:
        print("✅ Base ChromaDB déjà existante — ingestion PDF ignorée.")
        print("   (Mettez RESET_PDF=True pour forcer la réingestion)")
    else:
        print("🔄 Lancement de l'ingestion PDF...")
        lancer_ingestion(DATA_PATH, reset=RESET_PDF)
        print("✅ Ingestion PDF terminée !")

# ─────────────────────────────────────────
# ÉTAPE 2 — COLLECTE DONNÉES MARCHÉ
# ─────────────────────────────────────────
def etape_marche():
    print("\n" + "="*60)
    print("📊 ÉTAPE 2 — Collecte des données marché")
    print("="*60)

    print("\n📥 Téléchargement des données...")
    get_market_indices()
    get_multiple_stocks()
    get_crypto_and_forex()

    print("\n🧹 Nettoyage des données...")
    nettoyer_tous()

    print("\n🗄️ Indexation dans ChromaDB...")
    indexer_toutes_les_donnees()

    print("\n📰 Collecte des news yFinance...")
    news = collecter_toutes_news(ACTIONS, max_news=5)
    indexer_news_dans_chroma(news)

    print("\n📡 Collecte des flux RSS + Google News...")
    tous_articles = (
        collecter_rss(FLUX_RSS, max_articles=10)
        + collecter_rss(build_google_news_rss(GOOGLE_NEWS_QUERIES), max_articles=5)
    )
    indexer_articles_dans_chroma(tous_articles)

    print("\n✅ Collecte marché terminée !")

# ─────────────────────────────────────────
# ÉTAPE 3 — ANALYSES
# ─────────────────────────────────────────
def etape_analyses():
    print("\n" + "="*60)
    print("🔬 ÉTAPE 3 — Analyses techniques & fondamentales")
    print("="*60)

    try:
        from analyse_tech import pipeline_technique
        from analyse_fond import pipeline_fondamental

        pipeline_technique(PATHS["actions_clean"])
        pipeline_fondamental(list(ACTIONS.values()))
        print("✅ Analyses terminées !")
    except Exception as e:
        print(f"⚠️ Erreur analyses : {e}")
        print("   L'application sera quand même lancée.")

# ─────────────────────────────────────────
# ÉTAPE 4 — SCHEDULER EN ARRIÈRE-PLAN
# ─────────────────────────────────────────
def lancer_scheduler_background():
    print("\n" + "="*60)
    print("⏱️  ÉTAPE 4 — Scheduler en arrière-plan")
    print("="*60)

    import schedule
    import time
    from scheduler import configurer_planning

    def boucle():
        configurer_planning()
        print("✅ Scheduler actif (thread daemon)")
        while True:
            schedule.run_pending()
            time.sleep(60)

    thread = threading.Thread(target=boucle, daemon=True)
    thread.start()
    print("🔄 Scheduler démarré en arrière-plan")

# ─────────────────────────────────────────
# ÉTAPE 5 — LANCEMENT STREAMLIT
# ─────────────────────────────────────────
def etape_streamlit():
    print("\n" + "="*60)
    print("🚀 ÉTAPE 5 — Lancement de l'interface FinAI")
    print("="*60)
    print("🌐 Ouverture sur http://localhost:8501\n")

    subprocess.run([
        sys.executable, "-m", "streamlit", "run", "app.py",
        "--server.port", "8501",
        "--server.headless", "false",
        "--browser.gatherUsageStats", "false"
    ])

# ─────────────────────────────────────────
# POINT D'ENTRÉE
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🤖 FinAI — Démarrage du pipeline complet")
    print("="*60)

    try:
        etape_pdf()
        etape_marche()
        etape_analyses()
        lancer_scheduler_background()
    except KeyboardInterrupt:
        print("\n⛔ Interruption manuelle.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Erreur critique : {e}")
        sys.exit(1)

    if LANCER_APP:
        etape_streamlit()