import schedule
import time
import logging
import zoneinfo   
from datetime import datetime
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
    GOOGLE_NEWS_QUERIES
)

os.makedirs("logs", exist_ok=True)

# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/scheduler.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# ============================================================
# TÂCHES
# ============================================================

def tache_cours():
    """Mise à jour des cours — toutes les heures en journée"""
    log.info("📊 Mise à jour des cours...")
    try:
        get_market_indices()
        get_multiple_stocks()
        get_crypto_and_forex()
        nettoyer_tous()
        indexer_toutes_les_donnees()
        log.info("✅ Cours mis à jour")
    except Exception as e:
        log.error(f"❌ Erreur cours : {e}")

def tache_news():
    """Mise à jour des news — toutes les 30 minutes"""
    log.info("📰 Mise à jour des news...")
    try:
        # yFinance news
        news = collecter_toutes_news(ACTIONS, max_news=5)
        indexer_news_dans_chroma(news)

        # RSS + Google News
        articles_rss = collecter_rss(FLUX_RSS, max_articles=10)
        google_flux = build_google_news_rss(GOOGLE_NEWS_QUERIES)
        articles_google = collecter_rss(google_flux, max_articles=5)
        indexer_articles_dans_chroma(articles_rss + articles_google)

        log.info("✅ News mises à jour")
    except Exception as e:
        log.error(f"❌ Erreur news : {e}")

def tache_nuit():
    """Pipeline complet — chaque nuit à 2h"""
    log.info("🌙 Pipeline nuit complet...")
    try:
        tache_cours()
        tache_news()
        log.info("✅ Pipeline nuit terminé")
    except Exception as e:
        log.error(f"❌ Erreur pipeline nuit : {e}")

# ============================================================
# EST-CE QUE LES MARCHÉS SONT OUVERTS ?
# ============================================================

def marches_ouverts() -> bool:
    """
    Vérifie si on est en semaine entre 8h et 22h (CET)
    Couvre NYSE, NASDAQ, Euronext, LSE, Tokyo
    """
    now = datetime.now()
    # 0=lundi ... 4=vendredi
    if now.weekday() > 4:
        return False
    if now.hour < 8 or now.hour >= 22:
        return False
    return True

def tache_cours_conditionnelle():
    if marches_ouverts():
        tache_cours()
    else:
        log.info("💤 Marchés fermés — cours ignorés")

# ============================================================
# PLANNING
# ============================================================

def configurer_planning():
    # Cours — toutes les heures si marchés ouverts
    schedule.every(1).hours.do(tache_cours_conditionnelle)

    # News — toutes les 30 minutes
    schedule.every(30).minutes.do(tache_news)

    # Pipeline complet — chaque nuit à 2h00
    schedule.every().day.at("02:00").do(tache_nuit)

    log.info("📅 Planning configuré :")
    log.info("  • Cours    → toutes les heures (si marchés ouverts)")
    log.info("  • News     → toutes les 30 minutes")
    log.info("  • Nuit     → chaque jour à 02:00")

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    log.info("🚀 Démarrage du scheduler")

    # Pipeline initial au démarrage
    log.info("⚡ Pipeline initial...")
    tache_cours()
    tache_news()

    # Configurer le planning
    configurer_planning()

    # Boucle infinie
    log.info("🔄 Boucle de surveillance active...")
    while True:
        schedule.run_pending()
        time.sleep(60)  # vérifie chaque minute