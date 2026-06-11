import yfinance as yf
import pandas as pd
import os
from datetime import datetime
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import feedparser


os.makedirs("data/indices", exist_ok=True)
os.makedirs("data/actions", exist_ok=True)

os.makedirs("data/crypto", exist_ok=True)
os.makedirs("data/forex", exist_ok=True)

CHROMA_PATH = "chroma_db/"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

## Vadim

INDICES = {
    "CAC40":     "^FCHI",
    "SP500":    "^GSPC",
    "NASDAQ":    "^IXIC",
    "DAX":       "^GDAXI",
    "FTSE100":   "^FTSE",
    "Nikkei225": "^N225",
}



ACTIONS = {
    # US
    "Apple":        "AAPL",
    "Tesla":        "TSLA",
    "Microsoft":    "MSFT",
    "Google":       "GOOGL",
    "Amazon":       "AMZN",
    # NASDAQ
    "Nvidia":       "NVDA",
    "Meta":         "META",
    "Netflix":      "NFLX",
    "AMD":          "AMD",
    "Intel":        "INTC",
    # CAC40
    "Airbus":       "AIR.PA",
    "TotalEnergies":"TTE.PA",
    "LVMH":         "MC.PA",
    "BNP Paribas":  "BNP.PA",
    "Sanofi":       "SAN.PA",
    #DAX
    "SAP":          "SAP.DE",
    "Siemens":      "SIE.DE",
    "BMW":          "BMW.DE",
    "Volkswagen":   "VOW3.DE",
    "Adidas":       "ADS.DE",
    #FTSE100
    "HSBC":         "HSBA.L",
    "BP":           "BP.L",
    "Shell":        "SHEL.L",
    "Unilever":     "ULVR.L",
    "AstraZeneca":  "AZN.L",
    #Nikkei225
    "Toyota":       "7203.T",
    "Sony":         "6758.T",
    "SoftBank":     "9984.T",
    "Nintendo":     "7974.T",
    "Mitsubishi":   "8058.T",
    "Honda":        "7267.T",
}

CRYPTO = {
    "Bitcoin":  "BTC-USD",
    "Ethereum": "ETH-USD",
    "BNB":      "BNB-USD",
}

FOREX = {
    "EUR/USD": "EURUSD=X",
    "EUR/GBP": "EURGBP=X",
    "USD/JPY": "JPY=X",
}

FLUX_RSS = {
    # Médias français
    "Les Echos":     "https://www.lesechos.fr/rss/rss_finance.xml",
    "BFM Bourse":    "https://www.bfmtv.com/rss/bourse/",
    "Boursorama":    "https://www.boursorama.com/bourse/actualites/rss.phtml",
    
    # Médias internationaux
    "Reuters":       "https://feeds.reuters.com/reuters/businessNews",
    "MarketWatch":   "https://feeds.marketwatch.com/marketwatch/topstories/",
    "Yahoo Finance": "https://finance.yahoo.com/news/rssindex",
}

# Google News par mot-clé (sans clé API)
GOOGLE_NEWS_QUERIES = [
    "CAC40", "bourse finance", "Bitcoin crypto",
    "taux intérêt BCE", "inflation économie",
    "Wall Street NASDAQ", "matières premières"
]

# ============================================================
# RÉCUPÉRER LES INDICES
# ============================================================

def get_market_indices(indices: dict = INDICES, period: str = "2y") -> dict:
    
    print("RÉCUPÉRATION DES INDICES BOURSIERS")

    resultats = {}

    for nom, ticker in indices.items():
        try:
            indice = yf.Ticker(ticker)
            historique = indice.history(period=period)

            if historique.empty:
                print(f"{nom} ({ticker}) — Pas de données disponibles")
                continue

            # Garder uniquement les colonnes utiles
            colonnes = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in historique.columns]
            historique = historique[colonnes]

            print(f"\n{nom} ({ticker})")
            print(f"  Lignes récupérées : {len(historique)}")
            print(f"  Période : {historique.index[0].date()} → {historique.index[-1].date()}")
            print(historique.tail(3).to_string())

            # Sauvegarde CSV raw
            chemin = f"data/indices/{nom.lower()}.csv"
            historique.to_csv(chemin)
            print(f"  ✅ Sauvegardé : {chemin}")

            resultats[nom] = historique

        except Exception as e:
            print(f"❌ {nom} ({ticker}) — Erreur : {e}")

    print(f"\n✅ {len(resultats)}/{len(indices)} indices récupérés avec succès")
    return resultats


# ============================================
# RÉCUPÉRATION DES ACTIONS
# ============================================

def get_multiple_stocks(actions: dict = ACTIONS, period: str = "2y") -> dict:
    
    print(" RÉCUPÉRATION DES ACTIONS")


    resultats = {}

    for nom, ticker in actions.items():
        try:
            action = yf.Ticker(ticker)
            historique = action.history(period=period)

            if historique.empty:
                print(f"{nom} ({ticker}) — Pas de données disponibles")
                continue

            # Garder uniquement les colonnes utiles
            colonnes = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in historique.columns]
            historique = historique[colonnes]

            # Récupération des infos générales
            try:
                info = action.info
                print(f"\n{nom} ({ticker})")
                print(f"  Nom complet  : {info.get('longName', 'N/A')}")
                print(f"  Secteur      : {info.get('sector', 'N/A')}")
                print(f"  Industrie    : {info.get('industry', 'N/A')}")
                print(f"  Pays         : {info.get('country', 'N/A')}")
                print(f"  Prix actuel  : {info.get('currentPrice', 'N/A')}")
                print(f"  P/E Ratio    : {info.get('trailingPE', 'N/A')}")
                print(f"  Capitalisation : {info.get('marketCap', 'N/A')}")
            except Exception:
                print(f"\n{nom} ({ticker})")
                print(f" Infos générales non disponibles")

            print(f"  Lignes récupérées : {len(historique)}")
            print(f"  Période : {historique.index[0].date()} → {historique.index[-1].date()}")
            print(historique[["Close"]].tail(3).to_string())

            # Sauvegarde CSV raw
            nom_fichier = nom.lower().replace(" ", "_")
            chemin = f"data/actions/{nom_fichier}.csv"
            historique.to_csv(chemin)
            print(f" Sauvegardé : {chemin}")

            resultats[nom] = historique

        except Exception as e:
            print(f"❌ {nom} ({ticker}) — Erreur : {e}")

    print(f"\n{len(resultats)}/{len(actions)} actions récupérées avec succès")
    return resultats

def get_prix_actuel(ticker: str) -> dict:
    action = yf.Ticker(ticker)
    info = action.info
    return {
        "prix": info.get("currentPrice"),
        "variation": info.get("regularMarketChangePercent")
    }


def get_crypto_and_forex(
    crypto: dict = CRYPTO, 
    forex: dict = FOREX, 
    period: str = "2y"
) -> dict:
    
    os.makedirs("data/crypto", exist_ok=True)
    os.makedirs("data/forex", exist_ok=True)
    
    resultats = {}
    tous = {**crypto, **forex}
    
    for nom, ticker in tous.items():
        try:
            data = yf.Ticker(ticker)
            historique = data.history(period=period)
            
            if historique.empty:
                print(f"⚠️ {nom} — pas de données")
                continue
                
            colonnes = [c for c in ["Open","High","Low","Close","Volume"] 
                       if c in historique.columns]
            historique = historique[colonnes]
            
            # Dossier selon le type
            dossier = "data/crypto" if ticker in crypto.values() else "data/forex"
            nom_fichier = nom.lower().replace("/", "_")
            chemin = f"{dossier}/{nom_fichier}.csv"
            historique.to_csv(chemin)
            print(f"✅ {nom} sauvegardé : {chemin}")
            
            resultats[nom] = historique
            
        except Exception as e:
            print(f"❌ {nom} — Erreur : {e}")
    
    return resultats

def fetch_news(ticker: str, max_news: int = 5) -> list[dict]:
    try:
        t = yf.Ticker(ticker)
        news = t.news or []
        resultats = []
        for article in news[:max_news]:
            content = article.get("content", {})
            resultats.append({
                "titre":  content.get("title", "N/A"),
                "date":   content.get("pubDate", "N/A"),
                "url":    content.get("canonicalUrl", {}).get("url", "N/A"),
                "source": content.get("provider", {}).get("displayName", "N/A"),
            })
        return resultats
    except Exception as e:
        print(f"❌ Erreur news {ticker} : {e}")
        return []

def collecter_toutes_news(groupe: dict, max_news: int = 3) -> dict:
    toutes = {}
    for nom, ticker in groupe.items():
        news = fetch_news(ticker, max_news)
        if news:
            toutes[nom] = news
    return toutes


def build_google_news_rss(queries: list) -> dict:
    flux = {}
    for q in queries:
        nom = q.replace(" ", "_")
        url = f"https://news.google.com/rss/search?q={q.replace(' ', '+')}&hl=fr&gl=FR&ceid=FR:fr"
        flux[nom] = url
    return flux

def collecter_rss(flux: dict, max_articles: int = 10) -> list[dict]:
    """
    Collecte les articles depuis les flux RSS.
    """
    articles = []

    for source, url in flux.items():
        try:
            feed = feedparser.parse(url)

            if not feed.entries:
                print(f"⚠️ {source} — aucun article")
                continue

            for entry in feed.entries[:max_articles]:
                articles.append({
                    "titre":  entry.get("title", "N/A"),
                    "date":   entry.get("published", str(datetime.now())),
                    "resume": entry.get("summary", "N/A"),
                    "url":    entry.get("link", "N/A"),
                    "source": source
                })

            print(f"✅ {source} — {min(len(feed.entries), max_articles)} articles collectés")

        except Exception as e:
            print(f"❌ {source} — Erreur : {e}")

    print(f"\n📰 Total : {len(articles)} articles collectés")
    return articles

# ============================================
# NETTOYAGE DES DONNÉES
# ============================================

os.makedirs("data/indices_clean", exist_ok=True)
os.makedirs("data/actions_clean", exist_ok=True)

def nettoyer_fichier(chemin: str, nom: str) -> pd.DataFrame:
    df = pd.read_csv(chemin, index_col=0, parse_dates=True)

    print(f"\n{nom}")
    print(f"  Avant nettoyage : {df.shape[0]} lignes, {df.shape[1]} colonnes")

    # 1. Supprimer les doublons
    df = df[~df.index.duplicated(keep="first")]

    # 2. Supprimer les lignes entièrement vides
    df = df.dropna(how="all")

    # 3. Remplir les valeurs manquantes
    df = df.ffill().bfill()

    # 4. Garder uniquement les colonnes utiles
    colonnes_utiles = ["Open", "High", "Low", "Close", "Volume"]
    colonnes_presentes = [c for c in colonnes_utiles if c in df.columns]
    df = df[colonnes_presentes]

    # 5. Arrondir à 2 décimales
    df = df.round(2)

    # 6. Trier par date
    df = df.sort_index()

    print(f"  Après nettoyage : {df.shape[0]} lignes, {df.shape[1]} colonnes")
    print(f"  Valeurs nulles restantes : {df.isnull().sum().sum()}")

    return df


def nettoyer_tous():
    dossiers = {
        "data/indices": "data/indices_clean",
        "data/actions": "data/actions_clean",
        "data/crypto":  "data/crypto_clean",
        "data/forex":   "data/forex_clean",    
    }

    for dossier_src, dossier_dst in dossiers.items():
        os.makedirs(dossier_dst, exist_ok=True)

        for fichier in os.listdir(dossier_src):
            if fichier.endswith(".csv"):
                chemin_src = f"{dossier_src}/{fichier}"
                chemin_dst = f"{dossier_dst}/{fichier}"
                nom = fichier.replace(".csv", "")

                df_clean = nettoyer_fichier(chemin_src, nom)
                df_clean.to_csv(chemin_dst)
                print(f"  ✅ Sauvegardé : {chemin_dst}")

def indexer_csv_dans_chroma(dossier: str, type_contenu: str):
    """
    Lit les CSV nettoyés et les indexe dans ChromaDB.
    """
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_store = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )

    documents = []

    for fichier in os.listdir(dossier):
        if not fichier.endswith(".csv"):
            continue

        nom = fichier.replace(".csv", "")
        chemin = os.path.join(dossier, fichier)

        df = pd.read_csv(chemin, index_col=0)

        # Résumé global
        resume = (
            f"{nom} — données du {df.index[0]} au {df.index[-1]}.\n"
            f"Prix de clôture min : {df['Close'].min():.2f}, "
            f"max : {df['Close'].max():.2f}, "
            f"dernier : {df['Close'].iloc[-1]:.2f}.\n"
            f"Volume moyen : {df['Volume'].mean():.0f}."
        )

        documents.append(Document(
            page_content=resume,
            metadata={
                "source": fichier,
                "type": type_contenu,
                "nom": nom
            }
        ))

        # Indexer aussi les 5 dernières lignes (données récentes)
        for date, row in df.tail(5).iterrows():
            contenu = (
                f"{nom} le {date} : "
                f"Open={row.get('Open','N/A'):.2f}, "
                f"Close={row.get('Close','N/A'):.2f}, "
                f"High={row.get('High','N/A'):.2f}, "
                f"Low={row.get('Low','N/A'):.2f}, "
                f"Volume={row.get('Volume','N/A'):.0f}"
            )
            documents.append(Document(
                page_content=contenu,
                metadata={
                    "source": fichier,
                    "type": type_contenu,
                    "nom": nom,
                    "date": str(date)
                }
            ))

    if documents:
        vector_store.add_documents(documents)
        print(f"✅ {len(documents)} documents indexés depuis '{dossier}'")
    else:
        print(f"⚠️ Aucun CSV trouvé dans '{dossier}'")

def indexer_news_dans_chroma(news_dict: dict):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_store = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )
    documents = []
    for nom, articles in news_dict.items():
        for article in articles:
            contenu = f"{nom} — {article['date']} : {article['titre']} (source: {article['source']})"
            documents.append(Document(
                page_content=contenu,
                metadata={"source": article["url"], "type": "news", "nom": nom}
            ))
    if documents:
        vector_store.add_documents(documents)
        print(f"✅ {len(documents)} news indexées")

def indexer_articles_dans_chroma(articles: list[dict]):
    """
    Indexe les articles RSS dans ChromaDB.
    """
    if not articles:
        print("⚠️ Aucun article à indexer")
        return

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_store = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )

    documents = []

    for article in articles:
        contenu = (
            f"[{article['source']}] {article['date']}\n"
            f"Titre : {article['titre']}\n"
            f"Résumé : {article['resume']}"
        )

        documents.append(Document(
            page_content=contenu,
            metadata={
                "source": article["url"],
                "type":   "media_article",
                "nom":    article["source"],
                "date":   article["date"]
            }
        ))

    vector_store.add_documents(documents)
    print(f"✅ {len(documents)} articles indexés dans ChromaDB")

def indexer_toutes_les_donnees():
    indexer_csv_dans_chroma("data/indices_clean", "indice_boursier")
    indexer_csv_dans_chroma("data/actions_clean", "action")
    indexer_csv_dans_chroma("data/crypto_clean",  "crypto")
    indexer_csv_dans_chroma("data/forex_clean",   "forex") 


    
if __name__ == "__main__":
    # 1. Récupération des données
    get_market_indices()
    print("\n✅ Indices récupérés !")
    
    get_multiple_stocks()
    print("\n✅ Actions récupérées !")

    get_crypto_and_forex()        
    print("\n✅ Crypto & Forex récupérés !")
    
    # 2. Nettoyage
    nettoyer_tous()
    print("\n✅ Nettoyage terminé !")

    # 3. Indexation
    indexer_toutes_les_donnees()
    print("\n✅ Indexation financière terminée !")

    # 4. News yFinance
    print("\n📰 Collecte des news yFinance...")
    news = collecter_toutes_news(ACTIONS, max_news=5)
    indexer_news_dans_chroma(news)

    # 5. Médias RSS
    print("\n📡 Collecte des flux RSS médias...")
    
    # RSS directs
    articles_rss = collecter_rss(FLUX_RSS, max_articles=10)
    
    # Google News
    google_flux = build_google_news_rss(GOOGLE_NEWS_QUERIES)
    articles_google = collecter_rss(google_flux, max_articles=5)
    
    # Indexation de tous les articles
    tous_articles = articles_rss + articles_google
    indexer_articles_dans_chroma(tous_articles)
    print("\n✅ Médias indexés dans ChromaDB !")

    print("\n🎉 Pipeline complet terminé !")