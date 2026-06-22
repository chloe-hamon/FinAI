import re
import os
from datetime import datetime
from config import (
    ACTIONS, INDICES, CRYPTO, FOREX,
    FLUX_RSS, GOOGLE_NEWS_QUERIES,
    CHROMA_PATH, EMBEDDING_MODEL, PATHS
)
import feedparser
import pandas as pd
import yfinance as yf
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

# ============================================================
# DOSSIERS
# ============================================================

for _d in [
    "data/indices", "data/actions", "data/crypto", "data/forex",
    "data/indices_clean", "data/actions_clean",
    "data/crypto_clean", "data/forex_clean",
    "data/analyse_technique",    
    "data/analyse_fondamentale", 
    "data/scoring",              
    "logs",
    "data/macro",                      
]:
    os.makedirs(_d, exist_ok=True)


# ============================================================
# HELPER CHROMADB
# ============================================================

def _get_vector_store() -> Chroma:
    """Instancie (ou ouvre) le vector store Chroma — appel unique."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)


# ============================================================
# RÉCUPÉRATION — INDICES
# ============================================================

def get_market_indices(indices: dict = INDICES, period: str = "2y") -> dict:
    print("\n" + "=" * 60)
    print("RÉCUPÉRATION DES INDICES BOURSIERS")
    print("=" * 60)
    resultats = {}

    for nom, ticker in indices.items():
        try:
            hist = yf.Ticker(ticker).history(period=period)
            if hist.empty:
                print(f"  ⚠️ {nom} — pas de données")
                continue

            cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in hist.columns]
            hist = hist[cols]

            chemin = f"data/indices/{nom.lower()}.csv"
            hist.to_csv(chemin)
            print(f"  ✅ {nom} ({ticker}) — {len(hist)} lignes → {chemin}")
            resultats[nom] = hist

        except Exception as e:
            print(f"  ❌ {nom} ({ticker}) — {e}")

    print(f"\n  {len(resultats)}/{len(indices)} indices récupérés")
    return resultats


# ============================================================
# RÉCUPÉRATION — ACTIONS
# ============================================================

def get_multiple_stocks(actions: dict = ACTIONS, period: str = "2y") -> dict:
    print("\n" + "=" * 60)
    print("RÉCUPÉRATION DES ACTIONS")
    print("=" * 60)
    resultats = {}

    for nom, ticker in actions.items():
        try:
            t    = yf.Ticker(ticker)
            hist = t.history(period=period)

            if hist.empty:
                print(f"  ⚠️ {nom} — pas de données")
                continue

            cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in hist.columns]
            hist = hist[cols]

            try:
                info = t.info
                print(f"\n  {nom} ({ticker})")
                print(f"    Secteur : {info.get('sector', 'N/A')} | "
                      f"Prix : {info.get('currentPrice', 'N/A')} | "
                      f"PE : {info.get('trailingPE', 'N/A')}")
            except Exception:
                print(f"\n  {nom} ({ticker}) — infos non disponibles")

            nom_f  = nom.lower().replace(" ", "_")
            chemin = f"data/actions/{nom_f}.csv"
            hist.to_csv(chemin)
            print(f"    ✅ {len(hist)} lignes → {chemin}")
            resultats[nom] = hist

        except Exception as e:
            print(f"  ❌ {nom} ({ticker}) — {e}")

    print(f"\n  {len(resultats)}/{len(actions)} actions récupérées")
    return resultats


# ============================================================
# RÉCUPÉRATION — CRYPTO & FOREX
# ============================================================

def get_crypto_and_forex(
    crypto: dict = CRYPTO,
    forex:  dict = FOREX,
    period: str  = "2y",
) -> dict:
    resultats  = {}
    crypto_set = set(crypto.values())

    for nom, ticker in {**crypto, **forex}.items():
        try:
            hist = yf.Ticker(ticker).history(period=period)
            if hist.empty:
                print(f"  ⚠️ {nom} — pas de données")
                continue

            cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in hist.columns]
            hist = hist[cols]

            dossier    = "data/crypto" if ticker in crypto_set else "data/forex"
            nom_f      = nom.lower().replace("/", "_")
            chemin     = f"{dossier}/{nom_f}.csv"
            hist.to_csv(chemin)
            print(f"  ✅ {nom} → {chemin}")
            resultats[nom] = hist

        except Exception as e:
            print(f"  ❌ {nom} — {e}")

    return resultats


# ============================================================
# PRIX ACTUEL
# ============================================================

def get_prix_actuel(ticker: str) -> dict:
    """Retourne le prix et la variation du jour."""  
    try:
        info = yf.Ticker(ticker).info
        return {
            "prix":      info.get("currentPrice"),
            "variation": info.get("regularMarketChangePercent"),
        }
    except Exception as e:
        return {"ticker": ticker, "erreur": str(e)}


# ============================================================
# NEWS yFINANCE
# ============================================================

def fetch_news(ticker: str, max_news: int = 5) -> list[dict]:
    try:
        news = yf.Ticker(ticker).news or []
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
        print(f"  ❌ News {ticker} : {e}")
        return []


def collecter_toutes_news(groupe: dict, max_news: int = 3) -> dict:
    toutes = {}
    for nom, ticker in groupe.items():
        news = fetch_news(ticker, max_news)
        if news:
            toutes[nom] = news
    return toutes


# ============================================================
# FLUX RSS
# ============================================================

def _strip_html(text: str) -> str:
    """Supprime les balises HTML d'un texte."""  
    return re.sub(r"<[^>]+>", "", text or "").strip()


def build_google_news_rss(queries: list) -> dict:
    return {
        q.replace(" ", "_"): (
            f"https://news.google.com/rss/search"
            f"?q={q.replace(' ', '+')}&hl=fr&gl=FR&ceid=FR:fr"
        )
        for q in queries
    }


def collecter_rss(flux: dict, max_articles: int = 10) -> list[dict]:
    articles = []

    for source, url in flux.items():
        try:
            feed = feedparser.parse(url)
            if not feed.entries:
                print(f"  ⚠️ {source} — aucun article")
                continue

            for entry in feed.entries[:max_articles]:
                articles.append({
                    "titre":  entry.get("title", "N/A"),
                    "date":   entry.get("published", str(datetime.now())),
                    "resume": _strip_html(entry.get("summary", "")),  
                    "url":    entry.get("link", "N/A"),
                    "source": source,
                })

            print(f"  ✅ {source} — {min(len(feed.entries), max_articles)} articles")

        except Exception as e:
            print(f"  ❌ {source} — {e}")

    print(f"\n  📰 Total : {len(articles)} articles collectés")
    return articles


# ============================================================
# NETTOYAGE
# ============================================================

def nettoyer_fichier(chemin: str, nom: str) -> pd.DataFrame:
    df = pd.read_csv(chemin, index_col=0, parse_dates=True)

    avant = df.shape[0]
    df = (df
          .pipe(lambda d: d[~d.index.duplicated(keep="first")])
          .dropna(how="all")
          .ffill()
          .bfill())

    cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df   = df[cols].round(2).sort_index()

    print(f"  {nom:25} {avant} → {df.shape[0]} lignes | nulls={df.isnull().sum().sum()}")
    return df


def nettoyer_tous():
    print("\n" + "=" * 60)
    print("NETTOYAGE DES DONNÉES")
    print("=" * 60)

    dossiers = {
        "data/indices": "data/indices_clean",
        "data/actions": "data/actions_clean",
        "data/crypto":  "data/crypto_clean",
        "data/forex":   "data/forex_clean",
    }

    for src, dst in dossiers.items():
        os.makedirs(dst, exist_ok=True)

        
        if not os.path.isdir(src):
            print(f"  ⚠️ Dossier absent : {src}")
            continue

        for fichier in os.listdir(src):
            if not fichier.endswith(".csv"):
                continue
            nom = fichier.replace(".csv", "")
            try:
                df = nettoyer_fichier(f"{src}/{fichier}", nom)
                df.to_csv(f"{dst}/{fichier}")
            except Exception as e:
                print(f"  ❌ {nom} — {e}")


# ============================================================
# INDEXATION CHROMA
# ============================================================

def indexer_csv_dans_chroma(dossier: str, type_contenu: str):
    """Indexe les CSV nettoyés dans ChromaDB."""
    if not os.path.isdir(dossier):
        print(f"  ⚠️ Dossier absent : {dossier}")
        return

    vs        = _get_vector_store()   
    documents = []

    for fichier in os.listdir(dossier):
        if not fichier.endswith(".csv"):
            continue

        nom = fichier.replace(".csv", "")
        df  = pd.read_csv(os.path.join(dossier, fichier), index_col=0)

        if df.empty or "Close" not in df.columns:
            continue

        
        has_vol = "Volume" in df.columns
        resume  = (
            f"{nom} — données du {df.index[0]} au {df.index[-1]}.\n"
            f"Clôture min={df['Close'].min():.2f}, "
            f"max={df['Close'].max():.2f}, "
            f"dernier={df['Close'].iloc[-1]:.2f}."
            + (f"\nVolume moyen={df['Volume'].mean():.0f}." if has_vol else "")
        )
        documents.append(Document(
            page_content=resume,
            metadata={"source": fichier, "type": type_contenu, "nom": nom},
        ))

        # 5 dernières lignes
        for date, row in df.tail(5).iterrows():
            vol_str = f"{row['Volume']:.0f}" if has_vol and pd.notna(row.get("Volume")) else "N/A"
            contenu = (
                f"{nom} le {date} : "
                f"Open={row.get('Open', 0):.2f}, "
                f"Close={row.get('Close', 0):.2f}, "
                f"High={row.get('High', 0):.2f}, "
                f"Low={row.get('Low', 0):.2f}, "
                f"Volume={vol_str}"
            )
            documents.append(Document(
                page_content=contenu,
                metadata={"source": fichier, "type": type_contenu, "nom": nom, "date": str(date)},
            ))

    if documents:
        vs.add_documents(documents)
        print(f"  ✅ {len(documents)} documents indexés depuis '{dossier}'")
    else:
        print(f"  ⚠️ Aucun document valide dans '{dossier}'")


def indexer_news_dans_chroma(news_dict: dict):
    """Indexe les news yFinance dans ChromaDB."""
    vs        = _get_vector_store()
    documents = [
        Document(
            page_content=f"{nom} — {a['date']} : {a['titre']} (source: {a['source']})",
            metadata={"source": a["url"], "type": "news", "nom": nom},
        )
        for nom, articles in news_dict.items()
        for a in articles
    ]
    if documents:
        vs.add_documents(documents)
        print(f"  ✅ {len(documents)} news indexées")


def indexer_articles_dans_chroma(articles: list[dict]):
    """Indexe les articles RSS dans ChromaDB."""
    if not articles:
        print("  ⚠️ Aucun article à indexer")
        return

    vs        = _get_vector_store()
    documents = [
        Document(
            page_content=(
                f"[{a['source']}] {a['date']}\n"
                f"Titre : {a['titre']}\n"
                f"Résumé : {a['resume']}"
            ),
            metadata={"source": a["url"], "type": "media_article",
                      "nom": a["source"], "date": a["date"]},
        )
        for a in articles
    ]
    vs.add_documents(documents)
    print(f"  ✅ {len(documents)} articles indexés")


def indexer_toutes_les_donnees():
    indexer_csv_dans_chroma("data/indices_clean", "indice_boursier")
    indexer_csv_dans_chroma("data/actions_clean", "action")
    indexer_csv_dans_chroma("data/crypto_clean",  "crypto")
    indexer_csv_dans_chroma("data/forex_clean",   "forex")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    get_market_indices()
    get_multiple_stocks()
    get_crypto_and_forex()

    nettoyer_tous()
    print("\n✅ Nettoyage terminé !")

    indexer_toutes_les_donnees()
    print("\n✅ Indexation financière terminée !")

    print("\n📰 Collecte des news yFinance...")
    news = collecter_toutes_news(ACTIONS, max_news=5)
    indexer_news_dans_chroma(news)

    print("\n📡 Collecte des flux RSS...")
    tous_articles = (
        collecter_rss(FLUX_RSS, max_articles=10)
        + collecter_rss(build_google_news_rss(GOOGLE_NEWS_QUERIES), max_articles=5)
    )
    indexer_articles_dans_chroma(tous_articles)
    print("\n✅ Médias indexés !")

    print("\n🎉 Pipeline complet terminé !")
