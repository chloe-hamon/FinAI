import yfinance as yf
import pandas as pd
import os
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


os.makedirs("data/indices", exist_ok=True)
os.makedirs("data/actions", exist_ok=True)


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


def indexer_toutes_les_donnees():
    indexer_csv_dans_chroma("data/indices_clean", "indice_boursier")
    indexer_csv_dans_chroma("data/actions_clean", "action")

    
if __name__ == "__main__":
    nettoyer_tous()
    print("\n✅ Nettoyage terminé !")
    
    indexer_toutes_les_donnees()
    print("\n✅ Indexation ChromaDB terminée !")