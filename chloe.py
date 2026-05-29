import yfinance as yf
import pandas as pd
import os


os.makedirs("data/indices", exist_ok=True)
os.makedirs("data/actions", exist_ok=True)

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
