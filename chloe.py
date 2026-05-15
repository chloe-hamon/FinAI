import yfinance as yf
import pandas as pd
import os

# ============================================
# 1. RÉCUPÉRER UN INDICE (ex: CAC40)
# ============================================

os.makedirs("data/indices", exist_ok=True)
os.makedirs("data/actions", exist_ok=True)

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


# ============================================
# 1. RÉCUPÉRATION DES INDICES
# ============================================

print("INDICES BOURSIERS")

for nom, ticker in INDICES.items():
    indice = yf.Ticker(ticker)
    historique = indice.history(period="2y")
    
    print(f"\n{nom} ({ticker}) - Historique 2 ans :")
    print(historique[["Open", "High", "Low", "Close", "Volume"]].head())
    
    chemin = f"data/indices/{nom.lower()}.csv"
    historique.to_csv(chemin)
    print(f"✅ Sauvegardé : {chemin}")


# ============================================
# 2. RÉCUPÉRATION DES ACTIONS
# ============================================

print("ACTIONS")

for nom, ticker in ACTIONS.items():
    action = yf.Ticker(ticker)
    info = action.info

    print(f"\n=== {nom} ({ticker}) ===")
    print(f"  Nom complet : {info.get('longName')}")
    print(f"  Secteur     : {info.get('sector')}")
    print(f"  Prix actuel : {info.get('currentPrice')} $")
    print(f"  P/E Ratio   : {info.get('trailingPE')}")

    historique = action.history(period="2y")

    print(f"  Historique (5 derniers jours) :")
    print(historique[["Close"]].tail(5))

    nom_fichier = nom.lower().replace(' ', '_')
    chemin = f"data/actions/{nom_fichier}.csv"
    historique.to_csv(chemin)
    print(f"✅ Sauvegardé : {chemin}")


# ============================================
# 3. COMPARAISON MULTI-TICKERS
# ============================================

print("COMPARAISON - Cours de clôture (5 derniers jours)")

all_tickers = list(INDICES.values()) + list(ACTIONS.values())
data = yf.download(all_tickers, period="1mo")["Close"]
print(data.tail(5))