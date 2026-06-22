ACTIONS = {
    "Apple":         "AAPL",
    "Tesla":         "TSLA",
    "Microsoft":     "MSFT",
    "Google":        "GOOGL",
    "Amazon":        "AMZN",
    "Nvidia":        "NVDA",
    "Meta":          "META",
    "Netflix":       "NFLX",
    "AMD":           "AMD",
    "Intel":         "INTC",
    "Airbus":        "AIR.PA",
    "TotalEnergies": "TTE.PA",
    "LVMH":          "MC.PA",
    "BNP Paribas":   "BNP.PA",
    "Sanofi":        "SAN.PA",
    "SAP":           "SAP.DE",
    "Siemens":       "SIE.DE",
    "BMW":           "BMW.DE",
    "Volkswagen":    "VOW3.DE",
    "Adidas":        "ADS.DE",
    "HSBC":          "HSBC",
    "BP":            "BP.L",
    "Shell":         "SHEL.L",
    "Unilever":      "ULVR.L",
    "AstraZeneca":   "AZN.L",
    "Toyota":        "7203.T",
    "Sony":          "6758.T",
    "SoftBank":      "9984.T",
    "Nintendo":      "7974.T",
    "Mitsubishi":    "8058.T",
    "Honda":         "7267.T",
}

INDICES = {
    "CAC40":     "^FCHI",
    "SP500":     "^GSPC",
    "NASDAQ":    "^IXIC",
    "DAX":       "^GDAXI",
    "FTSE100":   "^FTSE",
    "Nikkei225": "^N225",
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
    "Le Monde Éco":  "https://www.lemonde.fr/economie/rss_full.xml",
    "L'Express Éco": "https://www.lexpress.fr/arc/outboundfeeds/rss/alaune.xml",
    "MarketWatch":   "https://feeds.marketwatch.com/marketwatch/topstories/",
    "CNBC":          "https://www.cnbc.com/id/10000664/device/rss/rss.html",
    "FT Markets":    "https://www.ft.com/rss/home/uk",
    "Yahoo Finance": "https://finance.yahoo.com/news/rssindex",
    "Investing.com": "https://www.investing.com/rss/news.rss",
}

GOOGLE_NEWS_QUERIES = [
    "CAC40", "bourse finance", "Bitcoin crypto",
    "taux intérêt BCE", "inflation économie",
    "Wall Street NASDAQ", "matières premières",
    "actions européennes", "Fed taux directeur",
]

CHROMA_PATH     = "chroma_db/"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

PATHS = {
    "indices_clean":        "data/indices_clean",
    "actions_clean":        "data/actions_clean",
    "crypto_clean":         "data/crypto_clean",
    "forex_clean":          "data/forex_clean",
    "scoring":              "data/scoring",
    "logs":                 "logs",
    "analyse_technique":    "data/analyse_technique",
    "analyse_fondamentale": "data/analyse_fondamentale",
    "macro":                "data/macro",
    "chroma":               "chroma_db",
    "pdf":                  "Annual report",
    "images":               "pages_images",
}