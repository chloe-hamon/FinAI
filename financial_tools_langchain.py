from langchain.tools import tool
from vadim import ask, retriever, charger_vector_store, construire_contexte
import re
import os


vector_store = charger_vector_store()


@tool
def calcul_marge_nette(input: str) -> str:
    """Calcule la marge nette.
        UTILISE SEULEMENT APRÈS avoir trouvé les chiffres avec recherche_financiere.
        NE PAS inventer de chiffres hypothétiques.
        Format STRICT : resultat_net, chiffre_affaires
        Exemple : 3200, 45000"""
    try:
        input_clean = input.strip().strip("'\"")
        parties = [p.strip() for p in input_clean.split(",")]
        resultat_net = float(parties[0])
        chiffre_affaires = float(parties[1])
    except:
        return "❌ Format attendu : resultat_net, chiffre_affaires — ex: 500, 2000"

    if chiffre_affaires == 0:
        return "Erreur : chiffre d'affaires ne peut pas être zéro."

    marge = round((resultat_net / chiffre_affaires) * 100, 2)

    if marge < 0:
        niveau = "🔴 Négative — entreprise en perte"
    elif marge < 5:
        niveau = "🟡 Faible"
    elif marge < 15:
        niveau = "✅ Correcte"
    else:
        niveau = "✅ Excellente"

    return f"Marge nette : {marge}% → {niveau}"


@tool
def calcul_croissance(input: str) -> str:
    """
    Calcule le taux de croissance en pourcentage entre deux périodes.
    Utilise cet outil pour comparer des revenus, bénéfices ou tout indicateur financier.
    Input : deux nombres séparés par une virgule. Ex: 120000, 100000 (valeur_actuelle, valeur_precedente)
    """
    try:
        input_clean = input.strip().strip("'\"")
        parties = [p.strip() for p in input_clean.split(",")]
        valeur_actuelle = float(parties[0])
        valeur_precedente = float(parties[1])
    except:
        return "❌ Format attendu : valeur_actuelle, valeur_precedente — ex: 120000, 100000"

    if valeur_precedente == 0:
        return "Erreur : valeur précédente ne peut pas être zéro."

    croissance = round(((valeur_actuelle - valeur_precedente) / valeur_precedente) * 100, 2)
    tendance = "📈" if croissance > 0 else "📉"
    return f"Croissance : {tendance} {croissance}%"


@tool
def calcul_ratio_endettement(input: str) -> str:
    """
    Calcule le ratio d'endettement (dette / capitaux propres).
    Utilise cet outil pour évaluer le niveau de risque financier d'une entreprise.
    Un ratio > 2 est généralement considéré comme élevé.
    Input : deux nombres séparés par une virgule. Ex: 50000, 100000 (dette_totale, capitaux_propres)
    """
    try:
        input_clean = input.strip().strip("'\"")
        parties = [p.strip() for p in input_clean.split(",")]
        dette_totale = float(parties[0])
        capitaux_propres = float(parties[1])
    except:
        return "❌ Format attendu : dette_totale, capitaux_propres — ex: 50000, 100000"

    if capitaux_propres == 0:
        return "Erreur : capitaux propres ne peuvent pas être zéro."

    ratio = round(dette_totale / capitaux_propres, 2)

    if ratio < 1:
        niveau = "✅ Faible — entreprise peu endettée"
    elif ratio < 2:
        niveau = "🟡 Modéré — acceptable"
    else:
        niveau = "🔴 Élevé — risque financier important"

    return f"Ratio d'endettement : {ratio} → {niveau}"


@tool
def verifier_alerte(input: str) -> str:
    """
    Vérifie si une valeur financière est en dehors des seuils d'alerte.
    Utilise cet outil pour surveiller un indicateur et déclencher une alerte si nécessaire.
    Input : valeur, seuil_bas, seuil_haut, nom séparés par une virgule. Ex: 4.5, 3.0, 5.0, RSI
    """
    try:
        input_clean = input.strip().strip("'\"")
        parties = [p.strip() for p in input_clean.split(",")]
        valeur = float(parties[0])
        seuil_bas = float(parties[1])
        seuil_haut = float(parties[2])
        nom = parties[3] if len(parties) > 3 else "Indicateur"
    except:
        return "❌ Format attendu : valeur, seuil_bas, seuil_haut, nom — ex: 4.5, 3.0, 5.0, RSI"

    if valeur < seuil_bas:
        return f"🚨 ALERTE : {nom} = {valeur} est SOUS le seuil bas ({seuil_bas})"
    elif valeur > seuil_haut:
        return f"🚨 ALERTE : {nom} = {valeur} est AU-DESSUS du seuil haut ({seuil_haut})"
    else:
        return f"✅ {nom} = {valeur} dans la zone normale [{seuil_bas} — {seuil_haut}]"

TICKERS = { 
    # Indices
    "CAC40":     "^FCHI",
    "SP500":     "^GSPC",
    "S&P500":    "^GSPC",
    "NASDAQ":    "^IXIC",
    "DAX":       "^GDAXI",
    "FTSE100":   "^FTSE",
    "Nikkei225": "^N225",
    # Crypto
    "Bitcoin":   "BTC-USD",
    "Ethereum":  "ETH-USD",
    "BNB":       "BNB-USD",
    # Forex
    "EUR/USD":   "EURUSD=X",
    "EUR/GBP":   "EURGBP=X",
    "USD/JPY":   "JPY=X",

    "Apple":         "AAPL",
    "Tesla":         "TSLA",
    "Microsoft":     "MSFT",
    "Google":        "GOOGL",
    "Amazon":        "AMZN",
    # NASDAQ
    "Nvidia":        "NVDA",
    "Meta":          "META",
    "Netflix":       "NFLX",
    "AMD":           "AMD",
    "Intel":         "INTC",
    # CAC40
    "Airbus":        "AIR.PA",
    "TotalEnergies": "TTE.PA",
    "LVMH":          "MC.PA",
    "BNP Paribas":   "BNP.PA",
    "Sanofi":        "SAN.PA",
    # DAX
    "SAP":           "SAP.DE",
    "Siemens":       "SIE.DE",
    "BMW":           "BMW.DE",
    "Volkswagen":    "VOW3.DE",
    "Adidas":        "ADS.DE",
    # FTSE100
    "HSBC":          "HSBC",
    "BP":            "BP.L",
    "Shell":         "SHEL.L",
    "Unilever":      "ULVR.L",
    "AstraZeneca":   "AZN.L",
    # Nikkei225
    "Toyota":        "7203.T",
    "Sony":          "6758.T",
    "SoftBank":      "9984.T",
    "Nintendo":      "7974.T",
    "Mitsubishi":    "8058.T",
    "Honda":         "7267.T",
}
ENTREPRISES_CONNUES = {
    "apple":          "Apple",
    "tesla":          "Tesla",
    "microsoft":      "Microsoft",
    "google":         "Google",
    "alphabet":       "Google",
    "amazon":         "Amazon",
    "nvidia":         "Nvidia",
    "meta":           "Meta",
    "facebook":       "Meta",
    "netflix":        "Netflix",
    "intel":          "Intel",
    "amd":            "AMD",
    "airbus":         "Airbus",
    "totalenergies":  "TotalEnergies",
    "total":          "TotalEnergies",
    "lvmh":           "LVMH",
    "bnp paribas":    "BNP Paribas",
    "bnp":            "BNP Paribas",
    "sanofi":         "Sanofi",
    "sap":            "SAP",
    "siemens":        "Siemens",
    "bmw":            "BMW",
    "volkswagen":     "Volkswagen",
    "vw":             "Volkswagen",
    "adidas":         "Adidas",
    "hsbc":           "HSBC",
    "bp":             "BP",
    "shell":          "Shell",
    "unilever":       "Unilever",
    "astrazeneca":    "AstraZeneca",
    "toyota":         "Toyota",
    "sony":           "Sony",
    "softbank":       "SoftBank",
    "nintendo":       "Nintendo",
    "mitsubishi":     "Mitsubishi",
    "honda":          "Honda",
}

TRADUCTIONS = {
    "chiffre d'affaires": "total revenue",
    "chiffre d affaires": "total revenue",
    "bénéfice net":       "net income",
    "benefice net":       "net income",
    "résultat net":       "net income",
    "resultat net":       "net income",
    "revenus totaux":     "total revenue",
    "revenus":            "revenue",
    "marge nette":        "net margin",
    "marge":              "margin",
    "dette":              "debt",
    "capitaux propres":   "shareholders equity",
    "trésorerie":         "cash",
    "tresorerie":         "cash",
    "croissance":         "growth",
    "bénéfice":           "income",
    "benefice":           "income",
    "charges":            "expenses",
    "coût":               "cost",
    "cout":               "cost",
    "dividende":          "dividend",
    "rachat d'actions":   "share buyback",
    "endettement":        "debt ratio",
    "résultat opérationnel": "operating income",
    "resultat operationnel": "operating income",
}

def traduire_requete(question: str) -> str:
    q = question.lower()
    for fr, en in sorted(TRADUCTIONS.items(), key=lambda x: len(x[0]), reverse=True):
        q = q.replace(fr, en)
    return q


def detecter_entreprise(question: str) -> str | None:
    """Détecte le nom canonique de l'entreprise dans la question."""
    question_lower = question.lower()
    # Trier par longueur décroissante → "bnp paribas" matché avant "bnp"
    for variante in sorted(ENTREPRISES_CONNUES, key=len, reverse=True):
        if variante in question_lower:
            entreprise = ENTREPRISES_CONNUES[variante]
            print(f"   🏢 Entreprise détectée : '{entreprise}'")
            return entreprise
    print("   🏢 Aucune entreprise détectée — recherche globale")
    return None

def detecter_ticker(question: str) -> str | None:
    """
    Détecte le ticker boursier depuis la question.
    Utilisé pour enrichir l'input de get_cours_action.
    """
    question_lower = question.lower()
    for nom, ticker in sorted(TICKERS.items(), key=lambda x: len(x[0]), reverse=True):
        if nom.lower() in question_lower:
            return ticker
    return None


@tool
def get_donnees_financieres(ticker: str) -> str:
    """Récupère les données financières d'une entreprise via yfinance. 
    Input : ticker boursier uniquement (ex: AAPL, TSLA, NVDA)."""
    import yfinance as yf
    try:
        ticker = ticker.strip()
        ticker = ticker.replace("ticker=", "").replace("TICKER=", "")
        ticker = ticker.split(",")[0].split("=")[-1].strip()
        ticker = ticker.upper()

        action = yf.Ticker(ticker)
        info = action.info

        revenus = info.get("totalRevenue")
        benefice = info.get("netIncomeToCommon")
        marge = info.get("profitMargins")

        revenus_str = f"{revenus:,} USD" if revenus is not None else "N/A"
        benefice_str = f"{benefice:,} USD" if benefice is not None else "N/A"
        marge_str = f"{marge:.2%}" if marge is not None else "N/A"

        return f"""
{ticker} - Données financières :
  Revenus totaux : {revenus_str}
  Bénéfice net   : {benefice_str}
  Marge nette    : {marge_str}
        """.strip()

    except Exception as e:
        return f"❌ Erreur pour '{ticker}' : {e}"



@tool
def recherche_financiere(question: str) -> str:
    """Recherche des informations financières dans les documents.
    UTILISE CET OUTIL EN PREMIER pour trouver les chiffres bruts.
    Input : une question en langage naturel."""

    entreprise = detecter_entreprise(question)
    requete_recherche = traduire_requete(question)
    print(f"   🔍 Requête traduite : '{requete_recherche}'")

    resultats = vector_store.similarity_search_with_score(requete_recherche, k=20)

    # ── 1. Filtre entreprise ──────────────────────────────────────
    if entreprise:
        resultats = [
            (doc, score) for doc, score in resultats
            if entreprise.lower() in doc.metadata.get("fichier", "").lower()
            or entreprise.lower() in doc.metadata.get("source", "").lower()
        ]

    print(f"\n   → {len(resultats)} chunks"
          + (f" (filtrés sur '{entreprise}')" if entreprise else " (global)") + " :")
    for doc, score in resultats:
        source = os.path.basename(doc.metadata.get("source", "inconnue"))
        page   = doc.metadata.get("page", "?")
        print(f"     Score: {score:.4f} | {source} p.{page}")

    # ── 2. Filtre score ───────────────────────────────────────────
    SEUIL_SCORE = 0.85
    resultats = [(doc, score) for doc, score in resultats if score < SEUIL_SCORE]

    # ── 3. Filtre disclaimers ─────────────────────────────────────
    MOTS_EXCLUS = [
        "éléments de projection",
        "facteurs d'incertitudes",
        "aucunement à mettre à jour",
        "obligations légales"
    ]
    resultats = [
        (doc, score) for doc, score in resultats
        if not any(mot in doc.page_content for mot in MOTS_EXCLUS)
    ]

    # ── 4. Déduplication ─────────────────────────────────────────
    vus = set()
    resultats_dedup = []
    for doc, score in resultats:
        cle = (doc.page_content[:100], doc.metadata.get("page", ""))
        if cle not in vus:
            vus.add(cle)
            resultats_dedup.append((doc, score))
    resultats = resultats_dedup

    print(f"   → {len(resultats)} chunks après filtrage (seuil={SEUIL_SCORE})")

    # ── 5. Aucun résultat ─────────────────────────────────────────
    if not resultats:
        if entreprise:
            return (
                f"❌ Aucun document trouvé pour '{entreprise}'. "
                f"Donnez une Final Answer indiquant que les données "
                f"ne sont pas disponibles dans la base."
            )
        return "❌ Aucun document pertinent trouvé dans la base de données."

    # ── 6. Construction réponse ───────────────────────────────────
    blocs = []
    for i, (doc, score) in enumerate(resultats[:5]):
        source         = doc.metadata.get("source", "inconnue")
        page           = doc.metadata.get("page", "?")
        entreprise_doc = doc.metadata.get("entreprise", "?")
        blocs.append(
            f"[Source {i+1} : {entreprise_doc} | "
            f"{os.path.basename(source)}, p.{page} | score={score:.4f}]\n"
            f"{doc.page_content.strip()}"
        )

    return "\n\n".join(blocs)


@tool
def get_cours_action(ticker: str) -> str:
    """Récupère le cours actuel d'une action ou indice. UN SEUL ticker par appel."""
    import yfinance as yf
    try:
        ticker = re.sub(r"['\"]", "", ticker).strip().split()[0].upper()
        base = ticker.split(".")[0]  # HSBC.L → HSBC
        
        for t in [base]:
            action = yf.Ticker(t)
            info = action.info
            prix = info.get("currentPrice") or info.get("regularMarketPrice")
            if prix:
                devise = info.get("currency", "USD")
                return f"{t} : {prix:.2f} {devise}"
        
        return f"❌ Aucun cours trouvé pour {ticker}."
    except Exception as e:
        return f"❌ Impossible de récupérer {ticker} : {e}"

@tool
def get_historique_action(input: str) -> str:
    """Récupère le cours historique d'une action ou indice.
    Input format STRICT: TICKER PERIODE — ex: ^FCHI 1mo, AAPL 6mo, ^GSPC 1y
    UN SEUL ticker par appel. Périodes: 5d, 1wk, 1mo, 3mo, 6mo, 1y"""
    import yfinance as yf
    ticker = "inconnu"
    try:
        # Nettoyer l'input : supprimer guillemets et tout ce qui suit le 2e mot
        input = input.strip().strip("'\"")
        
        # Extraire uniquement TICKER et PERIODE avec regex
        match = re.match(r"([A-Z0-9\^\.\-=]+)\s+(5d|1wk|1mo|3mo|6mo|1y|ytd|2y|5y)", input, re.IGNORECASE)
        if not match:
            return "❌ Format invalide. Utilise: TICKER PERIODE (ex: ^FCHI 1mo, AAPL 3mo)"
        
        ticker = match.group(1).upper()
        periode = match.group(2).lower()

        action = yf.Ticker(ticker)
        hist = action.history(period=periode)

        if hist.empty:
            return f"❌ Aucune donnée historique pour {ticker}."

        cours_debut = hist["Close"].iloc[0]
        cours_fin = hist["Close"].iloc[-1]
        variation = ((cours_fin - cours_debut) / cours_debut) * 100
        signe = "📈" if variation > 0 else "📉"

        return (
            f"{ticker} sur {periode} :\n"
            f"  Début : {cours_debut:.2f}\n"
            f"  Actuel : {cours_fin:.2f}\n"
            f"  Variation : {signe} {variation:+.2f}%"
        )
    except Exception as e:
        return f"❌ Erreur pour {ticker} : {e}"

@tool
def get_pe_ratio(ticker: str) -> str:
    """Récupère le P/E ratio d'UNE SEULE action. UN ticker par appel.
    Input: UN ticker ex: MC.PA ou SAN.PA (pas deux à la fois)"""
    import yfinance as yf
    import re
    # Prendre uniquement le premier ticker
    ticker = re.sub(r"['\"]", "", ticker).strip().split()[0].rstrip(",").upper()
    try:
        info = yf.Ticker(ticker).info
        pe = info.get("trailingPE") or info.get("forwardPE")
        nom = info.get("shortName", ticker)
        if pe is None:
            return f"❌ P/E ratio non disponible pour {ticker}."
        return f"{nom} ({ticker}) — P/E ratio : {pe:.2f}"
    except Exception as e:
        return f"❌ Erreur pour {ticker} : {e}"

@tool  
def get_top_performers(indice: str) -> str:
    """Récupère les meilleures performances hebdomadaires des actions d'un indice.
    Input: nom de l'indice parmi CAC40, DAX, FTSE100, NASDAQ, Nikkei225"""
    import yfinance as yf

    INDICE_ACTIONS = {
        "CAC40":    {"Airbus": "AIR.PA", "TotalEnergies": "TTE.PA", "LVMH": "MC.PA", "BNP Paribas": "BNP.PA", "Sanofi": "SAN.PA"},
        "DAX":      {"SAP": "SAP.DE", "Siemens": "SIE.DE", "BMW": "BMW.DE", "Volkswagen": "VOW3.DE", "Adidas": "ADS.DE"},
        "FTSE100":  {"HSBC": "HSBC", "BP": "BP.L", "Shell": "SHEL.L", "Unilever": "ULVR.L", "AstraZeneca": "AZN.L"},
        "NASDAQ":   {"Nvidia": "NVDA", "Meta": "META", "Netflix": "NFLX", "AMD": "AMD", "Intel": "INTC"},
        "Nikkei225":{"Toyota": "7203.T", "Sony": "6758.T", "SoftBank": "9984.T", "Nintendo": "7974.T", "Honda": "7267.T"},
    }

    indice = indice.strip().upper()
    actions = None
    for key in INDICE_ACTIONS:
        if key.upper() in indice:
            actions = INDICE_ACTIONS[key]
            break

    if not actions:
        return f"❌ Indice non reconnu. Disponibles : {', '.join(INDICE_ACTIONS.keys())}"

    resultats = []
    for nom, ticker in actions.items():
        try:
            hist = yf.Ticker(ticker).history(period="5d")
            if hist.empty:
                continue
            debut = hist["Close"].iloc[0]
            fin = hist["Close"].iloc[-1]
            variation = ((fin - debut) / debut) * 100
            resultats.append((nom, ticker, variation))
        except:
            continue

    if not resultats:
        return "❌ Impossible de récupérer les données."

    resultats.sort(key=lambda x: x[2], reverse=True)
    lignes = [f"  {'📈' if v > 0 else '📉'} {n} ({t}) : {v:+.2f}%" for n, t, v in resultats]
    return f"Top performers {indice} (semaine) :\n" + "\n".join(lignes)

ALIAS_TICKERS = {
    "NASDAQ": "^IXIC",
    "SP500":  "^GSPC",
    "S&P500": "^GSPC",
    "CAC40":  "^FCHI",
    "DAX":    "^GDAXI",
    "FTSE100":"^FTSE",
    "NIKKEI": "^N225",
}

@tool
def get_correlation(input: str) -> str:
    """Calcule la corrélation entre deux actifs sur une période.
    Input format: 'TICKER1 TICKER2 PERIODE' ex: 'TTE.PA CL=F 3mo', 'BTC-USD ^IXIC 3mo'
    Pétrole Brent=BZ=F, WTI=CL=F, NASDAQ=^IXIC, SP500=^GSPC"""
    import yfinance as yf
    import pandas as pd
    
    input = input.strip().strip("'\"")
    parts = input.split()
    if len(parts) < 2:
        return "❌ Format: TICKER1 TICKER2 PERIODE (ex: TTE.PA CL=F 3mo)"
    
    ticker1 = ALIAS_TICKERS.get(parts[0].upper(), parts[0].upper())
    ticker2 = ALIAS_TICKERS.get(parts[1].upper(), parts[1].upper())
    periode = parts[2] if len(parts) > 2 else "3mo"

    try:
        h1 = yf.Ticker(ticker1).history(period=periode)["Close"]
        h2 = yf.Ticker(ticker2).history(period=periode)["Close"]
        
        if h1.empty:
            return f"❌ Aucune donnée pour {ticker1}"
        if h2.empty:
            return f"❌ Aucune donnée pour {ticker2}"

        # Aligner les index (fuseau horaire différent possible)
        h1.index = pd.to_datetime(h1.index).tz_localize(None).normalize()
        h2.index = pd.to_datetime(h2.index).tz_localize(None).normalize()
        
        df = h1.to_frame("a").join(h2.to_frame("b"), how="inner")
        
        if len(df) < 5:
            return f"❌ Pas assez de données communes entre {ticker1} et {ticker2}."
        
        corr = df["a"].corr(df["b"])

        if corr > 0.7:
            interpretation = "forte corrélation positive 📈📈"
        elif corr > 0.3:
            interpretation = "corrélation modérée positive"
        elif corr > -0.3:
            interpretation = "faible corrélation"
        elif corr > -0.7:
            interpretation = "corrélation modérée négative"
        else:
            interpretation = "forte corrélation négative 📉📉"

        return (
            f"Corrélation {ticker1} / {ticker2} sur {periode} :\n"
            f"  Coefficient : {corr:.3f}\n"
            f"  Interprétation : {interpretation}"
        )
    except Exception as e:
        return f"❌ Erreur : {e}"

@tool
def analyser_action(ticker: str) -> str:
    """Analyse rapide d'une action : cours actuel, performance 1 mois, P/E ratio.
    Input: ticker ex: 7203.T, AAPL, BTC-USD"""
    import yfinance as yf
    import re
    ticker = re.sub(r"['\"]", "", ticker).strip().split()[0].upper()
    try:
        t = yf.Ticker(ticker)
        prix = t.fast_info.last_price
        devise = getattr(t.fast_info, "currency", "USD")
        hist = t.history(period="1mo")
        pe = t.info.get("trailingPE")

        if hist.empty or prix is None:
            return f"❌ Données non disponibles pour {ticker}."

        debut = hist["Close"].iloc[0]
        variation = ((prix - debut) / debut) * 100
        signe = "📈" if variation > 0 else "📉"

        result = (
            f"{ticker} — Analyse :\n"
            f"  Cours actuel : {prix:.2f} {devise}\n"
            f"  Performance 1 mois : {signe} {variation:+.2f}%\n"
        )
        if pe:
            result += f"  P/E ratio : {pe:.2f}\n"

        return result
    except Exception as e:
        return f"❌ Erreur pour {ticker} : {e}"

if __name__ == "__main__":
    print("=== Tests financial_tools_langchain ===\n")
    print(calcul_marge_nette.invoke("12000, 100000"))
    print(calcul_croissance.invoke("120000, 100000"))
    print(calcul_ratio_endettement.invoke("50000, 100000"))
    print(verifier_alerte.invoke("4.5, 3.0, 5.0, RSI"))