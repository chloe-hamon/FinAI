import requests
import json
import os
from datetime import datetime

os.makedirs("data/macro", exist_ok=True)

# ============================================================
# SOURCES MACRO GRATUITES
# ============================================================

# FRED (Federal Reserve Economic Data)
FRED_API_KEY = "5b957579a12195f4a9ca59ab0a27d8c5"  # https://fred.stlouisfed.org/docs/api/api_key.html
FRED_BASE    = "https://api.stlouisfed.org/fred/series/observations"

# Indicateurs FRED à suivre
FRED_SERIES = {
    # Inflation
    "CPI":          "CPIAUCSL",        # Inflation US (CPI)
    "PCE":          "PCEPI",           # PCE inflation (cible Fed)
    "PPI":          "PPIACO",          # Prix producteurs

    # Emploi
    "Chomage_US":   "UNRATE",          # Taux chômage US
    "NFP":          "PAYEMS",          # Non-Farm Payrolls
    "Chomage_Init": "ICSA",            # Inscriptions chômage hebdo

    # Croissance
    "PIB_US":       "GDP",             # PIB US
    "PIB_Growth":   "A191RL1Q225SBEA", # Croissance PIB US (%)

    # Taux
    "Fed_Funds":    "FEDFUNDS",        # Taux directeur Fed
    "T10Y":         "DGS10",           # Taux 10 ans US
    "T2Y":          "DGS2",            # Taux 2 ans US
    "T10Y2Y":       "T10Y2Y",          # Spread 10Y-2Y (courbe inversée)

    # Immobilier
    "Immo_Ventes":  "HSN1F",           # Ventes maisons neuves US
    "Immo_Prix":    "CSUSHPISA",       # Case-Shiller prix immobilier

    # Confiance
    "Confiance_Conso": "UMCSENT",      # Confiance consommateurs Michigan
    "ISM_Manuf":    "MANEMP",          # ISM Manufacturier

    # Monétaire
    "M2":           "M2SL",            # Masse monétaire M2
    "Credit":       "TOTALSL",         # Crédit consommateurs
}


# ============================================================
# RÉCUPÉRATION FRED
# ============================================================

def get_fred_serie(serie_id: str, nom: str, nb_obs: int = 12) -> dict:
    """
    Récupère les dernières observations d'une série FRED
    """
    params = {
        "series_id":     serie_id,
        "api_key":       FRED_API_KEY,
        "file_type":     "json",
        "sort_order":    "desc",
        "limit":         nb_obs
    }

    try:
        r    = requests.get(FRED_BASE, params=params, timeout=10)
        data = r.json()

        observations = data.get("observations", [])
        if not observations:
            return {"nom": nom, "serie": serie_id, "erreur": "Pas de données"}

        derniere = next((o for o in observations if o["value"] != "."), None)
        avant_derniere = next(
            (o for o in observations if o["value"] != "." and o != derniere), None
        )

        valeur     = float(derniere["value"])     if derniere      else None
        date       = derniere["date"]             if derniere      else None
        precedente = float(avant_derniere["value"]) if avant_derniere else None
        variation  = round(valeur - precedente, 4) if valeur and precedente else None

        return {
            "nom":        nom,
            "serie":      serie_id,
            "date":       date,
            "valeur":     valeur,
            "precedente": precedente,
            "variation":  variation
        }

    except Exception as e:
        return {"nom": nom, "serie": serie_id, "erreur": str(e)}


def get_toutes_series_fred() -> list[dict]:
    """
    Récupère tous les indicateurs FRED
    """
    resultats = []
    print("📡 Récupération des données FRED...")

    for nom, serie_id in FRED_SERIES.items():
        r = get_fred_serie(serie_id, nom)
        resultats.append(r)

        if "erreur" not in r:
            variation_str = f"({r['variation']:+.2f})" if r["variation"] is not None else ""
            print(f"  ✅ {nom:20} → {r['valeur']} {variation_str} [{r['date']}]")
        else:
            print(f"  ❌ {nom:20} → {r['erreur']}")

    return resultats


# ============================================================
# DONNÉES ALTERNATIVES GRATUITES (sans API key)
# ============================================================

def get_fear_greed() -> dict:
    """
    Fear & Greed Index (CNN) — sentiment de marché
    0-25 : Extreme Fear
    25-45 : Fear
    45-55 : Neutral
    55-75 : Greed
    75-100 : Extreme Greed
    """
    try:
        url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
        headers = {"User-Agent": "Mozilla/5.0"}
        r    = requests.get(url, headers=headers, timeout=10)
        data = r.json()

        score = data["fear_and_greed"]["score"]
        rating = data["fear_and_greed"]["rating"]

        return {
            "nom":    "Fear & Greed Index",
            "valeur": round(score, 1),
            "rating": rating,
            "date":   datetime.now().strftime("%Y-%m-%d")
        }
    except Exception as e:
        return {"nom": "Fear & Greed Index", "erreur": str(e)}


def get_vix() -> dict:
    """
    VIX via yFinance — indice de volatilité
    < 15 : calme
    15-25 : normal
    > 25 : stress
    > 35 : panique
    """
    try:
        import yfinance as yf
        vix  = yf.Ticker("^VIX")
        hist = vix.history(period="5d")

        if hist.empty:
            return {"nom": "VIX", "erreur": "Pas de données"}

        derniere = hist["Close"].iloc[-1]
        avant    = hist["Close"].iloc[-2] if len(hist) > 1 else None
        variation = round(derniere - avant, 2) if avant else None

        return {
            "nom":       "VIX",
            "valeur":    round(float(derniere), 2),
            "variation": variation,
            "date":      str(hist.index[-1].date()),
            "signal":    interpreter_vix(float(derniere))
        }
    except Exception as e:
        return {"nom": "VIX", "erreur": str(e)}


def interpreter_vix(valeur: float) -> str:
    if valeur < 15:
        return "😌 Marché calme"
    elif valeur < 25:
        return "😐 Volatilité normale"
    elif valeur < 35:
        return "😰 Stress de marché"
    else:
        return "😱 Panique — volatilité extrême"


def get_courbe_taux(fred_data: list[dict]=None) -> dict:
    """
    Analyse de la courbe des taux (2Y vs 10Y)
    Courbe inversée → signal récession historique
    """
    if fred_data:
        macro = {d["nom"]: d for d in fred_data if "erreur" not in d}
        t2y_data  = macro.get("T2Y")
        t10y_data = macro.get("T10Y")
        spread_data = macro.get("T10Y2Y")

        if t2y_data and t10y_data:
            taux_2y  = t2y_data["valeur"]
            taux_10y = t10y_data["valeur"]
            spread   = round(taux_10y - taux_2y, 3)
            return {
                "nom":      "Courbe_Taux",
                "taux_2y":  taux_2y,
                "taux_10y": taux_10y,
                "spread":   spread,
                "signal":   _signal_courbe(spread),
                "source":   "FRED",
                "date":     t10y_data["date"],
            }

    # Fallback yFinance
    try:
        import yfinance as yf
        # ← CORRIGÉ : ^FVX (5 ans) remplace ^IRX (3 mois)
        t5y  = yf.Ticker("^FVX")
        t10y = yf.Ticker("^TNX")

        h5  = t5y.history(period="5d")
        h10 = t10y.history(period="5d")

        if h5.empty or h10.empty:
            return {"nom": "Courbe_Taux", "erreur": "Données manquantes"}

        taux_5y  = round(float(h5["Close"].iloc[-1]), 3)
        taux_10y = round(float(h10["Close"].iloc[-1]), 3)
        spread   = round(taux_10y - taux_5y, 3)

        return {
            "nom":      "Courbe_Taux",
            "taux_2y":  taux_5y,    # proxy 5Y
            "taux_10y": taux_10y,
            "spread":   spread,
            "signal":   _signal_courbe(spread),
            "source":   "yFinance (proxy 5Y)",
            "date":     str(h10.index[-1].date()),
        }
    except Exception as e:
        return {"nom": "Courbe_Taux", "erreur": str(e)}

def _signal_courbe(spread: float) -> str:
    if spread > 1:    return "✅ Courbe normale — économie saine"
    if spread > 0:    return "⚠️ Courbe plate — ralentissement possible"
    if spread > -0.5: return "🔴 Courbe légèrement inversée — attention"
    return "🚨 Courbe très inversée — signal récession"


# ============================================================
# SCORING MACRO
# ============================================================

def scorer_macro(fred_data: list[dict], vix: dict,
                 fear_greed: dict, courbe: dict) -> dict:
    """
    Score macro global — contexte favorable ou défavorable pour investir
    """
    score   = 0
    details = {}

    # Dictionnaire par nom pour accès facile
    macro = {d["nom"]: d for d in fred_data if "erreur" not in d}

    # --- Inflation ---
    cpi = macro.get("CPI")
    if cpi and cpi["variation"] is not None:
        if cpi["variation"] < 0:
            score += 2; details["CPI"] = f"✅ Inflation en baisse ({cpi['variation']:+.2f})"
        elif cpi["variation"] < 0.3:
            score += 1; details["CPI"] = f"🟡 Inflation stable ({cpi['variation']:+.2f})"
        else:
            score -= 2; details["CPI"] = f"🔴 Inflation en hausse ({cpi['variation']:+.2f})"

    # --- Emploi ---
    chomage = macro.get("Chomage_US")
    if chomage and chomage["valeur"] is not None:
        if chomage["valeur"] < 4:
            score += 2; details["Chomage"] = f"✅ Chômage bas ({chomage['valeur']}%)"
        elif chomage["valeur"] < 5:
            score += 1; details["Chomage"] = f"🟡 Chômage modéré ({chomage['valeur']}%)"
        else:
            score -= 2; details["Chomage"] = f"🔴 Chômage élevé ({chomage['valeur']}%)"

    # --- Taux directeur ---
    fed = macro.get("Fed_Funds")
    if fed and fed["valeur"] is not None:
        if fed["valeur"] < 2:
            score += 2; details["Fed_Funds"] = f"✅ Taux bas — favorable marchés ({fed['valeur']}%)"
        elif fed["valeur"] < 4:
            score += 1; details["Fed_Funds"] = f"🟡 Taux modérés ({fed['valeur']}%)"
        elif fed["valeur"] < 6:
            score -= 1; details["Fed_Funds"] = f"🟠 Taux élevés ({fed['valeur']}%)"
        else:
            score -= 2; details["Fed_Funds"] = f"🔴 Taux très élevés ({fed['valeur']}%)"

    # --- Courbe des taux ---
    if "erreur" not in courbe:
        sp = courbe["spread"]
        if sp > 1:
            score += 2; details["Courbe"] = f"✅ Courbe normale (spread={sp:+.2f})"
        elif sp > 0:
            score += 1; details["Courbe"] = f"🟡 Courbe plate (spread={sp:+.2f})"
        elif sp > -0.5:
            score -= 1; details["Courbe"] = f"🟠 Courbe inversée (spread={sp:+.2f})"
        else:
            score -= 3; details["Courbe"] = f"🔴 Inversion sévère (spread={sp:+.2f})"

    # --- VIX ---
    if "erreur" not in vix:
        v = vix["valeur"]
        if v < 15:
            score += 2; details["VIX"] = f"✅ Volatilité faible (VIX={v})"
        elif v < 25:
            score += 1; details["VIX"] = f"🟡 Volatilité normale (VIX={v})"
        elif v < 35:
            score -= 1; details["VIX"] = f"🟠 Stress marché (VIX={v})"
        else:
            score -= 3; details["VIX"] = f"🔴 Panique (VIX={v})"

    # --- Fear & Greed ---
    if "erreur" not in fear_greed:
        fg = fear_greed["valeur"]
        if fg < 25:
            score += 2; details["Fear_Greed"] = f"✅ Extreme Fear = opportunité ({fg})"
        elif fg < 45:
            score += 1; details["Fear_Greed"] = f"🟡 Fear ({fg})"
        elif fg < 55:
            score += 0; details["Fear_Greed"] = f"⚪ Neutral ({fg})"
        elif fg < 75:
            score -= 1; details["Fear_Greed"] = f"🟠 Greed — prudence ({fg})"
        else:
            score -= 2; details["Fear_Greed"] = f"🔴 Extreme Greed — risque bulle ({fg})"

    # Recommandation globale
    if score >= 8:
        contexte = "🟢 TRÈS FAVORABLE — Conditions idéales pour investir"
    elif score >= 4:
        contexte = "🟡 FAVORABLE — Contexte positif"
    elif score >= 0:
        contexte = "⚪ NEUTRE — Prudence recommandée"
    elif score >= -4:
        contexte = "🟠 DÉFAVORABLE — Réduire l'exposition"
    else:
        contexte = "🔴 TRÈS DÉFAVORABLE — Risque élevé"

    return {
        "score":    score,
        "contexte": contexte,
        "details":  details,
        "date":     datetime.now().strftime("%Y-%m-%d %H:%M")
    }


# ============================================================
# PIPELINE MACRO
# ============================================================

def pipeline_macro() -> dict:
    """
    Pipeline complet — récupération + scoring macro
    """
    print("\n" + "="*60)
    print("🌍 ANALYSE MACROÉCONOMIQUE")
    print("="*60)

    # Données FRED
    fred_data = get_toutes_series_fred()

    # Données alternatives
    print("\n📡 Données de sentiment...")
    vix        = get_vix()
    fear_greed = get_fear_greed()
    courbe     = get_courbe_taux(fred_data=fred_data)

    print(f"  VIX          : {vix.get('valeur', 'N/A')} — {vix.get('signal', '')}")
    print(f"  Fear & Greed : {fear_greed.get('valeur', 'N/A')} ({fear_greed.get('rating', '')})")
    print(f"  Courbe taux  : spread={courbe.get('spread', 'N/A')} — {courbe.get('signal', '')}")

    # Scoring
    scoring = scorer_macro(fred_data, vix, fear_greed, courbe)

    # Affichage
    print(f"\n{'='*60}")
    print("📊 SCORING MACRO")
    print(f"{'='*60}")
    for indicateur, detail in scoring["details"].items():
        print(f"  {indicateur:15} → {detail}")
    print(f"\n  Score total : {scoring['score']:+d}")
    print(f"  Contexte    : {scoring['contexte']}")

    # Résultat complet
    resultat = {
        "fred":       fred_data,
        "vix":        vix,
        "fear_greed": fear_greed,
        "courbe":     courbe,
        "scoring":    scoring
    }

    # Sauvegarde
    chemin = "data/macro/macro_latest.json"
    with open(chemin, "w", encoding="utf-8") as f:
        json.dump(resultat, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n💾 Sauvegardé → {chemin}")
    return resultat


if __name__ == "__main__":
    pipeline_macro()