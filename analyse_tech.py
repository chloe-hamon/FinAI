import pandas as pd
import numpy as np
import os

os.makedirs("data/analyse_technique", exist_ok=True)

# ============================================================
# INDICATEURS TECHNIQUES
# ============================================================

def calcul_rsi(serie: pd.Series, periode: int = 14) -> pd.Series:
    """
    RSI — Relative Strength Index
    > 70 : surachat  |  < 30 : survente
    """
    delta     = serie.diff()
    gain      = delta.clip(lower=0)
    perte     = -delta.clip(upper=0)
    avg_gain  = gain.ewm(com=periode - 1, min_periods=periode).mean()
    avg_perte = perte.ewm(com=periode - 1, min_periods=periode).mean()
    rs        = avg_gain / avg_perte
    return (100 - (100 / (1 + rs))).round(2)


def calcul_macd(serie: pd.Series,
                rapide: int = 12,
                lent:   int = 26,
                signal: int = 9) -> pd.DataFrame:
    """
    MACD — Moving Average Convergence Divergence
    """
    ema_rapide  = serie.ewm(span=rapide, adjust=False).mean()
    ema_lent    = serie.ewm(span=lent,   adjust=False).mean()
    macd        = ema_rapide - ema_lent
    signal_line = macd.ewm(span=signal,  adjust=False).mean()
    histogramme = macd - signal_line
    return pd.DataFrame({
        "MACD":        macd.round(2),
        "Signal":      signal_line.round(2),
        "Histogramme": histogramme.round(2),
    })


def calcul_bollinger(serie: pd.Series,
                     periode:  int   = 20,
                     nb_ecart: float = 2.0) -> pd.DataFrame:
    """
    Bandes de Bollinger
    """
    moyenne     = serie.rolling(window=periode).mean()
    ecart_type  = serie.rolling(window=periode).std()
    bande_haute = moyenne + (nb_ecart * ecart_type)
    bande_basse = moyenne - (nb_ecart * ecart_type)
    return pd.DataFrame({
        "BB_Haute":   bande_haute.round(2),
        "BB_Moyenne": moyenne.round(2),
        "BB_Basse":   bande_basse.round(2),
    })


def calcul_moyennes_mobiles(serie: pd.Series) -> pd.DataFrame:
    """
    Moyennes mobiles simples : MM20, MM50, MM200
    """
    return pd.DataFrame({
        "MM20":  serie.rolling(20).mean().round(2),
        "MM50":  serie.rolling(50).mean().round(2),
        "MM200": serie.rolling(200).mean().round(2),
    })


def calcul_atr(df: pd.DataFrame, periode: int = 14) -> pd.Series:
    """
    ATR — Average True Range (volatilité)
    """
    high       = df["High"]
    low        = df["Low"]
    close_prec = df["Close"].shift(1)
    tr = pd.concat([
        high - low,
        (high - close_prec).abs(),
        (low  - close_prec).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(com=periode - 1, min_periods=periode).mean().round(2)


def calcul_stochastique(df: pd.DataFrame, periode: int = 14) -> pd.DataFrame:
    """
    Oscillateur Stochastique %K et %D
    > 80 : surachat  |  < 20 : survente
    """
    lowest_low   = df["Low"].rolling(periode).min()
    highest_high = df["High"].rolling(periode).max()

    
    denominateur = (highest_high - lowest_low).replace(0, np.nan)
    k = 100 * (df["Close"] - lowest_low) / denominateur
    d = k.rolling(3).mean()
    return pd.DataFrame({
        "Stoch_K": k.round(2),
        "Stoch_D": d.round(2),
    })


# ============================================================
# ANALYSE COMPLÈTE D'UN ACTIF
# ============================================================

def analyser_actif(df: pd.DataFrame, nom: str) -> pd.DataFrame:
    """
    Calcule tous les indicateurs techniques sur un DataFrame OHLCV.
    """

    if df.empty:
        raise ValueError(f"{nom} — DataFrame vide")
    if len(df) < 200:
        print(f"  ⚠️  {nom} — {len(df)} lignes seulement, MM200 sera partielle")

    result = df.copy()
    result["RSI"] = calcul_rsi(df["Close"])
    result = pd.concat([result, calcul_macd(df["Close"])],          axis=1)
    result = pd.concat([result, calcul_bollinger(df["Close"])],      axis=1)
    result = pd.concat([result, calcul_moyennes_mobiles(df["Close"])], axis=1)
    result["ATR"] = calcul_atr(df)
    result = pd.concat([result, calcul_stochastique(df)],            axis=1)

    chemin = f"data/analyse_technique/{nom.lower().replace(' ', '_')}.csv"
    result.to_csv(chemin)
    print(f"  ✅ {nom} — sauvegardé : {chemin}")
    return result


# ============================================================
# GÉNÉRATION DE SIGNAUX
# ============================================================

def _get(serie: pd.Series, col: str):
    """Lecture sûre sur pd.Series — retourne None si colonne absente."""
    return serie[col] if col in serie.index else None


def generer_signaux(df: pd.DataFrame, nom: str) -> dict:
    """
    Génère des signaux BUY / SELL / NEUTRAL basés sur les indicateurs.
    """
    derniere = df.iloc[-1]
    signaux  = {}
    score    = 0

    # --- RSI (+2 / -2) ---
    rsi = _get(derniere, "RSI")
    if rsi is not None:
        if rsi < 30:
            signaux["RSI"] = f"BUY (RSI={rsi} → survente)";    score += 2
        elif rsi > 70:
            signaux["RSI"] = f"SELL (RSI={rsi} → surachat)";   score -= 2
        else:
            signaux["RSI"] = f"NEUTRAL (RSI={rsi})"

    # --- MACD (+1 / -1) —
    macd   = _get(derniere, "MACD")
    signal = _get(derniere, "Signal")
    if macd is not None and signal is not None:
        if macd > signal:
            signaux["MACD"] = f"BUY (MACD={macd} > Signal={signal})";  score += 1
        else:
            signaux["MACD"] = f"SELL (MACD={macd} < Signal={signal})"; score -= 1

    # --- Bollinger (+1 / -1) ---
    close = _get(derniere, "Close")
    bb_h  = _get(derniere, "BB_Haute")
    bb_b  = _get(derniere, "BB_Basse")
    if all(v is not None for v in [close, bb_h, bb_b]):
        if close < bb_b:
            signaux["Bollinger"] = f"BUY (prix={close} < bande basse={bb_b})";  score += 1
        elif close > bb_h:
            signaux["Bollinger"] = f"SELL (prix={close} > bande haute={bb_h})"; score -= 1
        else:
            signaux["Bollinger"] = "NEUTRAL (prix dans les bandes)"

    # --- Moyennes Mobiles (+2 / -2 / +1 / -1) ---
    mm20  = _get(derniere, "MM20")
    mm50  = _get(derniere, "MM50")
    mm200 = _get(derniere, "MM200")
    if all(v is not None for v in [close, mm20, mm50, mm200]):
        if close > mm20 > mm50 > mm200:
            signaux["MM"] = "BUY (tendance haussière forte)";  score += 2
        elif close < mm20 < mm50 < mm200:
            signaux["MM"] = "SELL (tendance baissière forte)"; score -= 2
        elif close > mm50:
            signaux["MM"] = "BUY (prix au-dessus MM50)";       score += 1
        else:
            signaux["MM"] = "SELL (prix sous MM50)";           score -= 1

    # --- Stochastique (+1 / -1) ---
    stoch_k = _get(derniere, "Stoch_K")
    stoch_d = _get(derniere, "Stoch_D")
    if stoch_k is not None and stoch_d is not None:
        if stoch_k < 20 and stoch_k > stoch_d:
            signaux["Stochastique"] = f"BUY (K={stoch_k} croisement haussier)";  score += 1
        elif stoch_k > 80 and stoch_k < stoch_d:
            signaux["Stochastique"] = f"SELL (K={stoch_k} croisement baissier)"; score -= 1
        else:
            signaux["Stochastique"] = f"NEUTRAL (K={stoch_k})"

    # --- Recommandation finale ---
    # Score max = 2+1+1+2+1 = 7
    if score >= 5:
        recommandation = "🟢 STRONG BUY"
    elif score >= 2:
        recommandation = "🟡 BUY"
    elif score <= -5:
        recommandation = "🔴 STRONG SELL"
    elif score <= -2:
        recommandation = "🟠 SELL"
    else:
        recommandation = "⚪ NEUTRAL"

    return {
        "nom":            nom,
        "date":           str(df.index[-1]),
        "prix":           close,
        "score":          score,
        "recommandation": recommandation,
        "signaux":        signaux,
    }


# ============================================================
# PIPELINE COMPLET
# ============================================================

def pipeline_technique(dossier: str = "data/actions_clean") -> list[dict]:
    """
    Lance l'analyse technique sur tous les CSV d'un dossier.
    """
    if not os.path.isdir(dossier):
        print(f"⚠️  Dossier introuvable : {dossier}")
        return []

    resultats = []

    for fichier in sorted(os.listdir(dossier)):
        if not fichier.endswith(".csv"):
            continue

        nom    = fichier.replace(".csv", "")
        chemin = os.path.join(dossier, fichier)

        try:
            df     = pd.read_csv(chemin, index_col=0, parse_dates=True)
            df_ind = analyser_actif(df, nom)
            signal = generer_signaux(df_ind, nom)
            resultats.append(signal)

            print(f"\n📊 {nom}")
            print(f"  Prix           : {signal['prix']}")
            print(f"  Score          : {signal['score']}")
            print(f"  Recommandation : {signal['recommandation']}")
            for ind, val in signal["signaux"].items():
                print(f"    {ind:15} → {val}")

        except Exception as e:
            print(f"  ❌ {nom} — Erreur : {e}")

    return resultats


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    dossiers = [
        ("Actions",  "data/actions_clean"),
        ("Indices",  "data/indices_clean"),
        ("Crypto",   "data/crypto_clean"),
        ("Forex",    "data/forex_clean"),
    ]

    total = []
    for label, dossier in dossiers:
        print(f"\n{'='*60}")
        print(f"📊 Analyse technique — {label}")
        print(f"{'='*60}")
        total += pipeline_technique(dossier)

    print(f"\n🎉 {len(total)} actifs analysés au total")
