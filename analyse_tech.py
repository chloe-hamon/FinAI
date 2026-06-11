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
    > 70 : surachat (signal vente)
    < 30 : survente (signal achat)
    """
    delta = serie.diff()
    gain  = delta.clip(lower=0)
    perte = -delta.clip(upper=0)

    avg_gain = gain.ewm(com=periode - 1, min_periods=periode).mean()
    avg_perte = perte.ewm(com=periode - 1, min_periods=periode).mean()

    rs  = avg_gain / avg_perte
    rsi = 100 - (100 / (1 + rs))
    return rsi.round(2)


def calcul_macd(serie: pd.Series,
                rapide: int = 12,
                lent: int   = 26,
                signal: int = 9) -> pd.DataFrame:
    """
    MACD — Moving Average Convergence Divergence
    Croisement MACD > Signal → signal achat
    Croisement MACD < Signal → signal vente
    """
    ema_rapide = serie.ewm(span=rapide, adjust=False).mean()
    ema_lent   = serie.ewm(span=lent,   adjust=False).mean()

    macd        = ema_rapide - ema_lent
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogramme = macd - signal_line

    return pd.DataFrame({
        "MACD":       macd.round(2),
        "Signal":     signal_line.round(2),
        "Histogramme": histogramme.round(2)
    })


def calcul_bollinger(serie: pd.Series,
                     periode: int  = 20,
                     nb_ecart: float = 2.0) -> pd.DataFrame:
    """
    Bandes de Bollinger
    Prix > Bande haute → surachat
    Prix < Bande basse → survente
    """
    moyenne    = serie.rolling(window=periode).mean()
    ecart_type = serie.rolling(window=periode).std()

    bande_haute = moyenne + (nb_ecart * ecart_type)
    bande_basse = moyenne - (nb_ecart * ecart_type)

    return pd.DataFrame({
        "BB_Haute":  bande_haute.round(2),
        "BB_Moyenne": moyenne.round(2),
        "BB_Basse":  bande_basse.round(2)
    })


def calcul_moyennes_mobiles(serie: pd.Series) -> pd.DataFrame:
    """
    Moyennes mobiles simples : MM20, MM50, MM200
    MM20 > MM50 > MM200 → tendance haussière forte
    """
    return pd.DataFrame({
        "MM20":  serie.rolling(20).mean().round(2),
        "MM50":  serie.rolling(50).mean().round(2),
        "MM200": serie.rolling(200).mean().round(2)
    })


def calcul_atr(df: pd.DataFrame, periode: int = 14) -> pd.Series:
    """
    ATR — Average True Range
    Mesure la volatilité
    ATR élevé → forte volatilité
    """
    high = df["High"]
    low  = df["Low"]
    close_prec = df["Close"].shift(1)

    tr = pd.concat([
        high - low,
        (high - close_prec).abs(),
        (low  - close_prec).abs()
    ], axis=1).max(axis=1)

    return tr.ewm(com=periode - 1, min_periods=periode).mean().round(2)


def calcul_stochastique(df: pd.DataFrame, periode: int = 14) -> pd.DataFrame:
    """
    Oscillateur Stochastique %K et %D
    > 80 : surachat
    < 20 : survente
    """
    lowest_low   = df["Low"].rolling(periode).min()
    highest_high = df["High"].rolling(periode).max()

    k = 100 * (df["Close"] - lowest_low) / (highest_high - lowest_low)
    d = k.rolling(3).mean()

    return pd.DataFrame({
        "Stoch_K": k.round(2),
        "Stoch_D": d.round(2)
    })

# ============================================================
# ANALYSE COMPLÈTE D'UN ACTIF
# ============================================================

def analyser_actif(df: pd.DataFrame, nom: str) -> pd.DataFrame:
    """
    Calcule tous les indicateurs techniques sur un DataFrame OHLCV
    """
    result = df.copy()

    # RSI
    result["RSI"] = calcul_rsi(df["Close"])

    # MACD
    macd_df = calcul_macd(df["Close"])
    result  = pd.concat([result, macd_df], axis=1)

    # Bollinger
    boll_df = calcul_bollinger(df["Close"])
    result  = pd.concat([result, boll_df], axis=1)

    # Moyennes mobiles
    mm_df  = calcul_moyennes_mobiles(df["Close"])
    result = pd.concat([result, mm_df], axis=1)

    # ATR
    result["ATR"] = calcul_atr(df)

    # Stochastique
    stoch_df = calcul_stochastique(df)
    result   = pd.concat([result, stoch_df], axis=1)

    # Sauvegarde
    chemin = f"data/analyse_technique/{nom.lower().replace(' ', '_')}.csv"
    result.to_csv(chemin)
    print(f"✅ {nom} — analyse technique sauvegardée : {chemin}")

    return result

# ============================================================
# GÉNÉRATION DE SIGNAUX
# ============================================================

def generer_signaux(df: pd.DataFrame, nom: str) -> dict:
    """
    Génère des signaux BUY / SELL / NEUTRAL
    basés sur les indicateurs techniques
    """
    derniere = df.iloc[-1]
    signaux  = {}
    score    = 0  # positif = haussier, négatif = baissier

    # --- RSI ---
    rsi = derniere.get("RSI", None)
    if rsi is not None:
        if rsi < 30:
            signaux["RSI"] = f"BUY (RSI={rsi} → survente)"
            score += 2
        elif rsi > 70:
            signaux["RSI"] = f"SELL (RSI={rsi} → surachat)"
            score -= 2
        else:
            signaux["RSI"] = f"NEUTRAL (RSI={rsi})"

    # --- MACD ---
    macd   = derniere.get("MACD", None)
    signal = derniere.get("Signal", None)
    if macd is not None and signal is not None:
        if macd > signal:
            signaux["MACD"] = f"BUY (MACD={macd} > Signal={signal})"
            score += 2
        else:
            signaux["MACD"] = f"SELL (MACD={macd} < Signal={signal})"
            score -= 2

    # --- Bollinger ---
    close  = derniere.get("Close", None)
    bb_h   = derniere.get("BB_Haute", None)
    bb_b   = derniere.get("BB_Basse", None)
    if all(v is not None for v in [close, bb_h, bb_b]):
        if close < bb_b:
            signaux["Bollinger"] = f"BUY (prix={close} sous bande basse={bb_b})"
            score += 1
        elif close > bb_h:
            signaux["Bollinger"] = f"SELL (prix={close} sur bande haute={bb_h})"
            score -= 1
        else:
            signaux["Bollinger"] = f"NEUTRAL (prix dans les bandes)"

    # --- Moyennes Mobiles ---
    mm20  = derniere.get("MM20", None)
    mm50  = derniere.get("MM50", None)
    mm200 = derniere.get("MM200", None)
    if all(v is not None for v in [close, mm20, mm50, mm200]):
        if close > mm20 > mm50 > mm200:
            signaux["MM"] = "BUY (tendance haussière forte)"
            score += 2
        elif close < mm20 < mm50 < mm200:
            signaux["MM"] = "SELL (tendance baissière forte)"
            score -= 2
        elif close > mm50:
            signaux["MM"] = "BUY (prix au-dessus MM50)"
            score += 1
        else:
            signaux["MM"] = "SELL (prix sous MM50)"
            score -= 1

    # --- Stochastique ---
    stoch_k = derniere.get("Stoch_K", None)
    stoch_d = derniere.get("Stoch_D", None)
    if stoch_k is not None and stoch_d is not None:
        if stoch_k < 20 and stoch_k > stoch_d:
            signaux["Stochastique"] = f"BUY (K={stoch_k} < 20 et croisement haussier)"
            score += 1
        elif stoch_k > 80 and stoch_k < stoch_d:
            signaux["Stochastique"] = f"SELL (K={stoch_k} > 80 et croisement baissier)"
            score -= 1
        else:
            signaux["Stochastique"] = f"NEUTRAL (K={stoch_k})"

    # --- Recommandation finale ---
    if score >= 4:
        recommandation = "🟢 STRONG BUY"
    elif score >= 2:
        recommandation = "🟡 BUY"
    elif score <= -4:
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
        "signaux":        signaux
    }

# ============================================================
# PIPELINE COMPLET
# ============================================================

def pipeline_technique(dossier: str = "data/actions_clean") -> list[dict]:
    """
    Lance l'analyse technique sur tous les CSV d'un dossier
    """
    resultats = []

    for fichier in os.listdir(dossier):
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
            print(f"  Prix        : {signal['prix']}")
            print(f"  Score       : {signal['score']}")
            print(f"  Recommandation : {signal['recommandation']}")
            for ind, val in signal["signaux"].items():
                print(f"    {ind:15} → {val}")

        except Exception as e:
            print(f"❌ {nom} — Erreur : {e}")

    return resultats


if __name__ == "__main__":
    print("📊 Analyse technique — Actions")
    resultats_actions  = pipeline_technique("data/actions_clean")

    print("\n📊 Analyse technique — Indices")
    resultats_indices  = pipeline_technique("data/indices_clean")

    print("\n📊 Analyse technique — Crypto")
    resultats_crypto   = pipeline_technique("data/crypto_clean")

    print("\n📊 Analyse technique — Forex")
    resultats_forex    = pipeline_technique("data/forex_clean")

    total = resultats_actions + resultats_indices + resultats_crypto + resultats_forex
    print(f"\n🎉 {len(total)} actifs analysés au total")
