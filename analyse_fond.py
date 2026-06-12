import json
import os
from datetime import datetime

import yfinance as yf

os.makedirs("data/analyse_fondamentale", exist_ok=True)


# ============================================================
# RÉCUPÉRATION DES DONNÉES FONDAMENTALES
# ============================================================

def get_fondamentaux(ticker: str) -> dict:
    """Récupère toutes les données fondamentales via yFinance."""
    try:
        info = yf.Ticker(ticker).info
        data = {
            "ticker": ticker,
            "date":   datetime.now().strftime("%Y-%m-%d %H:%M"),
            # Valorisation
            "pe_ratio":              info.get("trailingPE"),
            "pe_forward":            info.get("forwardPE"),
            "pb_ratio":              info.get("priceToBook"),
            "ps_ratio":              info.get("priceToSalesTrailing12Months"),
            "ev_ebitda":             info.get("enterpriseToEbitda"),
            "peg_ratio":             info.get("pegRatio"),
            # Rentabilité
            "roe":                   info.get("returnOnEquity"),
            "roa":                   info.get("returnOnAssets"),
            "marge_nette":           info.get("profitMargins"),
            "marge_brute":           info.get("grossMargins"),
            "marge_operationnelle":  info.get("operatingMargins"),
            "ebitda":                info.get("ebitda"),
            # Croissance
            "croissance_revenus":    info.get("revenueGrowth"),
            "croissance_bpa":        info.get("earningsGrowth"),
            "revenus":               info.get("totalRevenue"),
            "bpa":                   info.get("trailingEps"),
            "bpa_forward":           info.get("forwardEps"),
            # Solidité financière
            "dette_totale":          info.get("totalDebt"),
            "cash":                  info.get("totalCash"),
            "ratio_dette_equity":    info.get("debtToEquity"),   
            "current_ratio":         info.get("currentRatio"),
            "quick_ratio":           info.get("quickRatio"),
            # Dividende
            "dividende_yield":       info.get("dividendYield"),
            "dividende_par_action":  info.get("dividendRate"),
            "payout_ratio":          info.get("payoutRatio"),
            # Marché
            "market_cap":            info.get("marketCap"),
            "beta":                  info.get("beta"),
            "52w_high":              info.get("fiftyTwoWeekHigh"),
            "52w_low":               info.get("fiftyTwoWeekLow"),
            "prix_actuel":           info.get("currentPrice"),
            "secteur":               info.get("sector"),
            "industrie":             info.get("industry"),
            "nom":                   info.get("longName"),
        }
        print(f"  ✅ {ticker} — données récupérées")
        return data

    except Exception as e:
        print(f"  ❌ {ticker} — {e}")
        return {"ticker": ticker, "erreur": str(e)}


# ============================================================
# SCORING FONDAMENTAL
# ============================================================

def scorer_valorisation(data: dict) -> tuple[int, dict]:
    """Score de valorisation — l'action est-elle bon marché ?"""
    score, details = 0, {}

    # P/E — 
    pe = data.get("pe_ratio")
    if pe is not None and pe > 0:
        if pe < 15:
            score += 2; details["PE"] = f"BUY (PE={pe:.1f} < 15 → sous-évalué)"
        elif pe < 25:
            score += 1; details["PE"] = f"NEUTRAL (PE={pe:.1f})"
        elif pe < 35:
            score -= 1; details["PE"] = f"ATTENTION (PE={pe:.1f} → cher)"
        else:
            score -= 2; details["PE"] = f"SELL (PE={pe:.1f} → très cher)"

    # PEG — 
    peg = data.get("peg_ratio")
    if peg is not None and peg > 0:
        if peg < 1:
            score += 2; details["PEG"] = f"BUY (PEG={peg:.2f} < 1)"
        elif peg < 2:
            score += 1; details["PEG"] = f"NEUTRAL (PEG={peg:.2f})"
        else:
            score -= 1; details["PEG"] = f"SELL (PEG={peg:.2f} > 2)"

    # P/B
    pb = data.get("pb_ratio")
    if pb is not None and pb > 0:
        if pb < 1:
            score += 2; details["PB"] = f"BUY (PB={pb:.2f} < 1 → sous valeur comptable)"
        elif pb < 3:
            score += 1; details["PB"] = f"NEUTRAL (PB={pb:.2f})"
        else:
            score -= 1; details["PB"] = f"ATTENTION (PB={pb:.2f} > 3)"

    # EV/EBITDA
    ev = data.get("ev_ebitda")
    if ev is not None and ev > 0:
        if ev < 10:
            score += 2; details["EV_EBITDA"] = f"BUY (EV/EBITDA={ev:.1f} < 10)"
        elif ev < 15:
            score += 1; details["EV_EBITDA"] = f"NEUTRAL (EV/EBITDA={ev:.1f})"
        else:
            score -= 1; details["EV_EBITDA"] = f"ATTENTION (EV/EBITDA={ev:.1f} > 15)"

    return score, details


def scorer_rentabilite(data: dict) -> tuple[int, dict]:
    """Score de rentabilité."""
    score, details = 0, {}

    roe = data.get("roe")
    if roe is not None:
        p = roe * 100
        if p > 20:
            score += 2; details["ROE"] = f"BUY (ROE={p:.1f}% > 20%)"
        elif p > 10:
            score += 1; details["ROE"] = f"NEUTRAL (ROE={p:.1f}%)"
        else:
            score -= 1; details["ROE"] = f"ATTENTION (ROE={p:.1f}% < 10%)"

    mn = data.get("marge_nette")
    if mn is not None:
        p = mn * 100
        if p > 20:
            score += 2; details["Marge_Nette"] = f"BUY (marge={p:.1f}%)"
        elif p > 10:
            score += 1; details["Marge_Nette"] = f"NEUTRAL (marge={p:.1f}%)"
        elif p > 0:
            score += 0; details["Marge_Nette"] = f"ATTENTION (marge={p:.1f}% faible)"
        else:
            score -= 2; details["Marge_Nette"] = f"SELL (marge={p:.1f}% négative)"

    roa = data.get("roa")
    if roa is not None:
        p = roa * 100
        if p > 10:
            score += 1; details["ROA"] = f"BUY (ROA={p:.1f}%)"
        elif p > 5:
            score += 0; details["ROA"] = f"NEUTRAL (ROA={p:.1f}%)"
        else:
            score -= 1; details["ROA"] = f"ATTENTION (ROA={p:.1f}% < 5%)"

    return score, details


def scorer_croissance(data: dict) -> tuple[int, dict]:
    """Score de croissance."""
    score, details = 0, {}

    cr = data.get("croissance_revenus")
    if cr is not None:
        p = cr * 100
        if p > 20:
            score += 2; details["Croissance_CA"] = f"BUY (croissance={p:.1f}%)"
        elif p > 10:
            score += 1; details["Croissance_CA"] = f"NEUTRAL (croissance={p:.1f}%)"
        elif p > 0:
            score += 0; details["Croissance_CA"] = f"ATTENTION (croissance={p:.1f}% faible)"
        else:
            score -= 2; details["Croissance_CA"] = f"SELL (croissance={p:.1f}% négative)"

    cb = data.get("croissance_bpa")
    if cb is not None:
        p = cb * 100
        if p > 15:
            score += 2; details["Croissance_BPA"] = f"BUY (BPA={p:.1f}%)"
        elif p > 5:
            score += 1; details["Croissance_BPA"] = f"NEUTRAL (BPA={p:.1f}%)"
        else:
            score -= 1; details["Croissance_BPA"] = f"ATTENTION (BPA={p:.1f}%)"

    return score, details


def scorer_sante_financiere(data: dict) -> tuple[int, dict]:
    """Score de santé financière."""
    score, details = 0, {}

    # debtToEquity retourné en % par yFinance (82.5 = 82.5%)
    de = data.get("ratio_dette_equity")
    if de is not None:
        if de < 50:
            score += 2; details["Dette_Equity"] = f"BUY (D/E={de:.1f}% → peu endetté)"
        elif de < 100:
            score += 1; details["Dette_Equity"] = f"NEUTRAL (D/E={de:.1f}%)"
        elif de < 200:
            score -= 1; details["Dette_Equity"] = f"ATTENTION (D/E={de:.1f}%)"
        else:
            score -= 2; details["Dette_Equity"] = f"SELL (D/E={de:.1f}% → très endetté)"

    cr = data.get("current_ratio")
    if cr is not None:
        if cr > 2:
            score += 1; details["Current_Ratio"] = f"BUY (CR={cr:.2f} → bonne liquidité)"
        elif cr > 1:
            score += 0; details["Current_Ratio"] = f"NEUTRAL (CR={cr:.2f})"
        else:
            score -= 2; details["Current_Ratio"] = f"SELL (CR={cr:.2f} < 1 → risque liquidité)"

    return score, details


def scorer_dividende(data: dict) -> tuple[int, dict]:
    """Score dividende."""
    score, details = 0, {}

    dy = data.get("dividende_yield")
    if dy is not None:
        p = dy * 100
        if p > 5:
            score += 2; details["Dividende"] = f"BUY (rendement={p:.1f}%)"
        elif p > 2:
            score += 1; details["Dividende"] = f"NEUTRAL (rendement={p:.1f}%)"
        elif p < 0.5:   # ← CORRIGÉ : seuil flottant au lieu de == 0
            details["Dividende"] = "Pas de dividende"

    pr = data.get("payout_ratio")
    if pr is not None:
        p = pr * 100
        if p > 80:
            score -= 1; details["Payout"] = f"ATTENTION (payout={p:.1f}% → risqué)"
        elif p < 50:
            score += 1; details["Payout"] = f"BUY (payout={p:.1f}% → soutenable)"

    return score, details


# ============================================================
# SCORING GLOBAL
# ============================================================

def analyser_fondamentaux(ticker: str) -> dict:
    """Pipeline complet d'analyse fondamentale."""
    data = get_fondamentaux(ticker)
    if "erreur" in data:
        return data

    s_val,  d_val  = scorer_valorisation(data)
    s_rent, d_rent = scorer_rentabilite(data)
    s_croi, d_croi = scorer_croissance(data)
    s_sant, d_sant = scorer_sante_financiere(data)
    s_div,  d_div  = scorer_dividende(data)

    score_total = s_val + s_rent + s_croi + s_sant + s_div

    if score_total >= 10:
        recommandation = "🟢 STRONG BUY"
    elif score_total >= 5:
        recommandation = "🟡 BUY"
    elif score_total <= -7:
        recommandation = "🔴 STRONG SELL"
    elif score_total <= -3:
        recommandation = "🟠 SELL"
    else:
        recommandation = "⚪ NEUTRAL"

    resultat = {
        "ticker":     ticker,
        "nom":        data.get("nom", ticker),
        "secteur":    data.get("secteur"),
        "date":       data.get("date"),
        "prix":       data.get("prix_actuel"),
        "market_cap": data.get("market_cap"),
        "beta":       data.get("beta"),
        "scores": {
            "valorisation":     s_val,
            "rentabilite":      s_rent,
            "croissance":       s_croi,
            "sante_financiere": s_sant,
            "dividende":        s_div,
            "total":            score_total,
        },
        "recommandation": recommandation,
        "details": {
            "valorisation":     d_val,
            "rentabilite":      d_rent,
            "croissance":       d_croi,
            "sante_financiere": d_sant,
            "dividende":        d_div,
        },
        "donnees_brutes": data,
    }

    chemin = f"data/analyse_fondamentale/{ticker.lower()}.json"
    with open(chemin, "w", encoding="utf-8") as f:
        json.dump(resultat, f, ensure_ascii=False, indent=2, default=str)

    return resultat


# ============================================================
# AFFICHAGE
# ============================================================

def afficher_analyse(resultat: dict):
    """Affichage formaté d'une analyse fondamentale."""
    if "erreur" in resultat:
        print(f"  ❌ {resultat['ticker']} — {resultat['erreur']}")
        return

    print(f"\n{'=' * 60}")
    print(f"📊 {resultat['nom']} ({resultat['ticker']})")
    print(f"   Secteur : {resultat['secteur']}")
    print(f"   Prix    : {resultat['prix']} | Beta : {resultat['beta']}")
    print(f"{'=' * 60}")

    for dim, details in resultat["details"].items():
        score = resultat["scores"][dim]
        print(f"\n  [{dim.upper()}]  Score : {score:+d}")
        for indicateur, valeur in details.items():
            print(f"    {indicateur:20} → {valeur}")

    print(f"\n  {'─' * 50}")
    print(f"  Score total    : {resultat['scores']['total']:+d}")
    print(f"  Recommandation : {resultat['recommandation']}")
    print(f"{'=' * 60}")


# ============================================================
# PIPELINE MULTI-TICKERS
# ============================================================

def pipeline_fondamental(tickers: list[str]) -> list[dict]:
    """Lance l'analyse fondamentale sur une liste de tickers."""
    resultats = []

    for ticker in tickers:
        print(f"\n🔍 {ticker}...")
        r = analyser_fondamentaux(ticker)
        afficher_analyse(r)
        resultats.append(r)

    valides = [r for r in resultats if "erreur" not in r]
    tries   = sorted(valides, key=lambda x: x["scores"]["total"], reverse=True)

    print(f"\n{'=' * 60}")
    print("🏆 CLASSEMENT FONDAMENTAL")
    print(f"{'=' * 60}")
    for i, r in enumerate(tries, 1):
        print(f"  {i:2}. {r['ticker']:8} | {r['scores']['total']:+3d} | {r['recommandation']}")

    return resultats


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    from chloe import ACTIONS

    tickers = list(ACTIONS.keys())
    print(f"🚀 Analyse fondamentale — {len(tickers)} actifs\n")
    resultats = pipeline_fondamental(tickers)
    print(f"\n✅ {len(resultats)} analyses terminées")
