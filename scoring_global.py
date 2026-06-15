import json
import os
from datetime import datetime
from chloe import PATHS

os.makedirs(PATHS["scoring"], exist_ok=True)

# ============================================================
# PONDÉRATIONS PAR TYPE D'ACTIF
# ============================================================

POIDS = {
    "action": {
        "technique":    0.35,
        "fondamental":  0.35,
        "sentiment":    0.15,
        "macro":        0.15,
    },
    "etf": {
        "technique":    0.40,
        "fondamental":  0.20,
        "sentiment":    0.15,
        "macro":        0.25,
    },
    "crypto": {
        "technique":    0.45,
        "fondamental":  0.00,
        "sentiment":    0.35,
        "macro":        0.20,
    },
    "matiere_premiere": {
        "technique":    0.35,
        "fondamental":  0.00,
        "sentiment":    0.25,
        "macro":        0.40,
    },
}
# ============================================================
# SENTIMENT PAR TICKER — DEPUIS FEAR & GREED GLOBAL
# ============================================================

def construire_senti_par_ticker(
    actifs:          dict,
    fear_greed:      dict | None = None,
    scores_news:     dict | None = None,
) -> dict:
    """
    Construit un score sentiment [-10, +10] par ticker.

    Sources utilisées (par priorité) :
    1. scores_news   → {ticker: score} si disponible (analyse NLP des news)
    2. fear_greed    → score global CNN Fear & Greed converti en [-10, +10]
    3. Fallback      → 0 (neutre)

    Ajustement par type d'actif :
    - crypto         → sentiment amplifié (×1.2)
    - matiere_premiere → légèrement amplifié (×1.1)
    - action / etf   → score brut
    """
    resultats = {}

    # Conversion Fear & Greed [0-100] → [-10, +10]
    score_fg_global = None
    if fear_greed and "erreur" not in fear_greed and "valeur" in fear_greed:
        # 50 = neutre → 0, 0 = extreme fear → +10 (opportunité), 100 = extreme greed → -10
        fg_val = fear_greed["valeur"]
        # Formule : (50 - fg) / 5  → [+10 à -10]
        score_fg_global = round((50 - fg_val) / 5, 2)
        score_fg_global = max(-10, min(10, score_fg_global))

    for ticker, type_actif in actifs.items():

        # Priorité 1 : score news NLP individuels
        if scores_news and ticker in scores_news:
            score_base = scores_news[ticker]

        # Priorité 2 : Fear & Greed global
        elif score_fg_global is not None:
            score_base = score_fg_global

        # Priorité 3 : neutre
        else:
            score_base = 0

        # Ajustement par type d'actif
        if type_actif == "crypto":
            score_final = round(score_base * 1.2, 2)
        elif type_actif == "matiere_premiere":
            score_final = round(score_base * 1.1, 2)
        else:
            score_final = score_base

        # Clamp final
        resultats[ticker] = max(-10, min(10, score_final))

    return resultats
# ============================================================
# SCORE GLOBAL
# ============================================================

def scorer_actif(
    ticker:       str,
    type_actif:   str,
    score_tech:   float | None = None,
    score_fond:   float | None = None,
    score_senti:  float | None = None,
    score_macro:  float | None = None,
) -> dict:
    """
    Calcule le score global pondéré d'un actif
    Tous les scores sont sur une échelle [-10, +10].

    """
    poids = POIDS.get(type_actif, POIDS["action"])

    scores = {
        "technique":   score_tech  if score_tech  is not None else 0,
        "fondamental": score_fond  if score_fond  is not None else 0,
        "sentiment":   score_senti if score_senti is not None else 0,
        "macro":       score_macro if score_macro is not None else 0,
    }

    # Score pondéré
    score_total = round(max(-10, min(10, sum(scores[k] * poids[k] for k in scores))), 2)


    # Recommandation
    if score_total >= 6:
        recommandation = "🟢 STRONG BUY"
        action         = "ACHETER fort"
    elif score_total >= 3:
        recommandation = "🟡 BUY"
        action         = "Acheter"
    elif score_total >= 1:
        recommandation = "🔵 WEAK BUY"
        action         = "Surveiller — légère opportunité"
    elif score_total >= -1:
        recommandation = "⚪ NEUTRAL"
        action         = "Ne rien faire"
    elif score_total >= -3:
        recommandation = "🟠 WEAK SELL"
        action         = "Réduire la position"
    elif score_total >= -6:
        recommandation = "🔴 SELL"
        action         = "Vendre"
    else:
        recommandation = "⛔ STRONG SELL"
        action         = "Sortir immédiatement"

    # Confiance (nombre de scores disponibles)
    nb_scores     = sum(1 for v in [score_tech, score_fond, score_senti, score_macro] if v is not None)
    niveaux = ["⚠️ Faible", "🟡 Moyenne", "🟠 Bonne", "🟢 Élevée"]
    confiance = niveaux[min(max(nb_scores - 1, 0), 3)]


    return {
        "ticker":        ticker,
        "type":          type_actif,
        "date":          datetime.now().strftime("%Y-%m-%d %H:%M"),
        "scores": {
            "technique":   scores["technique"],
            "fondamental": scores["fondamental"],
            "sentiment":   scores["sentiment"],
            "macro":       scores["macro"],
            "total":       score_total,
        },
        "poids":          poids,
        "recommandation": recommandation,
        "action":         action,
        "confiance":      confiance,
        "nb_sources":     nb_scores,
    }


# ============================================================
# AFFICHAGE
# ============================================================

def afficher_scoring(resultat: dict):
    print(f"\n{'='*60}")
    print(f"  {resultat['ticker']} — {resultat['type'].upper()}")
    print(f"{'='*60}")
    print(f"  Technique    : {resultat['scores']['technique']:+.1f}  (poids {resultat['poids']['technique']*100:.0f}%)")
    print(f"  Fondamental  : {resultat['scores']['fondamental']:+.1f}  (poids {resultat['poids']['fondamental']*100:.0f}%)")
    print(f"  Sentiment    : {resultat['scores']['sentiment']:+.1f}  (poids {resultat['poids']['sentiment']*100:.0f}%)")
    print(f"  Macro        : {resultat['scores']['macro']:+.1f}  (poids {resultat['poids']['macro']*100:.0f}%)")
    print(f"  {'─'*40}")
    print(f"  Score total  : {resultat['scores']['total']:+.2f}")
    print(f"  Décision     : {resultat['recommandation']}")
    print(f"  Action       : {resultat['action']}")
    print(f"  Confiance    : {resultat['confiance']} ({resultat['nb_sources']}/4 sources)")


# ============================================================
# PIPELINE MULTI-ACTIFS
# ============================================================

def pipeline_scoring_global(
    resultats_tech:  list[dict],   
    resultats_fond:  list[dict],   
    score_macro:     float,
    actifs:          dict,         
    fear_greed:      dict | None = None,
    scores_news:     dict | None = None,
) -> list[dict]:
    
    tech_par_ticker  = {
        r["nom"]: r["score"]
        for r in resultats_tech
        if "nom" in r and "score" in r
    }
    fond_par_ticker  = {
        r["ticker"]: r["scores"]["total"]
        for r in resultats_fond
        if "ticker" in r and "erreur" not in r
    }

     # Construction automatique du sentiment par ticker
    resultats_senti = construire_senti_par_ticker(
        actifs      = actifs,
        fear_greed  = fear_greed,
        scores_news = scores_news,
    )
    
    scores_finaux = []

    for ticker, type_actif in actifs.items():
        resultat = scorer_actif(
            ticker      = ticker,
            type_actif  = type_actif,
            score_tech  = tech_par_ticker.get(ticker),
            score_fond  = fond_par_ticker.get(ticker),
            score_senti = resultats_senti.get(ticker),  
            score_macro = score_macro,
        )
        afficher_scoring(resultat)
        scores_finaux.append(resultat)

    # Classement final
    tries = sorted(scores_finaux, key=lambda x: x["scores"]["total"], reverse=True)

    print(f"\n{'='*60}")
    print("🏆 CLASSEMENT GLOBAL")
    print(f"{'='*60}")
    for i, r in enumerate(tries, 1):
        print(f"  {i:2}. {r['ticker']:8} | {r['scores']['total']:+5.2f} | {r['recommandation']}")

    # Sauvegarde
    chemin = os.path.join(
        PATHS["scoring"],
        f"scoring_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    )
    with open(chemin, "w", encoding="utf-8") as f:
        json.dump(tries, f, ensure_ascii=False, indent=2)

    print(f"\n💾 Sauvegardé → {chemin}")
    return tries


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    # Test scorer_actif
    test = scorer_actif(
        ticker      = "AAPL",
        type_actif  = "action",
        score_tech  = 6,
        score_fond  = 4,
        score_senti = 2,
        score_macro = -1,
    )
    afficher_scoring(test)

    # Test construire_senti_par_ticker
    print("\n--- Test sentiment ---")
    actifs_test = {
        "AAPL": "action",
        "BTC-USD": "crypto",
        "GC=F": "matiere_premiere",
    }
    fg_test = {"valeur": 30, "rating": "Fear"}  # Fear → score positif
    senti = construire_senti_par_ticker(actifs_test, fear_greed=fg_test)
    for t, s in senti.items():
        print(f"  {t:12} → sentiment {s:+.2f}")