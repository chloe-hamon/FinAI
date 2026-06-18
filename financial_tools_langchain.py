from langchain.tools import tool


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


if __name__ == "__main__":
    print("=== Tests financial_tools_langchain ===\n")
    print(calcul_marge_nette.invoke("12000, 100000"))
    print(calcul_croissance.invoke("120000, 100000"))
    print(calcul_ratio_endettement.invoke("50000, 100000"))
    print(verifier_alerte.invoke("4.5, 3.0, 5.0, RSI"))