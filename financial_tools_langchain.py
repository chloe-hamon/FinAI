

from langchain.tools import tool


@tool
def calcul_marge_nette(resultat_net: float, chiffre_affaires: float) -> float:
    """
    Calcule la marge nette en % à partir du résultat net et du chiffre d'affaires.
    Utilise cet outil quand on demande la rentabilité ou la profitabilité d'une entreprise.
    Paramètres : resultat_net (€/$), chiffre_affaires (€/$)
    """
    if chiffre_affaires == 0:
        return "Erreur : chiffre d'affaires ne peut pas être zéro."
    
    marge = round((resultat_net / chiffre_affaires) * 100, 2)
    return f"Marge nette : {marge}%"


@tool
def calcul_croissance(valeur_actuelle: float, valeur_precedente: float) -> float:
    """
    Calcule le taux de croissance en pourcentage entre deux périodes.
    Utilise cet outil pour comparer des revenus, bénéfices ou tout 
    indicateur financier entre deux années ou trimestres.
    Paramètres : valeur_actuelle, valeur_precedente
    """
    if valeur_precedente == 0:
        return "Erreur : valeur précédente ne peut pas être zéro."
    
    croissance = round(
        ((valeur_actuelle - valeur_precedente) / valeur_precedente) * 100, 2
    )
    tendance = "📈" if croissance > 0 else "📉"
    return f"Croissance : {tendance} {croissance}%"


@tool
def calcul_ratio_endettement(dette_totale: float, capitaux_propres: float) -> float:
    """
    Calcule le ratio d'endettement (dette / capitaux propres).
    Utilise cet outil pour évaluer le niveau de risque financier 
    ou la structure du capital d'une entreprise.
    Un ratio > 2 est généralement considéré comme élevé.
    Paramètres : dette_totale, capitaux_propres
    """
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
def verifier_alerte_seuil(valeur: float, seuil: float) -> str:
    """
    Vérifie si une valeur financière dépasse ou est en dessous d'un seuil critique.
    Utilise cet outil pour déclencher des alertes sur des indicateurs 
    comme le RSI, un ratio financier ou un cours de bourse.
    Paramètres : valeur (valeur actuelle), seuil (limite critique)
    """
    if valeur < seuil:
        ecart = round(seuil - valeur, 4)
        return f"⚠️ ALERTE : {valeur} est inférieur au seuil {seuil} (écart : -{ecart})"
    
    ecart = round(valeur - seuil, 4)
    return f"✅ OK : {valeur} est au-dessus du seuil {seuil} (marge : +{ecart})"


if __name__ == "__main__":
    print("=== Tests financial_tools_langchain ===\n")

    print(calcul_marge_nette.invoke({
        "resultat_net": 12000,
        "chiffre_affaires": 100000
    }))

    print(calcul_croissance.invoke({
        "valeur_actuelle": 120000,
        "valeur_precedente": 100000
    }))

    print(calcul_ratio_endettement.invoke({
        "dette_totale": 50000,
        "capitaux_propres": 100000
    }))

    print(verifier_alerte_seuil.invoke({
        "valeur": 4.5,
        "seuil": 5
    }))
