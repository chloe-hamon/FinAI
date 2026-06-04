

from langchain.tools import tool


@tool
def calcul_marge_nette(resultat_net: float, chiffre_affaires: float) -> float:
    """
    Calcule la marge nette.
    """
    return round((resultat_net / chiffre_affaires) * 100, 2)


@tool
def calcul_croissance(valeur_actuelle: float, valeur_precedente: float) -> float:
    """
    Calcule le taux de croissance.
    """
    return round(
        ((valeur_actuelle - valeur_precedente) / valeur_precedente) * 100,
        2
    )


@tool
def calcul_ratio_endettement(dette_totale: float, capitaux_propres: float) -> float:
    """
    Calcule le ratio d'endettement.
    """
    return round(
        dette_totale / capitaux_propres,
        2
    )


@tool
def verifier_alerte_seuil(valeur: float, seuil: float) -> str:
    """
    Vérifie si une alerte doit être déclenchée.
    """
    if valeur < seuil:
        return f"ALERTE : {valeur} est inférieur au seuil {seuil}"

    return f"OK : {valeur} est supérieur au seuil {seuil}"


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
