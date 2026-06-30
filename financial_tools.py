#============================================================
# INDICATEURS FINANCIERS FONDAMENTAUX
# ============================================================


def calcul_marge_nette(resultat_net: float, chiffre_affaires: float) -> float:
    """
    Marge nette = (Résultat net / CA) × 100
    Lève ValueError si CA == 0.
    """
    if chiffre_affaires == 0:
        raise ValueError("Le chiffre d'affaires ne peut pas être nul.")
    return round((resultat_net / chiffre_affaires) * 100, 2)


def calcul_croissance(valeur_actuelle: float, valeur_precedente: float) -> float:
    """
    Taux de croissance = ((V_actuelle - V_précédente) / V_précédente) × 100
    Lève ValueError si valeur_precedente == 0.
    """
    if valeur_precedente == 0:
        raise ValueError("La valeur précédente ne peut pas être nulle.")
    return round(((valeur_actuelle - valeur_precedente) / valeur_precedente) * 100, 2)


def calcul_ratio_endettement(dette_totale: float, capitaux_propres: float) -> float:
    """
    Ratio d'endettement = Dette totale / Capitaux propres
    Lève ValueError si capitaux_propres == 0.
    """
    if capitaux_propres == 0:
        raise ValueError("Les capitaux propres ne peuvent pas être nuls.")
    return round(dette_totale / capitaux_propres, 2)


# ============================================================
# ALERTES
# ============================================================

_CONDITIONS = {"inferieur", "superieur", "egal"}


def alerte_seuil(
    nom_indicateur: str,
    valeur: float,
    seuil: float,
    condition: str,
) -> str:
    """
    Génère un message d'alerte selon la condition.

    condition : "inferieur" | "superieur" | "egal"
    """
    if condition not in _CONDITIONS:
        raise ValueError(f"Condition inconnue : '{condition}'. Valeurs : {_CONDITIONS}")

    if condition == "inferieur":
        if valeur < seuil:
            return f"🔴 Alerte : {nom_indicateur} sous le seuil ({valeur} < {seuil})."
        return     f"🟢 OK : {nom_indicateur} au-dessus du seuil ({valeur} >= {seuil})."

    if condition == "superieur":
        if valeur > seuil:
            return f"🔴 Alerte : {nom_indicateur} au-dessus du seuil ({valeur} > {seuil})."
        return     f"🟢 OK : {nom_indicateur} sous le seuil ({valeur} <= {seuil})."

    # egal
    if valeur == seuil:
        return f"🟡 Alerte : {nom_indicateur} exactement au seuil ({valeur} = {seuil})."
    return     f"🟢 OK : {nom_indicateur} différent du seuil ({valeur} ≠ {seuil})."


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    try:
        print("Marge nette     :", calcul_marge_nette(12_000, 100_000), "%")
        print("Croissance      :", calcul_croissance(120_000, 100_000), "%")
        print("Ratio dettement :", calcul_ratio_endettement(50_000, 100_000))
        print(alerte_seuil("Marge nette", 4.5, 5.0, "inferieur"))
    except ValueError as e:
        print(f"❌ Erreur : {e}")
