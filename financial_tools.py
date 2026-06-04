def calcul_marge_nette(resultat_net, chiffre_affaires):
    if chiffre_affaires == 0:
        return "Erreur : le chiffre d'affaires ne peut pas être égal à 0."

    marge = (resultat_net / chiffre_affaires) * 100
    return round(marge, 2)


def calcul_croissance(valeur_actuelle, valeur_precedente):
    if valeur_precedente == 0:
        return "Erreur : la valeur précédente ne peut pas être égale à 0."

    croissance = ((valeur_actuelle - valeur_precedente) / valeur_precedente) * 100
    return round(croissance, 2)


def calcul_ratio_endettement(dette_totale, capitaux_propres):
    if capitaux_propres == 0:
        return "Erreur : les capitaux propres ne peuvent pas être égaux à 0."

    ratio = dette_totale / capitaux_propres
    return round(ratio, 2)


def alerte_seuil(nom_indicateur, valeur, seuil, condition):
    if condition == "inferieur":
        if valeur < seuil:
            return f"Alerte : {nom_indicateur} est inférieur au seuil fixé ({valeur} < {seuil})."
        else:
            return f"OK : {nom_indicateur} est au-dessus du seuil fixé ({valeur} >= {seuil})."

    elif condition == "superieur":
        if valeur > seuil:
            return f"Alerte : {nom_indicateur} est supérieur au seuil fixé ({valeur} > {seuil})."
        else:
            return f"OK : {nom_indicateur} est sous le seuil fixé ({valeur} <= {seuil})."

    else:
        return "Erreur : condition inconnue."


if __name__ == "__main__":
    print("Marge nette :", calcul_marge_nette(12000, 100000), "%")
    print("Croissance :", calcul_croissance(120000, 100000), "%")
    print("Ratio endettement :", calcul_ratio_endettement(50000, 100000))
    print(alerte_seuil("Marge nette", 4.5, 5, "inferieur"))
