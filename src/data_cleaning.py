import numpy as np
import pandas as pd

from src import config
from src.data_loading import load_raw

ESSENTIAL_COLS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]


def clean_matches(raw: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Transforme les données brutes en un tableau propre : une ligne = un match."""
    df = raw.copy()

    def log(step: str, before: int) -> None:
        if verbose:
            print(f"  {step:<45} -{before - len(df)} lignes")

    # 1. Garder uniquement les colonnes utiles (stats de tirs : si présentes)
    stat_cols = [c for c in config.STAT_COLS if c in df.columns]
    df = df[config.ID_COLS + config.RESULT_COLS + stat_cols].copy()

    # 2. Dates : texte -> datetime (formats jj/mm/aaaa et jj/mm/aa mélangés)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, format="mixed", errors="coerce")

    # 3. Noms d'équipes : retirer les espaces parasites
    for col in ["HomeTeam", "AwayTeam"]:
        df[col] = df[col].str.strip()

    # 4. Supprimer les matchs sans information essentielle (date, équipes, score)
    n = len(df)
    df = df.dropna(subset=ESSENTIAL_COLS)
    log("Lignes sans date / équipes / score", n)

    # 5. Supprimer les doublons
    n = len(df)
    df = df.drop_duplicates(subset=["Date", "HomeTeam", "AwayTeam"])
    log("Doublons", n)

    # 6. Types : les buts sont des entiers
    df[["FTHG", "FTAG"]] = df[["FTHG", "FTAG"]].astype(int)

    # 7. Cible : 1 si l'équipe à domicile gagne, 0 sinon
    df[config.TARGET] = (df["FTHG"] > df["FTAG"]).astype(int)

    # 8. Ordre chronologique (indispensable pour le feature engineering)
    df = df.sort_values(["Date", "HomeTeam"]).reset_index(drop=True)

    validate(df)
    return df


def validate(df: pd.DataFrame) -> None:
    """Vérifie les invariants du tableau nettoyé. Lève une ValueError sinon."""
    problems = []

    if df[ESSENTIAL_COLS].isna().any().any():
        problems.append("valeurs manquantes dans les colonnes essentielles")
    if (df[["FTHG", "FTAG"]] < 0).any().any():
        problems.append("buts négatifs")
    if not df["Date"].is_monotonic_increasing:
        problems.append("dates non triées")
    if (df["HomeTeam"] == df["AwayTeam"]).any():
        problems.append("équipe qui joue contre elle-même")
    if not df[config.TARGET].isin([0, 1]).all():
        problems.append("cible non binaire")
    if (df[config.TARGET] != (df["FTR"] == "H").astype(int)).any():
        problems.append("cible incohérente avec la colonne FTR")

    if problems:
        raise ValueError("Données invalides : " + " ; ".join(problems))


def save_clean(df: pd.DataFrame):
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = config.PROCESSED_DIR / config.CLEAN_FILENAME
    df.to_csv(path, index=False)
    return path


def load_clean() -> pd.DataFrame:
    """Recharge le fichier nettoyé (avec les dates au bon format)."""
    path = config.PROCESSED_DIR / config.CLEAN_FILENAME
    return pd.read_csv(path, parse_dates=["Date"])


def run() -> pd.DataFrame:
    raw = load_raw()
    print(f"Données brutes : {raw.shape[0]} lignes, {raw.shape[1]} colonnes")
    clean = clean_matches(raw)
    path = save_clean(clean)
    print(f"Données propres : {clean.shape[0]} lignes, {clean.shape[1]} colonnes -> {path}")
    return clean


if __name__ == "__main__":
    run()
