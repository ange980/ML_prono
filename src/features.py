import numpy as np
import pandas as pd

from src import config
from src.data_cleaning import load_clean

STATS = ["pts", "gf", "ga"]  # points, buts marqués (goals for), buts encaissés (goals against)
INFO_COLS = ["Date", "Season", "HomeTeam", "AwayTeam"]  # identifiants : PAS des features


def feature_names(window: int = config.ROLLING_WINDOW) -> list[str]:
    """Liste ordonnée des colonnes utilisées comme variables d'entrée du modèle."""
    cols = []
    for prefix in ["home", "away", "diff"]:
        cols += [f"{prefix}_{stat}_last{window}" for stat in STATS]
    return cols


FEATURES = feature_names()


def to_team_matches(matches: pd.DataFrame) -> pd.DataFrame:
    """Passe d'une ligne par match à deux lignes par match (une par équipe).

    Chaque ligne décrit le match du point de vue d'une équipe : ses buts marqués (gf),
    encaissés (ga) et les points obtenus (3 victoire, 1 nul, 0 défaite).
    """
    matches = matches.reset_index(drop=True)
    home = pd.DataFrame({
        "match_id": matches.index, "Date": matches["Date"], "side": "home",
        "team": matches["HomeTeam"], "gf": matches["FTHG"], "ga": matches["FTAG"],
    })
    away = pd.DataFrame({
        "match_id": matches.index, "Date": matches["Date"], "side": "away",
        "team": matches["AwayTeam"], "gf": matches["FTAG"], "ga": matches["FTHG"],
    })
    long = pd.concat([home, away], ignore_index=True)
    long["pts"] = np.select([long["gf"] > long["ga"], long["gf"] == long["ga"]], [3, 1], default=0)
    return long


def add_rolling_form(long: pd.DataFrame, window: int = config.ROLLING_WINDOW) -> pd.DataFrame:
    """Ajoute, pour chaque équipe, la moyenne de ses `window` matchs PRÉCÉDENTS.

    shift(1) décale d'un cran : la ligne d'un match ne voit que les matchs d'avant.
    Sans lui, le résultat du match à prédire serait inclus dans sa propre feature
    (data leakage) et le modèle aurait l'air excellent... pour de mauvaises raisons.
    """
    long = long.sort_values(["team", "Date", "match_id"]).reset_index(drop=True)
    for stat in STATS:
        long[f"{stat}_last{window}"] = long.groupby("team")[stat].transform(
            lambda s: s.shift(1).rolling(window, min_periods=window).mean()
        )
    return long


def build_features(matches: pd.DataFrame, window: int = config.ROLLING_WINDOW) -> pd.DataFrame:
    """Construit le tableau final : une ligne par match, prêt pour l'entraînement."""
    matches = matches.sort_values(["Date", "HomeTeam"]).reset_index(drop=True)
    long = add_rolling_form(to_team_matches(matches), window)

    form_cols = [f"{stat}_last{window}" for stat in STATS]
    home = long[long["side"] == "home"].set_index("match_id")[form_cols].add_prefix("home_")
    away = long[long["side"] == "away"].set_index("match_id")[form_cols].add_prefix("away_")

    df = matches.join(home).join(away)

    # Comparaison directe : forme de l'équipe à domicile moins celle de l'équipe à l'extérieur
    for stat in STATS:
        col = f"{stat}_last{window}"
        df[f"diff_{col}"] = df[f"home_{col}"] - df[f"away_{col}"]

    # Les premiers matchs de chaque équipe n'ont pas assez d'historique : on les retire
    names = feature_names(window)
    df = df.dropna(subset=names)

    return df[INFO_COLS + names + [config.TARGET]].reset_index(drop=True)


def validate(df: pd.DataFrame, window: int = config.ROLLING_WINDOW) -> None:
    names = feature_names(window)
    problems = []
    if df[names].isna().any().any():
        problems.append("valeurs manquantes dans les features")
    if not df[config.TARGET].isin([0, 1]).all():
        problems.append("cible non binaire")
    leaking = {"FTHG", "FTAG", "FTR"} & set(names)
    if leaking:
        problems.append(f"colonnes du match en cours utilisées comme features : {leaking}")
    if problems:
        raise ValueError("Features invalides : " + " ; ".join(problems))


def save_features(df: pd.DataFrame):
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = config.PROCESSED_DIR / config.FEATURES_FILENAME
    df.to_csv(path, index=False)
    return path


def load_features() -> pd.DataFrame:
    path = config.PROCESSED_DIR / config.FEATURES_FILENAME
    return pd.read_csv(path, parse_dates=["Date"], dtype={"Season": str})


def run() -> pd.DataFrame:
    clean = load_clean()
    feats = build_features(clean)
    validate(feats)
    path = save_features(feats)
    print(f"Matchs nettoyés : {len(clean)}")
    print(f"Matchs avec features : {len(feats)} (retirés faute d'historique : {len(clean) - len(feats)})")
    print(f"Features ({len(FEATURES)}) : {FEATURES}")
    print(f"Sauvegardé dans {path}")
    return feats


if __name__ == "__main__":
    run()
