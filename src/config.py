"""Configuration centrale du projet : chemins, saisons, constantes."""
from pathlib import Path

# --- Chemins -----------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"

# --- Données -----------------------------------------------------------
# Source : https://www.football-data.co.uk/francem.php (Ligue 1)
BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/F1.csv"
SEASONS = ["2324", "2425", "2526","2627"]

# --- Colonnes conservées au preprocessing ------------------------------
ID_COLS = ["Season", "Date", "HomeTeam", "AwayTeam"]
RESULT_COLS = ["FTHG", "FTAG", "FTR"]
STAT_COLS = ["HS", "AS", "HST", "AST"]  # tirs / tirs cadrés (optionnels)
CLEAN_FILENAME = "matches_clean.csv"
FEATURES_FILENAME = "features.csv"
ROLLING_WINDOW = 5  # nombre de matchs passés pour la "forme récente"

# --- Métier ------------------------------------------------------------
# Cible : 1 si l'équipe à domicile gagne (FTR == "H"), 0 sinon (nul ou défaite)
TARGET = "home_win"
FOCUS_TEAM = "Paris SG"  # équipe utilisée pour les exemples / la prédiction finale
RANDOM_STATE = 42
