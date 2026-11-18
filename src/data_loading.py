import urllib.request

import pandas as pd

from src import config

def download_season(season: str, force: bool = False):
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    dest = config.RAW_DIR / f"F1_{season}.csv"

    if dest.exists() and not force:
        print(f"[skip]     {dest.name} existe déjà")
        return dest 

    url = config.BASE_URL.format(season=season)
    print(f"[download] {url}")
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request) as response, open(dest, "wb") as f:
        f.write(response.read())
    return dest

def download_all(force: bool = False) -> None:
    for season in config.SEASONS:
        download_season(season, force=force)


def load_raw() -> pd.DataFrame:
    """Charge et concatène toutes les saisons SANS aucune transformation,
    à part l'ajout d'une colonne 'Season' pour savoir d'où vient chaque ligne."""
    frames = []
    for season in config.SEASONS:
        path = config.RAW_DIR / f"F1_{season}.csv"
        df = pd.read_csv(path, encoding="latin-1")
        df = df.dropna(how="all")  # lignes totalement vides en fin de fichier
        df["Season"] = season
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    download_all()
    raw = load_raw()
    print(f"\nOK : {raw.shape[0]} matchs, {raw.shape[1]} colonnes")
