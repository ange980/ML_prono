"""Tests du preprocessing sur de faux fichiers au format football-data.co.uk."""
import pandas as pd
import pytest

from src import config, data_cleaning as preprocessing
from src.data_loading import load_raw

HEADER = "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,HS,AS,HST,AST"

ROWS_1819 = [
    "F1,10/08/2018,Paris SG,Caen,3,0,H,15,5,8,1",
    "F1,11/08/2018,Lyon ,Nice,2,2,D,10,9,4,3",   # espace parasite dans le nom
    "F1,11/08/2018,Lyon ,Nice,2,2,D,10,9,4,3",   # doublon exact
    "F1,12/08/2018,Lille,Marseille,,,,,,,",       # match sans score
    ",,,,,,,,,,",                                  # ligne vide
]
ROWS_1920 = [
    "F1,09/08/19,Monaco,Paris SG,1,3,A,8,12,3,7",  # année sur 2 chiffres
]


@pytest.fixture
def fake_project(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "F1_1819.csv").write_text("\n".join([HEADER] + ROWS_1819))
    (raw_dir / "F1_1920.csv").write_text("\n".join([HEADER] + ROWS_1920))
    monkeypatch.setattr(config, "RAW_DIR", raw_dir)
    monkeypatch.setattr(config, "PROCESSED_DIR", tmp_path / "processed")
    monkeypatch.setattr(config, "SEASONS", ["1819", "1920"])


def test_clean_matches(fake_project):
    clean = preprocessing.clean_matches(load_raw(), verbose=False)

    assert len(clean) == 3                      # doublon, match sans score et ligne vide supprimés
    assert clean["Date"].is_monotonic_increasing
    assert pd.api.types.is_datetime64_any_dtype(clean["Date"])
    assert clean[config.TARGET].tolist() == [1, 0, 0]
    assert "Lyon" in clean["HomeTeam"].tolist()  # espace retiré
    assert clean["FTHG"].dtype.kind == "i"       # entiers


def test_save_and_load_roundtrip(fake_project):
    clean = preprocessing.run()
    reloaded = preprocessing.load_clean()
    assert len(reloaded) == len(clean)
    assert pd.api.types.is_datetime64_any_dtype(reloaded["Date"])


def test_validate_rejects_bad_data(fake_project):
    clean = preprocessing.clean_matches(load_raw(), verbose=False)
    clean.loc[0, "FTHG"] = -1
    with pytest.raises(ValueError):
        preprocessing.validate(clean)
