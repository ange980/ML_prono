"""Tests du feature engineering sur un mini-championnat de 4 équipes."""
import numpy as np
import pandas as pd
import pytest

from src import config, features

WINDOW = 5
TEAMS = ["A", "B", "C", "D"]


def make_matches(n_rounds: int = 8, seed: int = 0) -> pd.DataFrame:
    """4 équipes, 2 matchs par journée (A-B et C-D), alternance domicile/extérieur."""
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(n_rounds):
        pairs = [("A", "B"), ("C", "D")] if r % 2 == 0 else [("B", "A"), ("D", "C")]
        for home, away in pairs:
            hg, ag = int(rng.integers(0, 4)), int(rng.integers(0, 4))
            rows.append({
                "Season": "1819", "Date": pd.Timestamp("2018-08-01") + pd.Timedelta(days=7 * r),
                "HomeTeam": home, "AwayTeam": away, "FTHG": hg, "FTAG": ag,
                config.TARGET: int(hg > ag),
            })
    return pd.DataFrame(rows)


def team_history(matches: pd.DataFrame, team: str) -> pd.DataFrame:
    """Calcule à la main (boucle simple) pts / gf / ga d'une équipe, match après match."""
    rows = []
    for _, m in matches.sort_values("Date").iterrows():
        if m["HomeTeam"] == team:
            gf, ga = m["FTHG"], m["FTAG"]
        elif m["AwayTeam"] == team:
            gf, ga = m["FTAG"], m["FTHG"]
        else:
            continue
        pts = 3 if gf > ga else (1 if gf == ga else 0)
        rows.append({"Date": m["Date"], "pts": pts, "gf": gf, "ga": ga})
    return pd.DataFrame(rows)


def test_early_matches_are_dropped():
    feats = features.build_features(make_matches(), window=WINDOW)
    # 8 journées, 5 matchs d'historique requis -> seules les journées 6, 7, 8 restent
    assert len(feats) == 3 * 2
    assert not feats[features.feature_names(WINDOW)].isna().any().any()


def test_form_matches_manual_computation():
    matches = make_matches()
    feats = features.build_features(matches, window=WINDOW)
    last_date = matches["Date"].max()
    hist = team_history(matches, "A")

    # Moyenne des 5 matchs PRÉCÉDENTS le dernier match de A (le dernier match exclu)
    expected = hist.iloc[-1 - WINDOW:-1][["pts", "gf", "ga"]].mean()

    row = feats[(feats["Date"] == last_date) & ((feats["HomeTeam"] == "A") | (feats["AwayTeam"] == "A"))].iloc[0]
    side = "home" if row["HomeTeam"] == "A" else "away"
    for stat in ["pts", "gf", "ga"]:
        assert row[f"{side}_{stat}_last{WINDOW}"] == pytest.approx(expected[stat])


def test_no_leakage_from_current_match():
    """Modifier le score d'un match ne doit PAS changer ses propres features."""
    matches = make_matches()
    before = features.build_features(matches, window=WINDOW)

    altered = matches.copy()
    last = altered["Date"] == altered["Date"].max()
    altered.loc[last, ["FTHG", "FTAG"]] = [9, 0]  # scores très différents
    after = features.build_features(altered, window=WINDOW)

    names = features.feature_names(WINDOW)
    pd.testing.assert_frame_equal(before[names], after[names])


def test_features_do_not_contain_match_outcome():
    assert not ({"FTHG", "FTAG", "FTR"} & set(features.FEATURES))
