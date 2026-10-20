"""Unit tests for the mathematical building blocks (no data download needed)."""
import numpy as np
import pandas as pd
import pytest

from footpred.evaluation.common import devig_power, folds, ll_vec, score_probs
from footpred.features.build import formation, game_state
from footpred.features.families import FAMILIES, LINEUP_KEYS, assign
from footpred.models.dixon_coles import DixonColes


# ----------------------------------------------------------------- de-vigging
def test_devig_power_sums_to_one():
    odds = np.array([[1.50, 4.20, 6.50], [2.40, 3.30, 3.10], [11.0, 6.0, 1.25]])
    p = devig_power(odds)
    assert np.allclose(p.sum(1), 1)
    assert (p > 0).all()


def test_devig_power_is_exact_without_margin():
    fair = np.array([[0.5, 0.3, 0.2]])
    assert np.allclose(devig_power(1 / fair), fair)


def test_devig_power_shrinks_longshots_more_than_proportional():
    """The power method removes more margin from longshots (favourite-longshot bias)."""
    odds = np.array([[1.30, 5.50, 11.0]])
    prop = (1 / odds) / (1 / odds).sum()
    power = devig_power(odds)
    assert power[0, 2] < prop[0, 2]     # longshot: lower probability
    assert power[0, 0] > prop[0, 0]     # favourite: higher probability


# ------------------------------------------------------------- score model
@pytest.mark.parametrize("rho", [-0.1, 0.0, 0.1])
def test_score_probs_are_distributions(rho):
    P = score_probs([1.6, 0.9, 2.5], [1.1, 0.9, 0.4], rho)
    assert np.allclose(P.sum(1), 1)
    assert (P >= 0).all()


def test_score_probs_symmetry():
    P = score_probs([1.3], [1.3], -0.1)[0]
    assert P[0] == pytest.approx(P[2])


def test_score_probs_stronger_side_more_likely_to_win():
    P = score_probs([2.2], [0.8], 0.0)[0]
    assert P[0] > P[2]


def test_log_loss_known_value():
    P = np.array([[0.5, 0.25, 0.25]])
    assert ll_vec(P, np.array([0]))[0] == pytest.approx(np.log(2))


# --------------------------------------------------------- walk-forward folds
def test_folds_are_contiguous_and_cover_the_range():
    a, b = pd.Timestamp("2023-01-01"), pd.Timestamp("2024-07-01")
    f = folds(a, b, 2)
    assert f[0][0] == a and f[-1][1] == b
    assert all(f[i][1] == f[i + 1][0] for i in range(len(f) - 1))
    assert all(x < y for x, y in f)


# ------------------------------------------------------------- features
def xi(*positions):
    return [{"pos": "GK"}] + [{"pos": p} for p in positions]


@pytest.mark.parametrize("positions, expected", [
    (("DR", "DC", "DC", "DL", "MC", "MC", "MC", "FWR", "FW", "FWL"), "4-3-3"),
    (("DR", "DC", "DC", "DL", "DMC", "DMC", "AMR", "AMC", "AML", "FW"), "4-2-3-1"),
    (("DC", "DC", "DC", "DMR", "DML", "MC", "MC", "MC", "FW", "FW"), "3-5-2"),
    (("DR", "DC", "DC", "DL", "MR", "MC", "MC", "ML", "FW", "FW"), "4-4-2"),
])
def test_formation_detection(positions, expected):
    assert formation(xi(*positions)) == expected


def shot(minute, xg, res="MissedShot"):
    return {"min": minute, "xg": xg, "res": res}


def test_game_state_splits_xg_by_score():
    # home scores at 30', away equalises at 60'
    home = [shot(10, 0.2), shot(30, 0.5, "Goal"), shot(45, 0.3)]
    away = [shot(50, 0.4), shot(60, 0.6, "Goal"), shot(80, 0.1)]
    xg, min_level, _ = game_state(home, away)
    # level periods: 0-30' and 60'-end -> home xG at level = 0.2 + 0.5 (the goal itself is taken at 0-0)
    assert xg["h"][0] == pytest.approx(0.7)
    # away at level: the 80' shot only (the 60' goal was taken while trailing)
    assert xg["a"][0] == pytest.approx(0.1)
    assert min_level == pytest.approx(30 + (95 - 60))


def test_every_feature_in_exactly_one_family_and_odds_isolated():
    cols = ["mkt_win", "mkt_draw", "att_xgf", "att_draw", "att_xi_off", "att_chem_xi", "h2h_xg",
            "att_level_xgf90", "att_xgf_trend", "form_att_cat", "travel_km", "att_shots_f"]
    fam = assign(cols)
    flat = [c for v in fam.values() for c in v]
    assert sorted(flat) == sorted(cols)                       # each column exactly once
    assert set(fam["M market odds (opening)"]) == {"mkt_win", "mkt_draw"}
    assert "att_draw" in fam["J luck"]                        # regression test: not swallowed by odds
    assert list(FAMILIES)[0].startswith("M")                  # odds matched first


def test_lineup_keys_cover_lineup_features():
    for c in ("att_xi_off", "def_chem_def", "att_missing_off", "att_gk_prev", "form_att_cat"):
        assert any(k in c for k in LINEUP_KEYS)
    for c in ("att_xgf", "elo_diff", "att_ppda", "att_euro_next_days"):
        assert not any(k in c for k in LINEUP_KEYS)


# ------------------------------------------------------------- Dixon-Coles
def test_dixon_coles_recovers_stronger_team():
    rng = np.random.default_rng(0)
    teams = ["A", "B", "C", "D"]
    strength = {"A": 0.5, "B": 0.1, "C": -0.1, "D": -0.5}
    rows, date = [], pd.Timestamp("2024-01-01")
    for k in range(60):
        for h in teams:
            for a in teams:
                if h != a:
                    rows.append({"Date": date + pd.Timedelta(days=k), "HomeTeam": h, "AwayTeam": a,
                                 "FTHG": rng.poisson(np.exp(0.2 + 0.25 + strength[h] - strength[a] / 2)),
                                 "FTAG": rng.poisson(np.exp(0.2 + strength[a] - strength[h] / 2))})
    m = DixonColes(xi=0.0, l2=0.01).fit(pd.DataFrame(rows))
    p = m.predict("A", "D")
    assert p["p_home"] > 0.6
    assert m.att["A"] > m.att["D"]
