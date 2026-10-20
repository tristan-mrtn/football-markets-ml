"""Outils partagés de l'évaluation glissante (walk-forward)."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import poisson

TRAIN_START = pd.Timestamp("2022-01-01")
TUNE_START, EVAL_START, END = pd.Timestamp("2023-01-01"), pd.Timestamp("2024-07-01"), pd.Timestamp("2026-10-21")
G = np.arange(11)
ID_COLS = ("match_id", "date", "league", "season", "team", "opp", "goals", "form_att", "form_def")


def folds(start, end, months):
    """Fenêtres [début, fin[ de `months` mois ; on réentraîne au début de chacune."""
    out, d = [], start
    while d < end:
        e = min(d + pd.DateOffset(months=months), end)
        out.append((d, e))
        d = e
    return out


def load(data):
    feats = pd.read_parquet(f"{data}/features.parquet")
    matches = pd.read_parquet(f"{data}/matches.parquet").set_index("id")
    for c in ("form_att", "form_def", "league"):
        feats[c + "_cat"] = feats[c].astype("category")
    feats["ok"] = (feats["att_n_played"] >= 3) & (feats["def_n_played"] >= 3)
    pure = [c for c in feats.columns if c not in ID_COLS and c != "ok" and not c.startswith("mkt_")
            and feats[c].dtype.kind in "fiubc" or c.endswith("_cat")]
    pure = [c for c in dict.fromkeys(pure) if not c.startswith("mkt_") and c not in ID_COLS and c != "ok"]
    return feats, matches, pure


def score_probs(lh, la, rho):
    lh, la = np.asarray(lh), np.asarray(la)
    out = np.zeros((len(lh), 3))
    for k, (a, b) in enumerate(zip(lh, la)):
        M = np.outer(poisson.pmf(G, a), poisson.pmf(G, b))
        M[0, 0] *= 1 - a * b * rho; M[0, 1] *= 1 + a * rho
        M[1, 0] *= 1 + b * rho;     M[1, 1] *= 1 - rho
        M /= M.sum()
        out[k] = [np.tril(M, -1).sum(), np.trace(M), np.triu(M, 1).sum()]
    return out


def to_matches(df, lam):
    d = df[["match_id", "is_home"]].assign(lam=lam)
    w = d.pivot(index="match_id", columns="is_home", values="lam")
    return pd.DataFrame({"lam_h": w[1], "lam_a": w[0]}).dropna()


def outcome(matches, ids):
    hg, ag = matches.loc[ids, "hg"].to_numpy(), matches.loc[ids, "ag"].to_numpy()
    return np.select([hg > ag, hg == ag], [0, 1], 2)


def ll_vec(P, y):
    return -np.log(np.clip(np.asarray(P)[np.arange(len(y)), y], 1e-12, None))


def fit_rho(lm, y):
    return minimize_scalar(lambda r: ll_vec(score_probs(lm.lam_h, lm.lam_a, r), y).mean(),
                           bounds=(-0.3, 0.3), method="bounded").x


def market_probs(matches, ids, cols):
    """Probabilités du marché, marge retirée par la méthode de la puissance (voir devig_power)."""
    return devig_power(matches.loc[ids, cols].astype(float).to_numpy())


def devig_power(odds):
    """Retire la marge par la méthode de la puissance : p_i = (1/o_i)^k avec somme = 1.
    Plus de marge est retirée sur les outsiders, ce qui corrige le biais favori-outsider
    (le retrait au prorata surestime les chances des outsiders)."""
    from scipy.optimize import brentq
    inv = 1 / np.asarray(odds, float)
    out = np.empty_like(inv)
    for i, row in enumerate(inv):
        k = brentq(lambda k: (row ** k).sum() - 1, 0.5, 3.0)
        out[i] = row ** k
    return out


import os as _os

RESULTS = _os.environ.get("FOOTPRED_RESULTS", "results")   # dossier des sorties
_os.makedirs(RESULTS, exist_ok=True)
