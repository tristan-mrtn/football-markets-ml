"""
Feature-family ablation inside the walk-forward protocol, on the TUNING period only
(Jan 2023 → Jun 2024), so that it never touches the evaluation period.

    python -m footpred.evaluation.ablation data

For each family F (market odds excluded):
  - "add"  : team strength (A) + F          vs  A alone
  - "drop" : all pure features minus F      vs  all pure features
Score: 1X2 log-loss per match, paired bootstrap 95 % CI on the difference.
"""
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

from footpred.evaluation.common import (RESULTS, EVAL_START, TRAIN_START, TUNE_START, fit_rho, folds, ll_vec,
                                        outcome, score_probs, to_matches)
from footpred.features.families import assign

DATA = sys.argv[1] if len(sys.argv) > 1 else "data"
feats = pd.read_parquet(f"{DATA}/features.parquet")
matches = pd.read_parquet(f"{DATA}/matches.parquet").set_index("id")
for c in ("form_att", "form_def", "league"):
    feats[c + "_cat"] = feats[c].astype("category")
feats["ok"] = (feats["att_n_played"] >= 3) & (feats["def_n_played"] >= 3)
fam = assign(feats.columns)
fam = {k: v for k, v in fam.items() if not k.startswith("M") and v}
ALL = sum(fam.values(), [])
PARAMS = dict(objective="poisson", verbose=-1, num_threads=2, learning_rate=0.02, num_leaves=15,
              min_data_in_leaf=100, feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1,
              lambda_l2=5.0, seed=0)
N_ITER = 480
rng = np.random.default_rng(0)


def per_match_logloss(cols):
    out = []
    for a, b in folds(TUNE_START, EVAL_START, 2):
        tr = feats[feats.ok & (feats.date >= TRAIN_START) & (feats.date < a)]
        te = feats[(feats.date >= a) & (feats.date < b)]
        assert tr.date.max() < a <= te.date.min()
        m = lgb.train(PARAMS, lgb.Dataset(tr[cols], tr["goals"]), N_ITER)
        out.append(to_matches(te, m.predict(te[cols])))
    lm = pd.concat(out)
    y = outcome(matches, lm.index)
    rho = fit_rho(lm, y)
    return pd.Series(ll_vec(score_probs(lm.lam_h, lm.lam_a, rho), y), index=lm.index)


def diff_ci(a, b):
    d = (a - b).dropna().to_numpy()
    bs = [rng.choice(d, len(d)).mean() for _ in range(2000)]
    return d.mean(), *np.percentile(bs, [2.5, 97.5])


base = per_match_logloss(fam["A team strength"])
full = per_match_logloss(ALL)
print(f"A alone : {base.mean():.4f}   all pure features : {full.mean():.4f}   ({len(base)} matches)", flush=True)
rows = []
for f, cols in fam.items():
    if f.startswith("A"):
        continue
    add = per_match_logloss(fam["A team strength"] + cols)
    drop = per_match_logloss([c for c in ALL if c not in cols])
    g_add = diff_ci(base, add)      # > 0 : adding F helps
    g_drop = diff_ci(drop, full)    # > 0 : removing F hurts (F carries unique information)
    rows.append({"family": f, "n_features": len(cols),
                 "gain_add": g_add[0], "gain_add_lo": g_add[1], "gain_add_hi": g_add[2],
                 "gain_unique": g_drop[0], "gain_unique_lo": g_drop[1], "gain_unique_hi": g_drop[2]})
    print(f"{f:28} ({len(cols):3d})  added to A: {g_add[0]:+.4f} [{g_add[1]:+.4f}; {g_add[2]:+.4f}]   "
          f"unique (all − F): {g_drop[0]:+.4f} [{g_drop[1]:+.4f}; {g_drop[2]:+.4f}]", flush=True)
pd.DataFrame(rows).to_csv(f"{RESULTS}/ablation_families.csv", index=False)
