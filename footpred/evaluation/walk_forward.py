"""
LightGBM en évaluation glissante.

    python wf_lgb.py data tune  N   # recherche aléatoire de N réglages (période de tuning seulement)
    python wf_lgb.py data eval      # meilleur réglage + réglage d'origine sur la période d'évaluation

Réentraînement tous les 2 mois sur tout le passé disponible.
"""
import json
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from footpred.evaluation.common import (RESULTS, EVAL_START, END, TRAIN_START, TUNE_START, fit_rho, folds, load, ll_vec,
                       outcome, score_probs, to_matches)

DATA, MODE = sys.argv[1], sys.argv[2]
feats, matches, PURE = load(DATA)
MKT = PURE + ["mkt_win", "mkt_draw", "mkt_lose"]
# variables qui dépendent du onze de départ (connu ~1 h avant le match seulement)
LINEUP_KEYS = ("xi_", "chem_", "missing", "bench_off", "continuity", "def_line", "gk_prev", "gk_new",
               "n_missing_regulars", "formation_change", "form_")
PRE = [c for c in PURE if not any(k in c for k in LINEUP_KEYS)]
PRE_MKT = PRE + ["mkt_win", "mkt_draw", "mkt_lose"]
BASE = dict(objective="poisson", verbose=-1, num_threads=2)
DEFAULT = dict(learning_rate=0.02, num_leaves=15, min_data_in_leaf=100, feature_fraction=0.7,
               bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, n_iter=480)


def walk(params, start, end, cols, seeds=(0,)):
    """Prédit chaque fenêtre de 2 mois avec un modèle entraîné sur tout ce qui précède."""
    p = dict(BASE, **{k: v for k, v in params.items() if k != "n_iter"})
    out = []
    for a, b in folds(start, end, 2):
        tr = feats[feats.ok & (feats.date >= TRAIN_START) & (feats.date < a)]
        te = feats[(feats.date >= a) & (feats.date < b)]
        # garde-fous anti-fuite : on n'apprend que sur des matchs terminés avant la fenêtre
        assert tr.date.max() < a <= te.date.min(), "fuite temporelle dans le découpage"
        assert not (set(tr.match_id) & set(te.match_id))
        if te.empty:
            continue
        lam = np.zeros(len(te))
        for s in seeds:
            m = lgb.train(dict(p, seed=s), lgb.Dataset(tr[cols], tr["goals"]), params["n_iter"])
            lam += m.predict(te[cols]) / len(seeds)
        out.append(to_matches(te, lam))
    return pd.concat(out)


def evaluate(lm):
    y = outcome(matches, lm.index)
    rho = fit_rho(lm, y)
    return ll_vec(score_probs(lm.lam_h, lm.lam_a, rho), y).mean(), rho


def sample(rng):
    return dict(learning_rate=float(rng.choice([0.01, 0.02, 0.04])),
                num_leaves=int(rng.choice([4, 7, 15, 31])),
                min_data_in_leaf=int(rng.choice([50, 100, 200, 400, 800])),
                feature_fraction=float(rng.choice([0.2, 0.35, 0.5, 0.7, 0.9])),
                bagging_fraction=float(rng.choice([0.6, 0.8, 1.0])), bagging_freq=1,
                lambda_l2=float(rng.choice([0, 1, 5, 20, 100])),
                min_gain_to_split=float(rng.choice([0, 0.01, 0.1])),
                n_iter=int(rng.choice([150, 300, 500, 800, 1200, 2000])))


if MODE == "tune":
    n = int(sys.argv[3])
    rng = np.random.default_rng(42)
    trials = []
    cands = [DEFAULT] + [sample(rng) for _ in range(n - 1)]
    for i, prm in enumerate(cands):
        t0 = time.time()
        ll, rho = evaluate(walk(prm, TUNE_START, EVAL_START, PURE))
        trials.append({**prm, "logloss": ll, "rho": rho})
        best = min(t["logloss"] for t in trials)
        print(f"essai {i + 1:3d}/{n} : log-loss {ll:.4f} (meilleur {best:.4f}) "
              f"[{time.time() - t0:.0f}s]", flush=True)
        pd.DataFrame(trials).to_csv(f"{RESULTS}/wf_lgb_trials.csv", index=False)
    best = min(trials, key=lambda t: t["logloss"])
    json.dump({k: v for k, v in best.items() if k not in ("logloss", "rho")},
              open(f"{RESULTS}/wf_lgb_best.json", "w"), indent=1)
    print("meilleur réglage :", best, flush=True)

elif MODE == "eval":
    best = json.load(open(f"{RESULTS}/wf_lgb_best.json"))
    res, tune_res = {}, {}
    print(f"{len(PURE)} variables, dont {len(PURE) - len(PRE)} liées au onze de départ", flush=True)
    for name, prm, cols in (("lgb_best", best, PURE), ("lgb_best_mkt", best, MKT),
                            ("lgb_pre", best, PRE), ("lgb_pre_mkt", best, PRE_MKT)):
        # rho calé sur la période de tuning, puis appliqué tel quel à l'évaluation
        lm_t = walk(prm, TUNE_START, EVAL_START, cols)
        _, rho = evaluate(lm_t)
        Pt = score_probs(lm_t.lam_h, lm_t.lam_a, rho)
        tune_res[name] = pd.DataFrame(Pt, index=lm_t.index, columns=[f"{name}_h", f"{name}_d", f"{name}_a"])
        lm = walk(prm, EVAL_START, END, cols, seeds=(0, 1, 2))
        P = score_probs(lm.lam_h, lm.lam_a, rho)
        res[name] = pd.DataFrame(P, index=lm.index, columns=[f"{name}_h", f"{name}_d", f"{name}_a"])
        res[name][f"{name}_lam_h"], res[name][f"{name}_lam_a"] = lm.lam_h, lm.lam_a
        y = outcome(matches, lm.index)
        print(f"{name} : {len(lm)} matchs, log-loss {ll_vec(P, y).mean():.4f} (rho {rho:+.3f})", flush=True)
    pd.concat(res.values(), axis=1).to_csv(f"{RESULTS}/wf_lgb_preds.csv")
    pd.concat(tune_res.values(), axis=1).to_csv(f"{RESULTS}/wf_lgb_tune_preds.csv")
    print("wf_lgb_preds.csv écrit", flush=True)
