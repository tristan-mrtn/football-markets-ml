"""
Rapport final de l'évaluation glissante (juillet 2024 → octobre 2026), LightGBM uniquement.

    python wf_report.py data

Règle d'équité : un modèle n'est comparé qu'au marché qui disposait de la même information.
  - modèles AVANT compositions (aucune variable liée au onze) → cotes d'OUVERTURE,
    paris à l'ouverture, jugés aussi par la valeur de clôture (CLV)
  - modèles AVEC compositions (onze connu ~1 h avant) → cotes de CLÔTURE,
    paris juste avant le coup d'envoi
Les poids des mélanges modèle + marché sont appris sur la période de tuning (2023 → juin 2024).
"""
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from footpred.models.dixon_coles import DixonColes
from footpred.evaluation.common import RESULTS, EVAL_START, END, ll_vec, market_probs, outcome

DATA = sys.argv[1]
rng = np.random.default_rng(0)
matches = pd.read_parquet(f"{DATA}/matches.parquet")
matches["id"] = matches["id"].astype(str)
matches = matches.set_index("id")
lines = []


def say(s=""):
    print(s, flush=True)
    lines.append(s)


def ci(d, n=4000):
    d = np.asarray(d)
    bs = [rng.choice(d, len(d)).mean() for _ in range(n)]
    return np.percentile(bs, [2.5, 97.5])


def read(path):
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.astype(str)
    return df


ev_p, tu_p = read(f"{RESULTS}/wf_lgb_preds.csv"), read(f"{RESULTS}/wf_lgb_tune_preds.csv")
NAMES = {"lgb_pre": "LightGBM sans compositions",
         "lgb_pre_mkt": "LightGBM sans compositions + cotes d'ouverture en variables",
         "lgb_best": "LightGBM avec compositions",
         "lgb_best_mkt": "LightGBM avec compositions + cotes d'ouverture en variables"}
pred = {k: ev_p[[f"{k}_h", f"{k}_d", f"{k}_a"]] for k in NAMES}
tune = {k: tu_p[[f"{k}_h", f"{k}_d", f"{k}_a"]] for k in NAMES}

OPEN, CLOSE = ["AvgH", "AvgD", "AvgA"], ["AvgCH", "AvgCD", "AvgCA"]
MAXO, MAXC = ["MaxH", "MaxD", "MaxA"], ["MaxCH", "MaxCD", "MaxCA"]
has_odds = matches[OPEN + CLOSE + MAXO + MAXC].notna().all(1)


def blend(Pm, Pk, w):
    z = w[0] * np.log(np.clip(Pm, 1e-9, 1)) + w[1] * np.log(np.clip(Pk, 1e-9, 1))
    z = np.exp(z - z.max(1, keepdims=True))
    return z / z.sum(1, keepdims=True)


def fit_blend(key, mkt_cols):
    ix = [i for i in tune[key].index if has_odds.get(i, False)]
    y = outcome(matches, ix)
    Pm, Pk = tune[key].loc[ix].to_numpy(), market_probs(matches, ix, mkt_cols)
    return minimize(lambda w: ll_vec(blend(Pm, Pk, w), y).mean(), [0.3, 0.7], method="Nelder-Mead").x


W_open = fit_blend("lgb_pre", OPEN)
W_close = fit_blend("lgb_best", CLOSE)
say("Mélanges appris sur la période de tuning (jamais sur l'évaluation) :")
say(f"  sans compositions + ouverture : {W_open[0]:.2f}·log(modèle) + {W_open[1]:.2f}·log(marché)")
say(f"  avec compositions + clôture   : {W_close[0]:.2f}·log(modèle) + {W_close[1]:.2f}·log(marché)")

# Dixon-Coles réentraîné chaque mois, par ligue (référence « buts seuls »)
hist = matches.reset_index().rename(columns={"home": "HomeTeam", "away": "AwayTeam", "hg": "FTHG",
                                             "ag": "FTAG", "date": "Date"})
evm = hist[(hist.Date >= EVAL_START) & (hist.Date < END)].copy()
evm["month"] = evm.Date.dt.to_period("M")
dc = {}
for (lg, mo), chunk in evm.groupby(["league", "month"]):
    start = chunk.Date.min()
    h = hist[(hist.league == lg) & (hist.Date < start) & (hist.Date > start - pd.DateOffset(years=3))]
    m = DixonColes().fit(h, ref_date=start)
    for r in chunk.itertuples():
        p = m.predict(r.HomeTeam, r.AwayTeam)
        dc[r.id] = [p["p_home"], p["p_draw"], p["p_away"]]
dc = pd.DataFrame.from_dict(dc, orient="index")

common = sorted(set.intersection(*[set(p.index) for p in pred.values()]) & set(dc.index) &
                set(matches.index[has_odds]))
y = outcome(matches, common)
P = {"Base (fréquences)": np.tile(np.bincount(y, minlength=3) / len(y), (len(y), 1)),
     "Dixon-Coles (buts seuls)": dc.loc[common].to_numpy()}
P.update({NAMES[k]: v.loc[common].to_numpy() for k, v in pred.items()})
P_open, P_close = market_probs(matches, common, OPEN), market_probs(matches, common, CLOSE)
P["Mélange sans compositions + marché d'ouverture"] = blend(P[NAMES["lgb_pre"]], P_open, W_open)
P["Mélange avec compositions + marché de clôture"] = blend(P[NAMES["lgb_best"]], P_close, W_close)
P["Bookmakers (ouverture)"], P["Bookmakers (clôture)"] = P_open, P_close

say(f"\n== Évaluation glissante : {len(common)} matchs, juillet 2024 → octobre 2026 ==")
say(f"(réentraînement tous les 2 mois sur le seul passé ; aucun réglage choisi sur cette période)\n")
say(f"{'':62}{'log-loss':>9}{'Brier':>8}")
for n, p in P.items():
    br = np.mean(np.sum((p - np.eye(3)[y]) ** 2, 1))
    say(f"{n:62}{ll_vec(p, y).mean():9.4f}{br:8.4f}")


comparisons = []


def vs(name, ref_name, label):
    d = ll_vec(P[name], y) - ll_vec(P[ref_name], y)
    lo, hi = ci(d)
    comparisons.append({"model": name, "reference": ref_name, "diff": d.mean(), "lo": lo, "hi": hi})
    verdict = "meilleur que le marché" if hi < 0 else "moins bon que le marché" if lo > 0 else "pas de différence significative"
    say(f"  {label:58} {d.mean():+.4f} [{lo:+.4f} ; {hi:+.4f}]  → {verdict}")
    return d.mean(), lo, hi


say("\n== Comparaisons équitables (écart de log-loss, négatif = mieux que le marché, IC 95 %) ==")
say("Sans compositions, face aux cotes d'ouverture :")
vs(NAMES["lgb_pre"], "Bookmakers (ouverture)", "LightGBM seul")
vs(NAMES["lgb_pre_mkt"], "Bookmakers (ouverture)", "LightGBM + cotes en variables")
vs("Mélange sans compositions + marché d'ouverture", "Bookmakers (ouverture)", "mélange modèle + marché")
say("Avec compositions, face aux cotes de clôture :")
vs(NAMES["lgb_best"], "Bookmakers (clôture)", "LightGBM seul")
vs("Mélange avec compositions + marché de clôture", "Bookmakers (clôture)", "mélange modèle + marché")
say("Apport des compositions (LightGBM avec − sans) :")
vs(NAMES["lgb_best"], NAMES["lgb_pre"], "LightGBM avec vs sans compositions")

# ------------------------------------------------------------- paris
win = np.eye(3)[y].astype(bool)
odds = {k: matches.loc[common, c].to_numpy(float) for k, c in
        (("open", OPEN), ("close", CLOSE), ("max_open", MAXO), ("max_close", MAXC))}
close_fair = 1 / P_close
bet_rows = []
say("\n== Paris simulés, 1 € par pari (ROI et CLV avec IC 95 %) ==")
say("CLV = cote obtenue / cote « juste » de clôture − 1. Une CLV moyenne > 0 est le meilleur signe d'un vrai avantage.")
plans = [(NAMES["lgb_pre"], "open", "ouverture, cote moyenne", True),
         (NAMES["lgb_pre"], "max_open", "ouverture, meilleure cote", True),
         ("Mélange sans compositions + marché d'ouverture", "open", "ouverture, cote moyenne", True),
         ("Mélange sans compositions + marché d'ouverture", "max_open", "ouverture, meilleure cote", True),
         (NAMES["lgb_best"], "close", "clôture, cote moyenne", False),
         (NAMES["lgb_best"], "max_close", "clôture, meilleure cote", False),
         ("Mélange avec compositions + marché de clôture", "max_close", "clôture, meilleure cote", False)]
for name, ok_key, label, with_clv in plans:
    o = odds[ok_key]
    for edge in (0.02, 0.05, 0.10):
        bet = (P[name] * o - 1 > edge) & ~np.isnan(o)
        if bet.sum() < 20:
            continue
        gain = np.where(win, o - 1, -1)[bet]
        g_lo, g_hi = ci(gain, 2000)
        s = f"{name[:46]:46} | {label:25} | seuil {edge:3.0%} | {bet.sum():5d} paris | ROI {gain.mean():+6.1%} [{g_lo:+.1%} ; {g_hi:+.1%}]"
        row = {"modèle": name, "cotes": label, "seuil": edge, "paris": int(bet.sum()),
               "roi": gain.mean(), "roi_bas": g_lo, "roi_haut": g_hi}
        if with_clv:
            clv = (o / close_fair - 1)[bet]
            c_lo, c_hi = ci(clv, 2000)
            s += f" | CLV {clv.mean():+5.1%} [{c_lo:+.1%} ; {c_hi:+.1%}]"
            row.update({"clv": clv.mean(), "clv_bas": c_lo, "clv_haut": c_hi})
        say(s)
        bet_rows.append(row)
pd.DataFrame(bet_rows).to_csv(f"{RESULTS}/wf_paris.csv", index=False)

say("\n== Par championnat : LightGBM sans compositions − bookmakers d'ouverture ==")
lgs = matches.loc[common, "league"].to_numpy()
for L in sorted(set(lgs)):
    k = lgs == L
    d = ll_vec(P[NAMES["lgb_pre"]][k], y[k]) - ll_vec(P_open[k], y[k])
    lo, hi = ci(d, 2000)
    say(f"  {L:12} {k.sum():5d} matchs : {d.mean():+.4f} [{lo:+.4f} ; {hi:+.4f}]")
    comparisons.append({"model": f"league:{L}", "reference": "Bookmakers (ouverture)", "diff": d.mean(), "lo": lo, "hi": hi})

pd.DataFrame([{"modèle": n, "logloss": ll_vec(p, y).mean()} for n, p in P.items()]).to_csv(f"{RESULTS}/wf_resultats.csv", index=False)
pd.DataFrame(comparisons).to_csv(f"{RESULTS}/wf_comparisons.csv", index=False)
open(f"{RESULTS}/wf_rapport.txt", "w").write("\n".join(lines))
