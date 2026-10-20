"""
Étude des inefficiences du marché des paris (aucune mise réelle : simulation à 1 € par pari).

    python market_study.py data

  A. Bookmaker isolé contre consensus (Kaunitz, Zhong & Kreiner, 2017)
  B. Biais favori-outsider (calibration des cotes par tranche de probabilité)
  C. Biais des grands clubs
  D. Zones peu suivies : début de saison, promus, fin de saison sans enjeu, par ligue
  E. Modèle + bookmaker isolé (période d'évaluation seulement)
Pour chaque stratégie : ROI et valeur de clôture (CLV), avec intervalles de confiance à 95 %.
"""
import os
import sys

import numpy as np
import pandas as pd

DATA = sys.argv[1]
rng = np.random.default_rng(0)
m = pd.read_parquet(f"{DATA}/matches.parquet")
m["id"] = m["id"].astype(str)
m = m.set_index("id")
OPEN, CLOSE = ["AvgH", "AvgD", "AvgA"], ["AvgCH", "AvgCD", "AvgCA"]
MAXO, MAXC = ["MaxH", "MaxD", "MaxA"], ["MaxCH", "MaxCD", "MaxCA"]
m = m[m[OPEN + CLOSE + MAXO + MAXC].notna().all(1) & (m[OPEN + CLOSE + MAXO + MAXC] > 1).all(1)].copy()
y = np.select([m.hg > m.ag, m.hg == m.ag], [0, 1], 2)
WIN = np.eye(3)[y].astype(bool)
lines = []


def say(s=""):
    print(s, flush=True)
    lines.append(s)


from footpred.evaluation.common import RESULTS, devig_power


def devig(cols, df=m):
    return devig_power(df[cols].to_numpy(float))   # méthode de la puissance (corrige le biais favori-outsider)


def ci(d, n=3000):
    d = np.asarray(d)
    if len(d) < 2:
        return np.nan, np.nan
    bs = [rng.choice(d, len(d)).mean() for _ in range(n)]
    return np.percentile(bs, [2.5, 97.5])


P_open, P_close = devig(OPEN), devig(CLOSE)
FAIR_CLOSE = 1 / P_close
ODDS = {"moyenne ouverture": m[OPEN].to_numpy(float), "meilleure ouverture": m[MAXO].to_numpy(float),
        "moyenne clôture": m[CLOSE].to_numpy(float), "meilleure clôture": m[MAXC].to_numpy(float)}
rows = []


def report(label, bet, odds, clv=True, mask=None):
    """bet : matrice booléenne (matchs × 3 issues) des paris pris."""
    if mask is not None:
        bet = bet & mask[:, None]
    n = int(bet.sum())
    if n < 30:
        say(f"  {label:58} {n:5d} paris (trop peu)")
        return
    gain = np.where(WIN, odds - 1, -1)[bet]
    g_lo, g_hi = ci(gain)
    s = f"  {label:58} {n:5d} paris | ROI {gain.mean():+6.1%} [{g_lo:+.1%} ; {g_hi:+.1%}]"
    row = {"stratégie": label, "paris": n, "roi": gain.mean(), "roi_bas": g_lo, "roi_haut": g_hi}
    if clv:
        c = (odds / FAIR_CLOSE - 1)[bet]
        c_lo, c_hi = ci(c)
        s += f" | CLV {c.mean():+5.1%} [{c_lo:+.1%} ; {c_hi:+.1%}]"
        row.update({"clv": c.mean(), "clv_bas": c_lo, "clv_haut": c_hi})
    say(s)
    rows.append(row)


say(f"Matchs avec cotes complètes : {len(m)} ({m.date.min():%Y-%m-%d} → {m.date.max():%Y-%m-%d})")
say(f"Marge moyenne des bookmakers : ouverture {np.mean((1 / m[OPEN]).sum(1)) - 1:.1%}, "
    f"clôture {np.mean((1 / m[CLOSE]).sum(1)) - 1:.1%}")

# ------------------------------------------------------------------ A
say("\n== A. Bookmaker isolé contre consensus ==")
say("Pari quand la meilleure cote dépasse la cote « juste » du consensus (moyenne, marge retirée) de plus de x %")
for when, cons, best in (("ouverture", P_open, ODDS["meilleure ouverture"]),
                         ("clôture", P_close, ODDS["meilleure clôture"])):
    for edge in (0.0, 0.03, 0.05, 0.08, 0.12):
        bet = cons * best - 1 > edge
        report(f"{when}, écart > {edge:.0%}", bet, best, clv=(when == "ouverture"))

# ------------------------------------------------------------------ A bis
say("\n== A bis. Piège de mesure : la CLV dépend de la façon de retirer la marge ==")
say("Mêmes paris (sélection au consensus PRORATA, meilleure cote d'ouverture, écart > 5 %), CLV mesurée de deux façons :")
_io = 1 / m[OPEN].to_numpy(float); _po = _io / _io.sum(1, keepdims=True)
_ic = 1 / m[CLOSE].to_numpy(float); _pc_prop = _ic / _ic.sum(1, keepdims=True)
_best = ODDS["meilleure ouverture"]
_bet = _po * _best - 1 > 0.05
_gain = np.where(WIN, _best - 1, -1)[_bet]
for lab, pc in (("retrait au prorata", _pc_prop), ("méthode de la puissance", P_close)):
    c = (_best * pc - 1)[_bet]
    lo, hi = ci(c)
    say(f"  CLV ({lab:24}) : {c.mean():+.1%} [{lo:+.1%} ; {hi:+.1%}]")
    rows.append({"stratégie": f"Abis CLV {lab}", "paris": int(_bet.sum()), "clv": c.mean(), "clv_bas": lo, "clv_haut": hi})
lo, hi = ci(_gain)
say(f"  rendement réel sur ces {int(_bet.sum())} paris : {_gain.mean():+.1%} [{lo:+.1%} ; {hi:+.1%}]")
say(f"  sur ces paris : fréquence de gain {WIN[_bet].mean():.1%}, proba de clôture prorata {_pc_prop[_bet].mean():.1%}, "
    f"puissance {P_close[_bet].mean():.1%}")

# ------------------------------------------------------------------ B
say("\n== B. Biais favori-outsider (cotes de clôture, marge retirée au PRORATA pour montrer le biais brut) ==")
say("  tranche de proba   issues   fréquence réelle   proba annoncée   ROI à la cote moyenne")
bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
_inv = 1 / m[CLOSE].to_numpy(float)
p, w, o = (_inv / _inv.sum(1, keepdims=True)).ravel(), WIN.ravel(), ODDS["moyenne clôture"].ravel()
for a, b in zip(bins, bins[1:]):
    k = (p >= a) & (p < b)
    if k.sum() < 50:
        continue
    roi = np.where(w[k], o[k] - 1, -1)
    lo, hi = ci(roi)
    say(f"  {a:.1f}-{b:.1f}           {k.sum():6d}     {w[k].mean():6.1%}            {p[k].mean():6.1%}"
        f"          {roi.mean():+6.1%} [{lo:+.1%} ; {hi:+.1%}]")
    rows.append({"stratégie": f"B tranche {a:.1f}-{b:.1f}", "paris": int(k.sum()), "roi": roi.mean(),
                 "roi_bas": lo, "roi_haut": hi, "freq": w[k].mean(), "proba": p[k].mean()})

# ------------------------------------------------------------------ C
say("\n== C. Biais des grands clubs (pari systématique, cote moyenne de clôture) ==")
BIG = {"Real Madrid", "Barcelona", "Atletico Madrid", "Paris Saint Germain", "Bayern Munich",
       "Borussia Dortmund", "Manchester City", "Manchester United", "Liverpool", "Arsenal", "Chelsea",
       "Tottenham", "Juventus", "AC Milan", "Inter", "Napoli", "Marseille"}
home_big, away_big = m.home.isin(BIG).to_numpy(), m.away.isin(BIG).to_numpy()
oc = ODDS["moyenne clôture"]
on_big = np.zeros_like(WIN); on_big[home_big & ~away_big, 0] = True; on_big[away_big & ~home_big, 2] = True
against = np.zeros_like(WIN); against[home_big & ~away_big, 2] = True; against[away_big & ~home_big, 0] = True
report("miser sur le grand club", on_big, oc, clv=False)
report("miser contre le grand club (victoire de l'adversaire)", against, oc, clv=False)
draw_big = np.zeros_like(WIN); draw_big[home_big ^ away_big, 1] = True
report("miser sur le nul quand un grand club joue", draw_big, oc, clv=False)

# ------------------------------------------------------------------ D
say("\n== D. Zones peu suivies : précision des cotes d'ouverture vs clôture ==")
say("(log-loss du marché ; plus l'écart ouverture − clôture est grand, plus l'ouverture était « fausse »)")
ll = lambda P: -np.log(P[np.arange(len(y)), y])
lo_, lc_ = ll(P_open), ll(P_close)
feats = pd.read_parquet(f"{DATA}/features.parquet")
fh = feats[feats.is_home == 1].assign(match_id=lambda d: d.match_id.astype(str)).set_index("match_id")
fh = fh.reindex(m.index)
segments = {
    "début de saison (5 premières journées)": (fh.att_played < 5).to_numpy(),
    "reste de la saison": (fh.att_played >= 5).to_numpy(),
    "match avec un promu (moins de 10 matchs dans la ligue)": ((fh.att_n_played < 10) | (fh.def_n_played < 10)).to_numpy(),
    "fin de saison, une équipe sans enjeu": ((fh.att_nothing_to_play == 1) | (fh.def_nothing_to_play == 1)).to_numpy(),
}
for L in sorted(m.league.unique()):
    segments[f"ligue : {L}"] = (m.league == L).to_numpy()
for name, k in segments.items():
    k = np.nan_to_num(k).astype(bool)
    if k.sum() < 50:
        continue
    d = lo_[k] - lc_[k]
    lo, hi = ci(d)
    say(f"  {name:55} {k.sum():5d} matchs | ouverture {lo_[k].mean():.4f} clôture {lc_[k].mean():.4f} "
        f"| écart {d.mean():+.4f} [{lo:+.4f} ; {hi:+.4f}]")
    rows.append({"stratégie": f"D {name}", "paris": int(k.sum()), "ll_open": lo_[k].mean(),
                 "ll_close": lc_[k].mean(), "diff": d.mean(), "diff_lo": lo, "diff_hi": hi})
say("\n  Bookmaker isolé (ouverture, écart > 5 %) par segment :")
bet5 = P_open * ODDS["meilleure ouverture"] - 1 > 0.05
for name, k in segments.items():
    report(name, bet5, ODDS["meilleure ouverture"], mask=np.nan_to_num(k).astype(bool))

# ------------------------------------------------------------------ E
if os.path.exists(f"{RESULTS}/wf_lgb_preds.csv"):
    say("\n== E. Modèle sans compositions + bookmaker isolé (évaluation juillet 2024 → octobre 2026) ==")
    pr = pd.read_csv(f"{RESULTS}/wf_lgb_preds.csv", index_col=0)
    pr.index = pr.index.astype(str)
    ix = m.index.isin(pr.index)
    Pm = np.full_like(P_open, np.nan)
    Pm[ix] = pr.loc[m.index[ix], ["lgb_pre_h", "lgb_pre_d", "lgb_pre_a"]].to_numpy()
    best = ODDS["meilleure ouverture"]
    for edge in (0.03, 0.05, 0.08):
        report(f"bookmaker isolé seul, écart > {edge:.0%} (même période)", (P_open * best - 1 > edge), best, mask=ix)
        report(f"modèle seul, avantage > {edge:.0%}", np.nan_to_num(Pm) * best - 1 > edge, best, mask=ix)
        both = (P_open * best - 1 > edge) & (np.nan_to_num(Pm) * best - 1 > edge)
        report(f"les deux d'accord, > {edge:.0%}", both, best, mask=ix)

pd.DataFrame(rows).to_csv(f"{RESULTS}/market_study.csv", index=False)
open(f"{RESULTS}/market_study.txt", "w").write("\n".join(lines))
