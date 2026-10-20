"""
Transformer sur les compositions, en évaluation glissante (réentraînement tous les 3 mois).

    python wf_nn.py data tune     # quelques configurations, période de tuning seulement
    python wf_nn.py data eval     # meilleure configuration sur la période d'évaluation

À chaque réentraînement : vocabulaire des joueurs, normalisation et arrêt anticipé
sont calculés uniquement avec le passé (les 3 derniers mois du passé servent d'arrêt anticipé).
"""
import json
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from footpred.evaluation.common import (RESULTS, EVAL_START, END, TRAIN_START, TUNE_START, fit_rho, folds, ll_vec, outcome,
                       score_probs, to_matches)

DATA, MODE = sys.argv[1], sys.argv[2]
torch.set_num_threads(2)
import os
SMOKE = bool(os.environ.get("SMOKE"))

feats = pd.read_parquet(f"{DATA}/features.parquet")
matches = pd.read_parquet(f"{DATA}/matches.parquet").set_index("id")
tok = pd.read_parquet(f"{DATA}/player_tokens.parquet")
feats["ok"] = (feats["att_n_played"] >= 3) & (feats["def_n_played"] >= 3)

# ---------------------------------------------------------- jetons -> tableaux
TOK_NUM = ["off", "chain", "buildup", "n_apps", "chem_xi", "days_since", "missed_streak",
           "absence_rate_6m", "load_30d", "back_from_red", "on_xga", "on_xgf", "gk_q", "age"]
tok["n_apps"] = np.log1p(tok["n_apps"])
tok["days_since"] = np.log1p(tok["days_since"].clip(upper=365))
tok["missed_streak"] = np.log1p(tok["missed_streak"].clip(upper=20))
tok["load_30d"] = tok["load_30d"] / 90
for c in ("days_since", "missed_streak", "absence_rate_6m", "age"):
    tok[c + "_nan"] = tok[c].isna().astype(float)
    tok[c] = tok[c].fillna(0)
TOK_NUM += ["days_since_nan", "missed_streak_nan", "absence_rate_6m_nan", "age_nan"]
pid_codes, pid_uni = pd.factorize(tok["pid"])
tok["pid_g"] = pid_codes + 1                      # 0 = vide
pos_vocab = {p: i for i, p in enumerate(sorted(tok["pos"].unique()))}
tok["pos_i"] = tok["pos"].map(pos_vocab)

LINEUP = ("xi_", "chem_", "missing", "bench_off", "continuity", "def_line", "gk_", "n_missing",
          "formation_change")
MATCH_NUM = [c for c in feats.columns if feats[c].dtype.kind in "fiub" and c not in ("goals", "season", "ok")
             and not c.startswith("mkt_") and not any(k in c for k in LINEUP)]
LEAGUES = sorted(feats["league"].unique())

N, F = len(feats), len(TOK_NUM)
idx_of = {(m, s): i for i, (m, s) in enumerate(zip(feats.match_id, feats.is_home))}
P = {r: np.zeros((N, 11), np.int64) for r in ("att", "def")}
Q = {r: np.zeros((N, 11), np.int64) for r in ("att", "def")}
X = {r: np.zeros((N, 11, F), np.float32) for r in ("att", "def")}
M = {r: np.ones((N, 11), bool) for r in ("att", "def")}
for (mid, side), g in tok.sort_values("slot").groupby(["match_id", "side"]):
    g = g.iloc[:11]
    n = len(g)
    is_home_side = 1 if side == "h" else 0
    for role, row_home in (("att", is_home_side), ("def", 1 - is_home_side)):
        i = idx_of.get((mid, row_home))
        if i is None:
            continue
        P[role][i, :n] = g.pid_g; Q[role][i, :n] = g.pos_i
        X[role][i, :n] = g[TOK_NUM].to_numpy(np.float32); M[role][i, :n] = False
for r in ("att", "def"):
    M[r][M[r].all(1), 0] = False
LG = feats.league.map({l: i for i, l in enumerate(LEAGUES)}).to_numpy()
Y = feats.goals.to_numpy(np.float32)


# -------------------------------------------------------------- modèle
class LineupNet(nn.Module):
    def __init__(self, n_players, n_match, d, emb, layers, drop):
        super().__init__()
        self.emb = nn.Embedding(n_players, emb)
        self.pos = nn.Embedding(len(pos_vocab), 8)
        self.inp = nn.Linear(emb + 8 + F, d)
        self.role = nn.Embedding(2, d)
        layer = nn.TransformerEncoderLayer(d, 4, 2 * d, dropout=drop, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.lg = nn.Embedding(len(LEAGUES), 4)
        self.head = nn.Sequential(nn.Linear(2 * d + n_match + 4, 64), nn.GELU(), nn.Dropout(drop),
                                  nn.Linear(64, 1))

    def forward(self, pa, qa, xa, ma, pd_, qd, xd, md, lg, num):
        ha = self.inp(torch.cat([self.emb(pa), self.pos(qa), xa], -1)) + self.role.weight[0]
        hd = self.inp(torch.cat([self.emb(pd_), self.pos(qd), xd], -1)) + self.role.weight[1]
        mask = torch.cat([ma, md], 1)
        h = self.enc(torch.cat([ha, hd], 1), src_key_padding_mask=mask)
        keep = (~mask).float().unsqueeze(-1)
        pool = lambda z, k: (z * k).sum(1) / k.sum(1).clamp(min=1)
        return self.head(torch.cat([pool(h[:, :11], keep[:, :11]), pool(h[:, 11:], keep[:, 11:]),
                                    num, self.lg(lg)], -1)).squeeze(-1)


def fit_predict(train_idx, test_idx, cfg, seed):
    """Entraîne sur train_idx (arrêt anticipé sur ses 3 derniers mois) et prédit test_idx."""
    torch.manual_seed(seed); np.random.seed(seed)
    dates = feats.date.to_numpy()
    cut = dates[train_idx].max() - np.timedelta64(91, "D")
    es_idx = train_idx[dates[train_idx] > cut]
    fit_idx = train_idx[dates[train_idx] <= cut]

    # vocabulaire : joueurs avec assez d'apparitions dans la partie d'entraînement
    cnt = np.bincount(np.concatenate([P["att"][fit_idx].ravel(), P["def"][fit_idx].ravel()]),
                      minlength=len(pid_uni) + 1)
    cnt[0] = 0
    keep = cnt >= cfg["min_apps"] * 2           # chaque apparition compte 2 fois (att + déf)
    lut = np.zeros(len(cnt), np.int64)
    lut[keep] = np.arange(1, keep.sum() + 1)

    mu = X["att"][fit_idx][~M["att"][fit_idx]].mean(0)
    sd = X["att"][fit_idx][~M["att"][fit_idx]].std(0) + 1e-6
    num = feats[MATCH_NUM].to_numpy(np.float32)
    med = np.nanmedian(num[fit_idx], 0)
    med = np.where(np.isnan(med), 0, med)
    num = np.where(np.isnan(num), med, num)
    nmu, nsd = num[fit_idx].mean(0), num[fit_idx].std(0) + 1e-6

    def batch(ix, drop_players=0.0):
        t = lambda a: torch.tensor(a)
        pa, pd_ = lut[P["att"][ix]], lut[P["def"][ix]]
        if drop_players:
            pa = np.where(np.random.rand(*pa.shape) < drop_players, 0, pa)
            pd_ = np.where(np.random.rand(*pd_.shape) < drop_players, 0, pd_)
        return (t(pa), t(Q["att"][ix]), t((X["att"][ix] - mu) / sd), t(M["att"][ix]),
                t(pd_), t(Q["def"][ix]), t((X["def"][ix] - mu) / sd), t(M["def"][ix]),
                t(LG[ix]), t((num[ix] - nmu) / nsd))

    net = LineupNet(keep.sum() + 1, len(MATCH_NUM), cfg["d"], cfg["emb"], cfg["layers"], cfg["drop"])
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    loss_f = lambda ll, y: (torch.exp(ll) - y * ll).mean()
    es_b, es_y = batch(es_idx), torch.tensor(Y[es_idx])
    best, state, wait = 1e9, None, 0
    for ep in range(1 if SMOKE else 60):
        net.train()
        perm = np.random.permutation(fit_idx)
        for i in range(0, len(perm), 256):
            ix = perm[i:i + 256]
            loss = loss_f(net(*batch(ix, cfg["pdrop"])), torch.tensor(Y[ix]))
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            v = loss_f(net(*es_b), es_y).item()
        if v < best - 1e-4:
            best, state, wait = v, {k: x.clone() for k, x in net.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= 6:
                break
    # réentraîne sur TOUT le passé (y compris les 3 derniers mois) avec le même nombre d'époques
    n_ep = max(ep - wait + 1, 1)
    torch.manual_seed(seed + 100)
    net = LineupNet(keep.sum() + 1, len(MATCH_NUM), cfg["d"], cfg["emb"], cfg["layers"], cfg["drop"])
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    for ep in range(n_ep):
        net.train()
        perm = np.random.permutation(train_idx)
        for i in range(0, len(perm), 256):
            ix = perm[i:i + 256]
            loss = loss_f(net(*batch(ix, cfg["pdrop"])), torch.tensor(Y[ix]))
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        return torch.exp(net(*batch(test_idx))).numpy()


def walk(cfg, start, end, seeds):
    out = []
    dates = feats.date
    for a, b in folds(start, end, 3)[:1 if SMOKE else None]:
        tr = np.where(feats.ok & (dates >= TRAIN_START) & (dates < a))[0]
        te = np.where((dates >= a) & (dates < b))[0]
        if len(te) == 0:
            continue
        lam = np.mean([fit_predict(tr, te, cfg, s) for s in seeds], 0)
        out.append(to_matches(feats.iloc[te], lam))
    return pd.concat(out)


def evaluate(lm):
    y = outcome(matches, lm.index)
    rho = fit_rho(lm, y)
    return ll_vec(score_probs(lm.lam_h, lm.lam_a, rho), y).mean(), rho


BASE_CFG = dict(d=48, emb=16, layers=2, drop=0.15, lr=2e-3, wd=1e-3, pdrop=0.15, min_apps=10)
CONFIGS = [
    BASE_CFG,
    dict(BASE_CFG, d=32, emb=8, drop=0.3, wd=1e-2),                   # plus petit, plus régularisé
    dict(BASE_CFG, emb=8, pdrop=0.3, min_apps=20),                    # vecteurs joueurs plus rares
    dict(BASE_CFG, lr=1e-3, wd=1e-2, drop=0.25),                      # apprentissage plus lent
    dict(BASE_CFG, layers=1, d=32, drop=0.2),                         # une seule couche d'attention
    dict(BASE_CFG, d=64, emb=24, layers=2, drop=0.3, wd=3e-2, pdrop=0.25),  # plus gros, très régularisé
]

if MODE == "tune":
    rows = []
    for i, cfg in enumerate(CONFIGS[:1] if SMOKE else CONFIGS):
        t0 = time.time()
        ll, rho = evaluate(walk(cfg, TUNE_START, EVAL_START, seeds=(0,)))
        rows.append({**cfg, "logloss": ll, "rho": rho})
        print(f"config {i + 1}/{len(CONFIGS)} : log-loss {ll:.4f} [{time.time() - t0:.0f}s] {cfg}", flush=True)
        pd.DataFrame(rows).to_csv(f"{RESULTS}/wf_nn_trials.csv", index=False)
    best = min(rows, key=lambda r: r["logloss"])
    json.dump({k: best[k] for k in BASE_CFG}, open(f"{RESULTS}/wf_nn_best.json", "w"), indent=1)
    print("meilleure configuration :", best, flush=True)

elif MODE == "eval":
    cfg = json.load(open(f"{RESULTS}/wf_nn_best.json"))
    lm_t = walk(cfg, TUNE_START, EVAL_START, seeds=(0,))
    _, rho = evaluate(lm_t)
    pd.DataFrame(score_probs(lm_t.lam_h, lm_t.lam_a, rho), index=lm_t.index,
                 columns=["nn_h", "nn_d", "nn_a"]).to_csv(f"{RESULTS}/wf_nn_tune_preds.csv")
    lm = walk(cfg, EVAL_START, END, seeds=(0, 1, 2))
    Pm = score_probs(lm.lam_h, lm.lam_a, rho)
    y = outcome(matches, lm.index)
    print(f"transformer : {len(lm)} matchs, log-loss {ll_vec(Pm, y).mean():.4f} (rho {rho:+.3f})", flush=True)
    out = pd.DataFrame(Pm, index=lm.index, columns=["nn_h", "nn_d", "nn_a"])
    out["nn_lam_h"], out["nn_lam_a"] = lm.lam_h, lm.lam_a
    out.to_csv(f"{RESULTS}/wf_nn_preds.csv")
    print("wf_nn_preds.csv écrit", flush=True)
