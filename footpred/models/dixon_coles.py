"""
Modèle Dixon-Coles (1997) pour prédire les matchs de foot.

Chaque équipe i a une force d'attaque a_i et une force de défense d_i.
Buts attendus pour un match domicile h contre extérieur e :
    lambda = exp(mu + home + a_h - d_e)
    nu     = exp(mu        + a_e - d_h)
Les buts suivent une loi de Poisson, avec la correction "tau" de Dixon-Coles
pour les petits scores (0-0, 1-0, 0-1, 1-1), qui ne sont pas indépendants.
Les vieux matchs pèsent moins grâce à une pondération exponentielle dans le temps.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

MAX_GOALS = 10


def load_matches(paths):
    frames = []
    for p in paths:
        df = pd.read_csv(p, encoding="utf-8-sig")
        df = df.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"])
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    df["FTHG"] = df["FTHG"].astype(int)
    df["FTAG"] = df["FTAG"].astype(int)
    return df.sort_values("Date").reset_index(drop=True)


def _tau(x, y, lam, nu, rho):
    """Correction de Dixon-Coles pour les scores 0-0, 1-0, 0-1, 1-1."""
    t = np.ones_like(lam)
    m00 = (x == 0) & (y == 0)
    m01 = (x == 0) & (y == 1)
    m10 = (x == 1) & (y == 0)
    m11 = (x == 1) & (y == 1)
    t[m00] = 1 - lam[m00] * nu[m00] * rho
    t[m01] = 1 + lam[m01] * rho
    t[m10] = 1 + nu[m10] * rho
    t[m11] = 1 - rho
    return t


class DixonColes:
    def __init__(self, xi=0.0025, l2=0.5):
        self.xi = xi    # vitesse d'oubli (par jour) : demi-vie ≈ ln2/xi ≈ 280 jours
        self.l2 = l2    # régularisation : rapproche de la moyenne les équipes avec peu de matchs

    def fit(self, df, ref_date=None):
        ref_date = ref_date or df["Date"].max()
        self.teams = sorted(set(df["HomeTeam"]) | set(df["AwayTeam"]))
        idx = {t: i for i, t in enumerate(self.teams)}
        n = len(self.teams)
        h = df["HomeTeam"].map(idx).to_numpy()
        a = df["AwayTeam"].map(idx).to_numpy()
        x = df["FTHG"].to_numpy()
        y = df["FTAG"].to_numpy()
        age = (ref_date - df["Date"]).dt.days.to_numpy()
        w = np.exp(-self.xi * age)

        def nll(p):
            att, dfn = p[:n], p[n:2 * n]
            mu, home, rho = p[2 * n], p[2 * n + 1], p[2 * n + 2]
            lam = np.exp(mu + home + att[h] - dfn[a])
            nu = np.exp(mu + att[a] - dfn[h])
            t = np.clip(_tau(x, y, lam, nu, rho), 1e-10, None)
            ll = (np.log(t) + poisson.logpmf(x, lam) + poisson.logpmf(y, nu))
            pen = self.l2 * (np.sum(att ** 2) + np.sum(dfn ** 2))
            # contrainte d'identifiabilité : somme des attaques = 0
            pen += 100 * np.sum(att) ** 2
            return -np.sum(w * ll) + pen

        p0 = np.concatenate([np.zeros(2 * n), [0.2, 0.25, -0.05]])
        bounds = [(None, None)] * (2 * n + 2) + [(-0.3, 0.3)]
        res = minimize(nll, p0, method="L-BFGS-B", bounds=bounds)
        p = res.x
        self.att = dict(zip(self.teams, p[:n]))
        self.dfn = dict(zip(self.teams, p[n:2 * n]))
        self.mu, self.home, self.rho = p[2 * n], p[2 * n + 1], p[2 * n + 2]
        return self

    def expected_goals(self, home, away):
        a_h, d_h = self.att.get(home, 0.0), self.dfn.get(home, 0.0)
        a_a, d_a = self.att.get(away, 0.0), self.dfn.get(away, 0.0)
        lam = np.exp(self.mu + self.home + a_h - d_a)
        nu = np.exp(self.mu + a_a - d_h)
        return lam, nu

    def score_matrix(self, home, away):
        """Matrice M[i, j] = P(home marque i, away marque j)."""
        lam, nu = self.expected_goals(home, away)
        g = np.arange(MAX_GOALS + 1)
        M = np.outer(poisson.pmf(g, lam), poisson.pmf(g, nu))
        X, Y = np.meshgrid(g, g, indexing="ij")
        M *= _tau(X, Y, np.full_like(M, lam), np.full_like(M, nu), self.rho)
        return M / M.sum()

    def predict(self, home, away):
        M = self.score_matrix(home, away)
        lam, nu = self.expected_goals(home, away)
        i, j = np.unravel_index(M.argmax(), M.shape)
        tot = np.add.outer(np.arange(MAX_GOALS + 1), np.arange(MAX_GOALS + 1))
        return {
            "home": home, "away": away,
            "xG_home": lam, "xG_away": nu,
            "p_home": np.tril(M, -1).sum(),
            "p_draw": np.trace(M),
            "p_away": np.triu(M, 1).sum(),
            "p_over25": M[tot > 2.5].sum(),
            "p_btts": M[1:, 1:].sum(),
            "score_probable": f"{i}-{j}",
            "p_score": M[i, j],
        }

    def simulate(self, home, away, n, rng):
        """Tirage Monte-Carlo de n scores selon la matrice du modèle."""
        M = self.score_matrix(home, away).ravel()
        k = rng.choice(M.size, size=n, p=M)
        return k // (MAX_GOALS + 1), k % (MAX_GOALS + 1)
