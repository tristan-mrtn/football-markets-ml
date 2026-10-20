"""Rattache les cotes football-data.co.uk aux matchs Understat (les noms d'équipes diffèrent)."""
import glob
from collections import Counter, defaultdict

import pandas as pd

CODES = {"E0": "EPL", "SP1": "La_liga", "D1": "Bundesliga", "I1": "Serie_A", "F1": "Ligue_1"}
ODDS_COLS = ["AvgH", "AvgD", "AvgA", "MaxH", "MaxD", "MaxA", "PSH", "PSD", "PSA",
             "PSCH", "PSCD", "PSCA", "AvgCH", "AvgCD", "AvgCA", "MaxCH", "MaxCD", "MaxCA",
             # stats de match (toutes les ligues) et arbitre (Angleterre)
             "HS", "AS", "HST", "AST", "HC", "AC", "HF", "AF", "HY", "AY", "HR", "AR", "Referee"]


def load_odds(data):
    frames = []
    for p in glob.glob(f"{data}/odds/*.csv"):
        code = p.split("/")[-1].split("_")[0]
        d = pd.read_csv(p, encoding="utf-8-sig", encoding_errors="replace", on_bad_lines="skip")
        d = d.dropna(subset=["HomeTeam", "AwayTeam", "FTHG"]).reset_index(drop=True)
        cols = {"league": CODES[code],
                "date": pd.to_datetime(d["Date"], dayfirst=True, format="mixed").dt.normalize(),
                "HomeTeam": d["HomeTeam"], "AwayTeam": d["AwayTeam"], "FTHG": d["FTHG"], "FTAG": d["FTAG"]}
        for c in ODDS_COLS:
            if c not in d:
                cols[c] = float("nan")
            else:
                cols[c] = d[c] if c == "Referee" else pd.to_numeric(d[c], errors="coerce")
        frames.append(pd.DataFrame(cols))
    return pd.concat(frames, ignore_index=True)


def join(matches, data):
    odds = load_odds(data)
    m = matches.copy()
    m["day"] = m["date"].dt.normalize()

    # 1) apprendre la correspondance des noms : même ligue, même jour, même score
    votes = defaultdict(Counter)
    key = odds.set_index(["league", "date", "FTHG", "FTAG"]).sort_index()
    for r in m.itertuples():
        for dd in (0, -1, 1):
            k = (r.league, r.day + pd.Timedelta(days=dd), r.hg, r.ag)
            if k in key.index:
                cand = key.loc[[k]]
                if len(cand) == 1:
                    votes[(r.league, r.home)][cand.HomeTeam.iloc[0]] += 1
                    votes[(r.league, r.away)][cand.AwayTeam.iloc[0]] += 1
                break
    name = {k: c.most_common(1)[0][0] for k, c in votes.items()}

    # 2) jointure sur (ligue, noms traduits, date ± 1 jour)
    idx = {(r.league, r.HomeTeam, r.AwayTeam, r.date): i for i, r in enumerate(odds.itertuples())}
    rows = []
    for r in m.itertuples():
        h, a = name.get((r.league, r.home)), name.get((r.league, r.away))
        hit = None
        for dd in (0, -1, 1):
            hit = idx.get((r.league, h, a, r.day + pd.Timedelta(days=dd)))
            if hit is not None:
                break
        rows.append(odds.iloc[hit][ODDS_COLS].to_dict() if hit is not None else {})
    out = pd.concat([m.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    print(f"cotes trouvées pour {out['AvgH'].notna().mean():.1%} des matchs")
    return out
