"""Relie les joueurs Understat à leur date de naissance Wikidata (même club + nom proche).

Usage : python build_ages.py <dossier_data>   →  <data>/ext/player_birth.csv
"""
import json
import os
import re
import sys
import unicodedata
from collections import defaultdict
from difflib import SequenceMatcher

import pandas as pd

DATA = sys.argv[1]
E = f"{DATA}/ext"


def norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z ]", " ", s).split()


venues = pd.read_csv(f"{E}/venues.csv")
team2qid = dict(zip(venues.team, venues.qid))
wd = pd.read_csv(f"{E}/players_wd.csv")
by_club = defaultdict(list)
for r in wd.itertuples():
    by_club[r.club].append((norm(r.name), r.player, r.birth))

# clubs fréquentés par chaque joueur Understat
teams_of, names = defaultdict(set), {}
for f in os.listdir(f"{DATA}/understat/full"):
    if not f.endswith(".json"):
        continue
    d = json.load(open(f"{DATA}/understat/full/{f}"))
    for side in ("h", "a"):
        sh = d["shots"][side]
        for x in (d["rosters"][side].values() if isinstance(d["rosters"][side], dict) else []):
            names[x["player_id"]] = x["player"]
            # le nom d'équipe est dans les tirs ; à défaut on passe par team_id plus bas
            teams_of[x["player_id"]].add(x["team_id"])

# team_id Understat -> nom d'équipe (via les fichiers de ligue)
tid2name = {}
for f in os.listdir(f"{DATA}/understat"):
    if f.endswith(".json"):
        for t in json.load(open(f"{DATA}/understat/{f}"))["teams"].values():
            tid2name[t["id"]] = t["title"]

rows = []
for pid, name in names.items():
    n = norm(name)
    best, score = None, 0.0
    for tid in teams_of[pid]:
        q = team2qid.get(tid2name.get(tid))
        for cand, wid, birth in by_club.get(q, []):
            s = SequenceMatcher(None, " ".join(n), " ".join(cand)).ratio()
            if n and cand and n[-1] == cand[-1]:
                s += 0.15                       # même nom de famille
            if len(n) == 1 and n[0] in cand:
                s += 0.3                        # joueurs connus par un seul nom (ex. « Rodri »)
            if s > score:
                best, score = (wid, birth), s
    if best and score >= 0.85:
        rows.append({"pid": pid, "name": name, "wikidata": best[0], "birth": best[1],
                     "score": round(score, 2)})

out = pd.DataFrame(rows)
out.to_csv(f"{E}/player_birth.csv", index=False)
print(f"{len(out)}/{len(names)} joueurs avec une date de naissance ({len(out) / len(names):.0%})")
