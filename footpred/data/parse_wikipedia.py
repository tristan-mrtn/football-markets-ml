"""
Transforme les pages Wikipédia en tables propres, avec les noms d'équipes Understat.

Usage : python parse_external.py <dossier_data>
Écrit dans <data>/ext/ :
  clubs.csv            équipe Understat ↔ article Wikipédia
  euro_matches.csv     date + équipe Understat pour chaque match européen
  coach_changes.csv    date de nomination de chaque nouvel entraîneur
"""
import glob
import io
import json
import os
import re
import sys
import unicodedata
from difflib import SequenceMatcher

import pandas as pd
from bs4 import BeautifulSoup

DATA = sys.argv[1]
W = f"{DATA}/wiki"
OUT = f"{DATA}/ext"
os.makedirs(OUT, exist_ok=True)
SEASONS = [2021, 2022, 2023, 2024, 2025, 2026]
WIKI_LEAGUE = {"EPL": "Premier_League", "La_liga": "La_Liga", "Bundesliga": "Bundesliga",
               "Serie_A": "Serie_A", "Ligue_1": "Ligue_1"}
STOP = {"fc", "afc", "cf", "ac", "as", "ss", "ssc", "sc", "rc", "ogc", "us", "ud", "cd", "rcd",
        "club", "football", "de", "of", "the", "calcio", "1", "04", "05", "07", "09", "1846",
        "1899", "1900", "1901", "1907", "1909", "1910", "1913", "bc", "sv", "vfb", "vfl", "tsg",
        "fsv", "sad", "stade", "olympique", "real", "atletico", "athletic", "racing", "and", "hotspur",
        "united", "city", "town", "albion", "wanderers", "county", "rovers"}
KEEP_WORDS = {"real", "athletic", "atletico", "united", "city", "stade"}  # utiles s'ils restent seuls


def norm(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s.replace("_", " "))
    return s.split()


def sim(a, b):
    ta, tb = norm(a), norm(b)
    ca = [w for w in ta if w not in STOP] or ta
    cb = [w for w in tb if w not in STOP] or tb
    s1 = SequenceMatcher(None, " ".join(ca), " ".join(cb)).ratio()
    inter = len(set(ca) & set(cb)) / max(1, min(len(set(ca)), len(set(cb))))
    full = SequenceMatcher(None, " ".join(ta), " ".join(tb)).ratio()
    return max(s1, inter * 0.95, full * 0.9) + 0.05 * full   # départage : nom complet


def understat_teams():
    out = {}
    for lg in WIKI_LEAGUE:
        for s in SEASONS:
            p = f"{DATA}/understat/{lg}_{s}.json"
            if os.path.exists(p):
                for t in json.load(open(p))["teams"].values():
                    out.setdefault(t["title"], lg)
    return out


# ------------------------------------------------------------ clubs
def league_clubs():
    """Articles des clubs, pris dans le tableau des stades de chaque page de saison."""
    rows = []
    for lg, wl in WIKI_LEAGUE.items():
        for s in SEASONS:
            p = f"{W}/{s}–{(s + 1) % 100:02d}_{wl}.html"
            if not os.path.exists(p):
                continue
            soup = BeautifulSoup(open(p, encoding="utf-8").read(), "lxml")
            for tab in soup.select("table.wikitable"):
                head = tab.find("tr").get_text(" ").lower()
                if not ("stadium" in head or "venue" in head):
                    continue
                for tr in tab.find_all("tr")[1:]:
                    cell = tr.find(["th", "td"])
                    a = cell.find("a", href=True) if cell else None
                    href = a["href"].split("/wiki/")[-1] if a and "/wiki/" in a["href"] else ""
                    if href and ":" not in href:
                        rows.append({"league": lg, "season": s, "article": href,
                                     "label": a.get_text(strip=True)})
                break
    return pd.DataFrame(rows).drop_duplicates(["league", "article"])


def map_clubs(us, clubs):
    out = []
    for team, lg in us.items():
        cand = clubs[clubs.league == lg]
        if cand.empty:
            continue
        sc = cand.apply(lambda r: max(sim(team, r.label), sim(team, r.article)), axis=1)
        best = sc.idxmax()
        out.append({"team": team, "league": lg, "article": cand.loc[best, "article"],
                    "label": cand.loc[best, "label"], "score": round(sc.max(), 2)})
    return pd.DataFrame(out)


# ------------------------------------------------------------ Coupes d'Europe
BOX = re.compile(r"\{\{\s*(?:#invoke:\s*)?football\s*box", re.I)
DATE = re.compile(r"\{\{\s*start date\s*\|\s*(\d{4})\s*\|\s*(\d{1,2})\s*\|\s*(\d{1,2})", re.I)
LINK = re.compile(r"\[\[([^\]|#]+)(?:\|[^\]]*)?\]\]")


def euro_matches():
    rows = []
    for p in glob.glob(f"{W}/*UEFA*.txt"):
        txt = open(p, encoding="utf-8").read()
        starts = [m.start() for m in BOX.finditer(txt)] + [len(txt)]
        comp = "C1" if "Champions" in p else "C3" if "Europa_League" in p else "C4"
        for a, b in zip(starts, starts[1:]):
            blk = txt[a:b]
            d = DATE.search(blk)
            t1 = re.search(r"\|\s*team1\s*=\s*(.*)", blk)
            t2 = re.search(r"\|\s*team2\s*=\s*(.*)", blk)
            if not (d and t1 and t2):
                continue
            date = pd.Timestamp(int(d[1]), int(d[2]), int(d[3]))
            for t in (t1[1], t2[1]):
                lk = LINK.search(t)
                if lk:
                    rows.append({"date": date, "article": lk[1].strip().replace(" ", "_"), "comp": comp})
    return pd.DataFrame(rows).drop_duplicates()


# ------------------------------------------------------- changements d'entraîneur
def coach_changes(mapping):
    rows = []
    for lg, wl in WIKI_LEAGUE.items():
        teams = mapping[mapping.league == lg]
        for s in SEASONS:
            p = f"{W}/{s}–{(s + 1) % 100:02d}_{wl}.html"
            if not os.path.exists(p):
                continue
            html = open(p, encoding="utf-8").read()
            try:
                tables = pd.read_html(io.StringIO(html), match="Incoming|appointment|Replaced by", flavor="lxml")
            except ValueError:
                continue
            for t in tables:
                t.columns = [" ".join(map(str, c)) if isinstance(c, tuple) else str(c) for c in t.columns]
                tc = next((c for c in t.columns if "team" in c.lower() or "club" in c.lower()), None)
                dc = next((c for c in t.columns if "date of appointment" in c.lower() or "incoming date" in c.lower()), None)
                vc = next((c for c in t.columns if "vacancy" in c.lower() or "exit date" in c.lower()), None)
                ic = next((c for c in t.columns if ("incoming" in c.lower() or "replaced" in c.lower()) and "date" not in c.lower()), None)
                if tc is None or dc is None:
                    continue
                for r in t.itertuples(index=False):
                    r = dict(zip(t.columns, r))
                    date = pd.to_datetime(re.sub(r"\[.*?\]", "", str(r[dc])), errors="coerce",
                                          dayfirst=True)
                    if pd.isna(date) and vc:
                        date = pd.to_datetime(re.sub(r"\[.*?\]", "", str(r[vc])), errors="coerce",
                                              dayfirst=True)
                    if pd.isna(date):
                        continue
                    name = re.sub(r"\[.*?\]", "", str(r[tc]))
                    sc = teams.apply(lambda x: max(sim(name, x.team), sim(name, x.label)), axis=1)
                    if len(sc) and sc.max() >= 0.6:
                        rows.append({"league": lg, "team": teams.loc[sc.idxmax(), "team"],
                                     "date": date.normalize(),
                                     "incoming": str(r[ic]) if ic else ""})
    return pd.DataFrame(rows).drop_duplicates(["team", "date"])


if __name__ == "__main__":
    us = understat_teams()
    clubs = league_clubs()
    mapping = map_clubs(us, clubs)
    mapping.to_csv(f"{OUT}/clubs.csv", index=False)
    print(f"{len(us)} équipes Understat, {len(mapping)} reliées à un article")
    print("Correspondances les moins sûres :")
    print(mapping.sort_values("score").head(12).to_string(index=False))

    em = euro_matches()
    art2team = dict(zip(mapping.article, mapping.team))
    em["team"] = em.article.map(art2team)
    em = em.dropna(subset=["team"])
    em.to_csv(f"{OUT}/euro_matches.csv", index=False)
    print(f"\n{len(em)} matchs européens d'équipes des 5 ligues "
          f"({em.date.min():%Y-%m-%d} → {em.date.max():%Y-%m-%d})")
    print(em.groupby(em.date.dt.to_period("Q")).size().to_string())

    cc = coach_changes(mapping)
    cc.to_csv(f"{OUT}/coach_changes.csv", index=False)
    print(f"\n{len(cc)} changements d'entraîneur")
    print(cc.groupby("league").size().to_string())
