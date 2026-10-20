"""Télécharge les pages Wikipédia utiles (texte brut + HTML), lentement.

Usage : python fetch_external.py <dossier_data>
  - matchs de Coupe d'Europe (C1, C3, C4 : qualifications, phase de groupes/ligue, phase finale)
  - pages de saison des 5 championnats (clubs + changements d'entraîneur)
"""
import os
import sys
import time
import urllib.parse
import urllib.request

OUT = sys.argv[1] + "/wiki"
os.makedirs(OUT, exist_ok=True)
UA = "football-markets-ml/1.0 (+https://github.com/tristan-mrtn/football-markets-ml)"
SEASONS = [2021, 2022, 2023, 2024, 2025, 2026]
LEAGUE_PAGES = ["Premier_League", "La_Liga", "Bundesliga", "Serie_A", "Ligue_1"]


def label(s):
    return f"{s}–{(s + 1) % 100:02d}"


def euro_pages(s):
    L = label(s)
    conf = "UEFA_Europa_Conference_League" if s <= 2023 else "UEFA_Conference_League"
    group = "group_stage" if s <= 2023 else "league_phase"
    pages = []
    for comp in ("UEFA_Champions_League", "UEFA_Europa_League", conf):
        for part in ("qualifying_phase_and_play-off_round", group, "knockout_phase"):
            pages.append(f"{L}_{comp}_{part}")
    return pages


def get(url):
    for k in range(4):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=40) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(5 * (k + 1))
        except Exception:
            time.sleep(5 * (k + 1))
    return None


jobs = []
for s in SEASONS:
    for p in euro_pages(s):
        jobs.append((p, "raw"))
    for lg in LEAGUE_PAGES:
        p = f"{label(s)}_{lg}"
        jobs.append((p, "raw"))
        jobs.append((p, "html"))

for page, kind in jobs:
    path = f"{OUT}/{page}.{'txt' if kind == 'raw' else 'html'}"
    if os.path.exists(path):
        continue
    url = "https://en.wikipedia.org/wiki/" + urllib.parse.quote(page)
    data = get(url + ("?action=raw" if kind == "raw" else ""))
    if data is None:
        print(f"absent : {page}", flush=True)
    else:
        open(path, "wb").write(data)
    time.sleep(1.5)
print("wiki terminé", flush=True)
