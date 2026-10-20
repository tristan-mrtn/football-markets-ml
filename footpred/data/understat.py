"""Collecte des données : matchs + stats joueurs (Understat) et cotes (football-data.co.uk).

Usage : python fetch_data.py <dossier_data>
"""
import gzip
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

OUT = sys.argv[1]
LEAGUES = {"EPL": "E0", "La_liga": "SP1", "Bundesliga": "D1", "Serie_A": "I1", "Ligue_1": "F1"}
SEASONS = [2021, 2022, 2023, 2024, 2025, 2026]  # année de début de saison
UA = "Mozilla/5.0 (projet etudiant, collecte lente)"


def get(url, ajax_ref=None, tries=4):
    for k in range(tries):
        try:
            h = {"User-Agent": UA, "Accept-Encoding": "gzip"}
            if ajax_ref:
                h["X-Requested-With"] = "XMLHttpRequest"
                h["Referer"] = ajax_ref
            with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=30) as r:
                data = r.read()
                if r.headers.get("Content-Encoding") == "gzip":
                    data = gzip.decompress(data)
                return data
        except Exception as e:
            if k == tries - 1:
                raise
            time.sleep(2 ** k)


def main():
    os.makedirs(f"{OUT}/understat/full", exist_ok=True)
    os.makedirs(f"{OUT}/odds", exist_ok=True)

    # 1) Cotes et résultats football-data.co.uk
    for lg, code in LEAGUES.items():
        for s in SEASONS:
            tag = f"{s % 100:02d}{(s + 1) % 100:02d}"
            path = f"{OUT}/odds/{code}_{tag}.csv"
            if not os.path.exists(path):
                with open(path, "wb") as f:
                    f.write(get(f"https://www.football-data.co.uk/mmz4281/{tag}/{code}.csv"))

    # 2) Liste des matchs par ligue/saison (xG d'équipe)
    match_ids = []
    for lg in LEAGUES:
        for s in SEASONS:
            path = f"{OUT}/understat/{lg}_{s}.json"
            if not os.path.exists(path):
                d = get(f"https://understat.com/getLeagueData/{lg}/{s}",
                        ajax_ref=f"https://understat.com/league/{lg}/{s}")
                open(path, "wb").write(d)
                time.sleep(0.5)
            for m in json.load(open(path))["dates"]:
                if m["isResult"]:
                    match_ids.append(m["id"])
    print(f"{len(match_ids)} matchs joués trouvés", flush=True)

    # 3) Feuilles de match (stats de chaque joueur)
    todo = [i for i in match_ids if not os.path.exists(f"{OUT}/understat/full/{i}.json")]
    print(f"{len(todo)} feuilles de match à télécharger", flush=True)

    def fetch(mid):
        try:
            d = json.loads(get(f"https://understat.com/getMatchData/{mid}",
                               ajax_ref=f"https://understat.com/match/{mid}"))
            tmp = f"{OUT}/understat/full/{mid}.json.tmp"
            json.dump({"rosters": d["rosters"], "shots": d["shots"]}, open(tmp, "w"))
            os.replace(tmp, f"{OUT}/understat/full/{mid}.json")
        except Exception as e:
            print(f"  échec {mid}: {e}", flush=True)
        time.sleep(0.3)

    done = 0
    with ThreadPoolExecutor(4) as ex:
        for _ in ex.map(fetch, todo):
            done += 1
            if done % 500 == 0:
                print(f"  {done}/{len(todo)}", flush=True)
    print("terminé", flush=True)


main()
