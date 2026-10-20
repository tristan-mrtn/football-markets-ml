"""
Wikidata (stades, dates de naissance) + Open-Meteo (météo au stade).

Usage : python fetch_wikidata.py <dossier_data>     (après parse_external.py)
Écrit dans <data>/ext/ : venues.csv, players_wd.csv, weather/<club>.json
"""
import json
import os
import sys
import time
import urllib.parse
import urllib.request

import pandas as pd

DATA = sys.argv[1]
OUT = f"{DATA}/ext"
os.makedirs(f"{OUT}/weather", exist_ok=True)
UA = "football-markets-ml/1.0 (+https://github.com/tristan-mrtn/football-markets-ml)"


def http(url, data=None, tries=5):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, data=data, headers={
                "User-Agent": UA, "Accept": "application/sparql-results+json"})
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except Exception as e:
            print(f"  nouvel essai ({e})", flush=True)
            time.sleep(10 * (k + 1))
    raise RuntimeError(url)


def sparql(q):
    body = urllib.parse.urlencode({"query": q}).encode()
    for k in range(5):
        try:
            res = json.loads(http("https://query.wikidata.org/sparql", body))
            break
        except json.JSONDecodeError:        # réponse tronquée : on réessaie
            print("  réponse tronquée, nouvel essai", flush=True)
            time.sleep(15 * (k + 1))
    time.sleep(2)
    return [{k: v["value"] for k, v in b.items()} for b in res["results"]["bindings"]]


def iri_title(article):
    """Suit les redirections Wikipédia et encode le titre comme Wikidata."""
    title = urllib.parse.unquote(article)
    raw = http("https://en.wikipedia.org/wiki/" + urllib.parse.quote(title) + "?action=raw").decode("utf-8", "ignore")
    m = __import__("re").match(r"#REDIRECT\s*\[\[([^\]|#]+)", raw, flags=2)
    if m:
        title = m.group(1).strip()
    time.sleep(1)
    return urllib.parse.quote(title.replace(" ", "_"), safe=";@$!*(),/~:_-.")


clubs = pd.read_csv(f"{OUT}/clubs.csv")

# ---------------------------------------------------------------- 1) clubs et stades
if not os.path.exists(f"{OUT}/venues.csv"):
    rows = []
    canon = {a: iri_title(a) for a in clubs.article.unique()}
    clubs["iri"] = clubs.article.map(canon)
    arts = clubs.iri.unique().tolist()
    for i in range(0, len(arts), 40):
        vals = " ".join(f"<https://en.wikipedia.org/wiki/{a}>" for a in arts[i:i + 40])
        rows += sparql(f"""
        SELECT ?article ?club ?coord ?end WHERE {{
          VALUES ?article {{ {vals} }}
          ?article schema:about ?club .
          OPTIONAL {{ ?club p:P115 ?st . ?st ps:P115 ?venue . ?venue wdt:P625 ?coord .
                     OPTIONAL {{ ?st pq:P582 ?end }} }}
        }}""")
    v = pd.DataFrame(rows)
    v["article"] = v.article.str.replace("https://en.wikipedia.org/wiki/", "", regex=False)
    v = v.sort_values("end", na_position="first").drop_duplicates("article")   # stade actuel d'abord
    xy = v.coord.str.extract(r"Point\(([-\d.]+) ([-\d.]+)\)").astype(float)
    v["lon"], v["lat"] = xy[0], xy[1]
    v["qid"] = v.club.str.rsplit("/", n=1).str[-1]
    v = clubs.merge(v.rename(columns={"article": "iri"})[["iri", "qid", "lat", "lon"]], on="iri", how="left")
    v.to_csv(f"{OUT}/venues.csv", index=False)
    print(f"stades : {v.lat.notna().sum()}/{len(v)} clubs localisés", flush=True)
venues = pd.read_csv(f"{OUT}/venues.csv")

# ------------------------------------------------- 2) joueurs passés par ces clubs
if not os.path.exists(f"{OUT}/players_wd.csv"):
    rows = []
    qids = venues.qid.dropna().unique().tolist()
    for i in range(0, len(qids), 3):
        vals = " ".join(f"wd:{q}" for q in qids[i:i + 3])
        rows += sparql(f"""
        SELECT ?player ?club ?name ?birth WHERE {{
          VALUES ?club {{ {vals} }}
          ?player p:P54 ?st . ?st ps:P54 ?club .
          OPTIONAL {{ ?st pq:P582 ?end }}
          FILTER(!BOUND(?end) || ?end >= "2020-01-01T00:00:00Z"^^xsd:dateTime)
          ?player wdt:P569 ?birth .
          ?player rdfs:label ?name . FILTER(LANG(?name) = "en")
        }}""")
        print(f"  joueurs : clubs {i + 3}/{len(qids)}, {len(rows)} lignes", flush=True)
    p = pd.DataFrame(rows)
    p["club"] = p.club.str.rsplit("/", n=1).str[-1]
    p["birth"] = p.birth.str[:10]
    p.drop_duplicates().to_csv(f"{OUT}/players_wd.csv", index=False)

# ------------------------------------------------------- 3) météo horaire au stade
for r in venues.dropna(subset=["lat"]).drop_duplicates("qid").itertuples():
    path = f"{OUT}/weather/{r.qid}.json"
    if os.path.exists(path):
        continue
    url = ("https://archive-api.open-meteo.com/v1/archive?" + urllib.parse.urlencode({
        "latitude": round(r.lat, 4), "longitude": round(r.lon, 4),
        "start_date": "2021-08-01", "end_date": "2026-10-18",
        "hourly": "temperature_2m,precipitation,wind_speed_10m", "timezone": "GMT"}))
    open(path, "wb").write(http(url))
    time.sleep(1)
print("wikidata + météo terminés", flush=True)
