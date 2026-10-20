"""Météo Open-Meteo, très lentement (quota partagé). Usage : python fetch_weather_slow.py <data>"""
import os
import sys
import time
import urllib.parse
import urllib.request

import pandas as pd

OUT = sys.argv[1] + "/ext/weather"
os.makedirs(OUT, exist_ok=True)
venues = pd.read_csv(sys.argv[1] + "/ext/venues.csv").dropna(subset=["lat"]).drop_duplicates("qid")
for r in venues.itertuples():
    path = f"{OUT}/{r.qid}.json"
    if os.path.exists(path):
        continue
    url = ("https://archive-api.open-meteo.com/v1/archive?" + urllib.parse.urlencode({
        "latitude": round(r.lat, 4), "longitude": round(r.lon, 4),
        "start_date": "2021-08-01", "end_date": "2026-10-18",
        "hourly": "temperature_2m,precipitation,wind_speed_10m", "timezone": "GMT"}))
    for k in range(20):
        try:
            data = urllib.request.urlopen(urllib.request.Request(
                url, headers={"User-Agent": "football-markets-ml/1.0 (+https://github.com/tristan-mrtn/football-markets-ml)"}), timeout=120).read()
            open(path, "wb").write(data)
            print(f"ok {r.team}", flush=True)
            break
        except Exception as e:
            time.sleep(300)          # 5 min d'attente quand le quota est atteint
    time.sleep(60)
print("météo terminée", flush=True)
