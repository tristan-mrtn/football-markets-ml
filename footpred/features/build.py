"""
Construit toutes les variables, une ligne par (match, équipe qui attaque), plus une
table « joueurs » (un jeton par titulaire) pour le réseau de neurones.

RÈGLE ANTI-FUITE : chaque variable d'un match est calculée avec des données
strictement antérieures au match. On parcourt les matchs dans l'ordre et on ne met
à jour les historiques qu'APRÈS avoir écrit le match.
Informations connues avant le coup d'envoi, donc autorisées : le onze de départ
(publié ~1 h avant), le calendrier (matchs européens à venir), l'arbitre désigné,
la météo (prévisions fiables quelques heures avant ; on utilise l'observé).

Familles (préfixe de colonne → famille, utilisé par ablation.py) :
  A force      : xgf, xga, gf, ga, npxg, elo, is_home, rest
  B forme      : versions courtes / longues, écarts (retour à la moyenne)
  C attaque    : xi_off, xi_chain, xi_buildup, xi_top3_off
  D défense    : def_line_onxga, gk_prev, xi_onxgf
  E dispo      : absences, retours, suspensions, charge, banc
  F automatismes : chem_*
  G tactique   : formation, ppda, deep, coups de pied arrêtés, style des tirs
  H face-à-face: h2h_*
  I contexte   : classement/enjeu, calendrier, Europe, entraîneur, météo, distance, arbitre
  J chance     : finition, xpts
  K stats match: tirs, cadrés, corners, fautes, cartons (football-data)
  M marché     : cotes d'ouverture (variante « modèle + marché » uniquement)
"""
import bisect
import json
import math
import os
import sys
from collections import defaultdict, deque
from itertools import combinations

import numpy as np
import pandas as pd

from footpred.data.odds import join as odds_join

DATA = sys.argv[1] if len(sys.argv) > 1 else "data"
LEAGUES = ["EPL", "La_liga", "Bundesliga", "Serie_A", "Ligue_1"]
SEASONS = [2021, 2022, 2023, 2024, 2025, 2026]
A_SHORT, A_MID, A_LONG = 0.30, 0.10, 0.03      # ~2, ~7, ~23 matchs de demi-vie
ALPHA_PLAYER = 0.08
CHEM_DECAY = 0.97
DEF_POS = {"GK", "DR", "DC", "DL", "DMR", "DML"}
SET_PIECE = {"FromCorner", "SetPiece", "DirectFreekick"}
NAN = float("nan")


# ================================================================ chargement
def load_matches():
    rows, hist = [], {}
    for lg in LEAGUES:
        for s in SEASONS:
            p = f"{DATA}/understat/{lg}_{s}.json"
            if not os.path.exists(p):
                continue
            d = json.load(open(p))
            for t in d["teams"].values():
                for h in t["history"]:
                    hist[(t["title"], h["date"])] = h
            for m in d["dates"]:
                if m["isResult"]:
                    rows.append({
                        "id": m["id"], "league": lg, "season": s, "dt": m["datetime"],
                        "date": pd.Timestamp(m["datetime"]),
                        "home": m["h"]["title"], "away": m["a"]["title"],
                        "hg": int(m["goals"]["h"]), "ag": int(m["goals"]["a"]),
                        "hxg": float(m["xG"]["h"]), "axg": float(m["xG"]["a"]),
                    })
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    if os.environ.get("DROP_ID"):           # test anti-fuite : on retire un match
        df = df[df.id != os.environ["DROP_ID"]].reset_index(drop=True)
    if os.environ.get("CUTOFF"):            # test anti-fuite : on coupe les données
        df = df[df.date < pd.Timestamp(os.environ["CUTOFF"])].reset_index(drop=True)
    return df, hist


def load_full(mid):
    p = f"{DATA}/understat/full/{mid}.json"
    if not os.path.exists(p):
        return None, None
    d = json.load(open(p))
    roster = {}
    for side in ("h", "a"):
        roster[side] = [{
            "pid": x["player_id"], "name": x["player"], "pos": x["position"],
            "min": int(x["time"]), "xg": float(x["xG"]), "xa": float(x["xA"]),
            "chain": float(x["xGChain"]), "buildup": float(x["xGBuildup"]),
            "red": int(x["red_card"]), "start": x["position"] != "Sub",
        } for x in (d["rosters"][side].values() if isinstance(d["rosters"][side], dict) else [])]
    shots = {side: [{"min": int(s["minute"]), "xg": float(s["xG"]), "res": s["result"],
                     "sit": s["situation"], "type": s["shotType"],
                     "x": float(s["X"]), "y": float(s["Y"])} for s in d["shots"][side]]
             for side in ("h", "a")}
    return roster, shots


def load_ext():
    E = f"{DATA}/ext"
    ext = {"euro": defaultdict(list), "coach": defaultdict(list), "venue": {},
           "birth": {}, "weather": {}}
    if os.path.exists(f"{E}/euro_matches.csv"):
        for r in pd.read_csv(f"{E}/euro_matches.csv", parse_dates=["date"]).itertuples():
            ext["euro"][r.team].append(r.date)
        for k in ext["euro"]:
            ext["euro"][k] = sorted(set(ext["euro"][k]))
    if os.path.exists(f"{E}/coach_changes.csv"):
        for r in pd.read_csv(f"{E}/coach_changes.csv", parse_dates=["date"]).itertuples():
            ext["coach"][r.team].append(r.date)
        for k in ext["coach"]:
            ext["coach"][k].sort()
    if os.path.exists(f"{E}/venues.csv"):
        for r in pd.read_csv(f"{E}/venues.csv").dropna(subset=["lat"]).itertuples():
            ext["venue"][r.team] = (r.lat, r.lon, r.qid)
    if os.path.exists(f"{E}/player_birth.csv"):
        b = pd.read_csv(f"{E}/player_birth.csv", dtype={"pid": str}, parse_dates=["birth"])
        ext["birth"] = dict(zip(b.pid, b.birth))
    wdir = f"{E}/weather"
    if os.path.isdir(wdir):
        for f in os.listdir(wdir):
            try:
                h = json.load(open(f"{wdir}/{f}"))["hourly"]
            except Exception:
                continue
            ext["weather"][f[:-5]] = pd.DataFrame(h).assign(
                time=lambda d: pd.to_datetime(d.time)).set_index("time")
    return ext


def haversine(a, b):
    la1, lo1, la2, lo2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 6371 * 2 * math.asin(math.sqrt(h))


def formation(xi):
    """4-2-3-1, 3-5-2… déduit des postes Understat des titulaires."""
    pos = [p["pos"] for p in xi if p["pos"] != "GK"]
    back = sum(p in ("DR", "DC", "DL") for p in pos)
    wingback = sum(p in ("DMR", "DML") for p in pos)
    dm = sum(p == "DMC" for p in pos)
    mid = sum(p in ("MR", "MC", "ML") for p in pos)
    am = sum(p in ("AMR", "AMC", "AML") for p in pos)
    fw = sum(p in ("FW", "FWR", "FWL") for p in pos)
    if back == 3 and wingback == 2:
        return f"3-{dm + mid + 2}-{am + fw}" if am <= 1 else f"3-{dm + mid + 2}-{am}-{fw}"
    back += wingback
    parts = [back] + [x for x in (dm, mid, am, fw) if x]
    return "-".join(map(str, parts))


# ================================================================ états
class Ewma:
    """Moyenne glissante exponentielle avec démarrage propre (moyenne simple au début)."""
    __slots__ = ("v", "n", "a")

    def __init__(self, a):
        self.v, self.n, self.a = None, 0, a

    def add(self, x):
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return
        a = max(self.a, 1 / (self.n + 1))
        self.v = x if self.v is None else (1 - a) * self.v + a * x
        self.n += 1

    def get(self):
        return NAN if self.v is None else self.v


class Ratio:
    """Ratio de deux sommes décroissantes (ex. xG par 90 min à score égal)."""
    __slots__ = ("num", "den", "d")

    def __init__(self, decay):
        self.num = self.den = 0.0
        self.d = decay

    def add(self, num, den):
        self.num = self.d * self.num + num
        self.den = self.d * self.den + den

    def get(self, scale=1.0, min_den=1e-9):
        return self.num / self.den * scale if self.den > min_den else NAN


TEAM_EWMA = ["xgf", "xga", "gf", "ga", "npxgf", "npxga", "sp_xgf", "sp_xga", "op_xgf", "op_xga",
             "xg_per_shot_f", "xg_per_shot_a", "box_share_f", "head_share_f", "big_ch_f", "big_ch_a",
             "ppda", "ppda_allowed", "deep", "deep_allowed", "pts", "xpts", "draw",
             "shots_f", "shots_a", "sot_f", "sot_a", "corners_f", "corners_a", "fouls_f",
             "fouls_a", "yellow_f", "red_f"]


class Team:
    def __init__(self):
        self.mid = {k: Ewma(A_MID) for k in TEAM_EWMA}
        self.short = {k: Ewma(A_SHORT) for k in ("xgf", "xga", "pts")}
        self.long = {k: Ewma(A_LONG) for k in ("xgf", "xga", "gf", "ga", "pts", "xpts")}
        self.level_f, self.level_a = Ratio(0.9), Ratio(0.9)     # xG à score égal
        self.close_f, self.close_a = Ratio(0.9), Ratio(0.9)     # xG à ±1 but
        self.home_xgd, self.away_xgd = Ewma(A_MID), Ewma(A_MID)
        self.n = 0
        self.dates = []
        self.last_xi = set()
        self.last_formation = None
        self.recent_minutes = deque(maxlen=10)
        self.pairs = {}

    def chem(self, a, b):
        key = (a, b) if a < b else (b, a)
        if key not in self.pairs:
            return 0.0
        v, n0 = self.pairs[key]
        return v * CHEM_DECAY ** (self.n - n0)

    def add_chem(self, a, b, minutes):
        key = (a, b) if a < b else (b, a)
        self.pairs[key] = (self.chem(a, b) + minutes / 90, self.n)

    def matches_between(self, d0, d1):
        return bisect.bisect_left(self.dates, d1) - bisect.bisect_right(self.dates, d0)


class Player:
    __slots__ = ("off", "chain", "buildup", "n", "last_date", "last_team", "apps", "first_with",
                 "last_red", "on_xga", "on_xgf", "gk_prev", "gk_shots")

    def __init__(self):
        self.off = self.chain = self.buildup = 0.0
        self.n = 0
        self.last_date = self.last_team = None
        self.apps = deque(maxlen=80)
        self.first_with = {}
        self.last_red = 0
        self.on_xga, self.on_xgf = Ratio(0.95), Ratio(0.95)   # xG de l'équipe par 90 quand il joue
        self.gk_prev, self.gk_shots = 0.0, 0.0               # buts évités (gardien)

    def update(self, p, date, team, xgf, xga):
        self.apps.append((date, team, p["min"]))
        self.first_with.setdefault(team, date)
        self.last_date, self.last_team, self.last_red = date, team, p["red"]
        if p["min"] < 15:
            return
        f = 90 / p["min"]
        a = ALPHA_PLAYER
        self.off = (1 - a) * self.off + a * (p["xg"] + p["xa"]) * f
        self.chain = (1 - a) * self.chain + a * p["chain"] * f
        self.buildup = (1 - a) * self.buildup + a * p["buildup"] * f
        self.n += 1
        if p["min"] >= 60:   # approximation : xG du match entier attribués s'il a joué 60+ min
            self.on_xga.add(xga * p["min"] / 90, p["min"])
            self.on_xgf.add(xgf * p["min"] / 90, p["min"])

    def q(self, attr):
        prior = {"off": 0.25, "chain": 0.45, "buildup": 0.25}[attr]
        w = 1 - (1 - ALPHA_PLAYER) ** self.n
        return w * getattr(self, attr) + (1 - w) * prior

    def onoff(self, which, team_val):
        """xG/90 de l'équipe avec lui, rétréci vers la valeur de l'équipe."""
        r = self.on_xga if which == "a" else self.on_xgf
        k = 900.0                                        # ~10 matchs de poids pour le prior
        prior = team_val if not math.isnan(team_val) else 1.35
        return (r.num + prior * k / 90) / (r.den + k) * 90

    def gk_quality(self):
        k = 40.0                                          # ~10 matchs de tirs cadrés
        return self.gk_prev / (self.gk_shots + k)

    def availability(self, date, team, tobj):
        days_since = (date - self.last_date).days if self.last_date is not None else NAN
        missed = tobj.matches_between(self.last_date, date) if self.last_team == team else NAN
        start = max(date - pd.Timedelta(days=180), self.first_with.get(team, date))
        team_m = tobj.matches_between(start - pd.Timedelta(seconds=1), date)
        played = sum(1 for d, t, _ in self.apps if t == team and d >= start)
        abs_rate = 1 - played / team_m if team_m >= 3 else NAN
        load30 = sum(mn for d, _, mn in self.apps if d >= date - pd.Timedelta(days=30))
        return {"days_since": days_since, "missed_streak": missed, "absence_rate_6m": abs_rate,
                "load_30d": load30, "back_from_red": int(self.last_red > 0 and missed == 1)}


def elo_expect(ra, rb):
    return 1 / (1 + 10 ** ((rb - ra) / 400))


# ================================================================ stats d'un match
def game_state(shots_h, shots_a):
    """xG et minutes à score égal / serré, pour chaque camp."""
    goals = sorted([(s["min"], "h") for s in shots_h if s["res"] in ("Goal", "OwnGoal")] +
                   [(s["min"], "a") for s in shots_a if s["res"] in ("Goal", "OwnGoal")])
    END = max(95, max([s["min"] for s in shots_h + shots_a], default=90))
    segs, diff, t0 = [], 0, 0
    for mn, side in goals:
        segs.append((t0, mn, diff))
        diff += 1 if side == "h" else -1
        t0 = mn
    segs.append((t0, END, diff))

    def diff_at(minute):
        # A shot taken at minute t belongs to the score BEFORE any goal scored at t:
        # the shot that produces the 1-0 was taken at 0-0 (segments are ]a, b]).
        for a, b, d in segs:
            if minute <= b:
                return d
        return segs[-1][2]

    out = {}
    for side, shots, sign in (("h", shots_h, 1), ("a", shots_a, -1)):
        lvl = [s["xg"] for s in shots if s["res"] != "OwnGoal" and diff_at(s["min"]) == 0]
        cls = [s["xg"] for s in shots if s["res"] != "OwnGoal" and abs(diff_at(s["min"])) <= 1]
        out[side] = (sum(lvl), sum(cls))
    min_level = sum(b - a for a, b, d in segs if d == 0)
    min_close = sum(b - a for a, b, d in segs if abs(d) <= 1)
    return out, min_level, min_close


def shot_profile(shots):
    real = [s for s in shots if s["res"] != "OwnGoal"]
    xg = sum(s["xg"] for s in real)
    np_ = sum(s["xg"] for s in real if s["sit"] != "Penalty")
    sp = sum(s["xg"] for s in real if s["sit"] in SET_PIECE)
    op = sum(s["xg"] for s in real if s["sit"] == "OpenPlay")
    n = len(real)
    return {"npxg": np_, "sp": sp, "op": op,
            "per_shot": xg / n if n else NAN,
            "box": np.mean([s["x"] > 0.83 and 0.21 < s["y"] < 0.79 for s in real]) if n else NAN,
            "head": np.mean([s["type"] == "Head" for s in real]) if n else NAN,
            "big": sum(s["xg"] >= 0.3 for s in real)}


def gk_faced(shots_against):
    """(xG des tirs cadrés subis − buts encaissés, nombre de tirs cadrés)"""
    on = [s for s in shots_against if s["res"] in ("Goal", "SavedShot")]
    return sum(s["xg"] for s in on) - sum(s["res"] == "Goal" for s in on), len(on)


# ================================================================ pipeline
def build():
    matches, hist = load_matches()
    matches = odds_join(matches, DATA)
    ext = load_ext()

    teams = defaultdict(Team)
    players = defaultdict(Player)
    elo = defaultdict(lambda: 1500.0)
    league_goals = defaultdict(lambda: deque(maxlen=380))
    h2h = defaultdict(lambda: deque(maxlen=4))           # (ligue, att, def) -> xG marqués
    form_vs = defaultdict(lambda: [0.0, 0.0])             # (formation att, formation déf) -> [xG, n]
    form_base = defaultdict(lambda: [0.0, 0.0])           # formation att -> [xG, n]
    ref = defaultdict(lambda: {"cards": Ewma(0.05), "fouls": Ewma(0.05)})
    table = defaultdict(lambda: defaultdict(lambda: [0, 0, 0]))   # (lg, saison) -> équipe -> [pts, j, diff]
    n_teams = {}
    rows, tokens = [], []

    for (lg, s), g in matches.groupby(["league", "season"]):
        n_teams[(lg, s)] = len(set(g.home) | set(g.away))

    pending, pending_date = [], None
    for m in matches.itertuples():
        if m.date != pending_date:              # nouvelle heure de coup d'envoi : on applique
            for f in pending:                   # les résultats des matchs précédents
                f()
            pending, pending_date = [], m.date
        roster, shots = load_full(m.id)
        kh, ka = (m.league, m.home), (m.league, m.away)
        th, ta = teams[kh], teams[ka]
        lg_avg = np.mean(league_goals[m.league]) if league_goals[m.league] else 1.35
        day = m.date.normalize()

        # ---------------------------------------------------------- composition
        def lineup_feats(side, team, key):
            if roster is None:
                return {}, None
            xi = [p for p in roster[side] if p["start"]][:11]
            pids = [p["pid"] for p in xi]
            pl = {pid: players[pid] for pid in pids}
            av = {pid: pl[pid].availability(m.date, key, team) for pid in pids}
            team_xga, team_xgf = team.mid["xga"].get(), team.mid["xgf"].get()
            age = {pid: (m.date - ext["birth"][pid]).days / 365.25 for pid in pids if pid in ext["birth"]}
            gk = next((p["pid"] for p in xi if p["pos"] == "GK"), None)

            for slot, p in enumerate(xi):
                pid = p["pid"]
                others = [o for o in pids if o != pid]
                tokens.append({"match_id": m.id, "side": side, "slot": slot, "pid": pid,
                               "pos": p["pos"], "off": pl[pid].q("off"),
                               "chain": pl[pid].q("chain"), "buildup": pl[pid].q("buildup"),
                               "n_apps": pl[pid].n,
                               "chem_xi": np.mean([team.chem(pid, o) for o in others]),
                               "on_xga": pl[pid].onoff("a", team_xga),
                               "on_xgf": pl[pid].onoff("f", team_xgf),
                               "gk_q": pl[pid].gk_quality() if p["pos"] == "GK" else 0.0,
                               "age": age.get(pid, NAN), **av[pid]})

            off = {pid: pl[pid].q("off") for pid in pids}
            field = [p["pid"] for p in xi if p["pos"] != "GK"]
            returning = [pid for pid in pids if (av[pid]["missed_streak"] or 0) >= 3]
            defenders = [p["pid"] for p in xi if p["pos"] in DEF_POS]
            pair_chem = [team.chem(a, b) for a, b in combinations(pids, 2)]
            def_chem = [team.chem(a, b) for a, b in combinations(defenders, 2)]

            tot = defaultdict(int)
            for mins in list(team.recent_minutes)[-5:]:
                for pid, mn in mins.items():
                    tot[pid] += mn
            nmatch = max(min(len(team.recent_minutes), 5), 1)
            regulars = {pid for pid, mn in tot.items() if mn >= 0.6 * 90 * nmatch}
            missing = regulars - set(pids)
            squad = {pid for mins in team.recent_minutes for pid in mins} - set(pids)
            bench = sorted((players[pid].q("off") for pid in squad), reverse=True)[:5]
            rates = [av[pid]["absence_rate_6m"] for pid in pids]
            ages = list(age.values())
            form = formation(xi)
            return {
                "xi_off": sum(off[pid] for pid in field),
                "xi_chain": sum(pl[pid].q("chain") for pid in pids),
                "xi_buildup": sum(pl[pid].q("buildup") for pid in pids),
                "xi_top3_off": sum(sorted((off[pid] for pid in field), reverse=True)[:3]),
                "xi_onxgf": np.mean([pl[pid].onoff("f", team_xgf) for pid in field]),
                "def_line_onxga": np.mean([pl[pid].onoff("a", team_xga) for pid in defenders]) if defenders else NAN,
                "gk_prev": pl[gk].gk_quality() if gk else NAN,
                "gk_new": int(gk is not None and pl[gk].n < 3),
                "n_missing_regulars": len(missing),
                "missing_off": sum(players[pid].q("off") for pid in missing),
                "missing_def_onxga": np.mean([players[pid].onoff("a", team_xga) for pid in missing]) if missing else NAN,
                "bench_off": sum(bench),
                "continuity": len(set(pids) & team.last_xi) / 11 if team.last_xi else NAN,
                "xi_new_players": sum(pl[pid].n < 3 for pid in pids),
                "xi_returning_n": len(returning),
                "xi_returning_off": sum(off[pid] for pid in returning),
                "xi_back_from_red": sum(av[pid]["back_from_red"] for pid in pids),
                "xi_absence_rate": np.nanmean(rates) if any(not math.isnan(r) for r in rates) else NAN,
                "xi_load_30d": np.mean([av[pid]["load_30d"] for pid in pids]),
                "xi_low_load_n": sum(av[pid]["load_30d"] < 90 for pid in pids),
                "chem_xi": np.mean(pair_chem) if pair_chem else NAN,
                "chem_def": np.mean(def_chem) if def_chem else NAN,
                "chem_min_pair": np.min(pair_chem) if pair_chem else NAN,
                "xi_age": np.mean(ages) if len(ages) >= 6 else NAN,
                "xi_age_std": np.std(ages) if len(ages) >= 6 else NAN,
                "formation_change": int(team.last_formation is not None and form != team.last_formation),
            }, form

        lf_h, form_h = lineup_feats("h", th, kh)
        lf_a, form_a = lineup_feats("a", ta, ka)
        lf = {"h": lf_h, "a": lf_a}
        forms = {"h": form_h, "a": form_a}

        # ---------------------------------------------------------- équipe
        def team_feats(t):
            f = {k: t.mid[k].get() for k in TEAM_EWMA}
            f.update({f"{k}_short": t.short[k].get() for k in t.short})
            f.update({f"{k}_long": t.long[k].get() for k in t.long})
            f["xgf_trend"] = f["xgf_short"] - f["xgf_long"]          # forme au-dessus de sa moyenne ?
            f["xga_trend"] = f["xga_short"] - f["xga_long"]
            f["pts_trend"] = f["pts_short"] - f["pts_long"]
            f["finishing"] = f["gf_long"] - f["xgf_long"]            # réussite devant le but
            f["keeping"] = f["xga_long"] - f["ga_long"]              # réussite défensive
            f["luck_pts"] = f["pts_long"] - f["xpts_long"]
            f["level_xgf90"] = t.level_f.get(90, 30)
            f["level_xga90"] = t.level_a.get(90, 30)
            f["close_xgf90"] = t.close_f.get(90, 30)
            f["close_xga90"] = t.close_a.get(90, 30)
            f["n_played"] = t.n
            f["home_xgd"], f["away_xgd"] = t.home_xgd.get(), t.away_xgd.get()
            return f

        # ---------------------------------------------------------- contexte
        tab = table[(m.league, m.season)]
        nt = n_teams[(m.league, m.season)]

        def standing(team):
            ranked = sorted(tab.items(), key=lambda kv: (-kv[1][0], -kv[1][2]))
            pts = {t: v[0] for t, v in ranked}
            order = [t for t, _ in ranked]
            me = tab[team][0] if team in tab else 0
            played = tab[team][1] if team in tab else 0
            def pts_at(rank):
                return pts[order[rank - 1]] if len(order) >= rank else 0
            remaining = 2 * (nt - 1) - played
            gaps = {"gap_top": pts_at(1) - me, "gap_ucl": pts_at(4) - me,
                    "gap_safe": me - pts_at(nt - 3)}  # >0 : au-dessus de la zone rouge
            stake_none = int(played >= 20 and gaps["gap_ucl"] > 3 * remaining and
                             gaps["gap_safe"] > 3 * remaining)
            return {"rank": order.index(team) + 1 if team in order else NAN, "played": played,
                    "remaining": remaining, "season_progress": played / (2 * (nt - 1)),
                    "nothing_to_play": stake_none, **gaps}

        def calendar(team_name, t):
            eu = ext["euro"].get(team_name, [])
            i = bisect.bisect_left(eu, day)
            last_eu = (day - eu[i - 1]).days if i > 0 else NAN
            next_eu = (eu[i] - day).days if i < len(eu) and eu[i] > day else NAN
            n_eu_30 = i - bisect.bisect_left(eu, day - pd.Timedelta(days=30))
            n_lg_14 = t.matches_between(m.date - pd.Timedelta(days=14), m.date)
            season_start = pd.Timestamp(m.season, 7, 1)
            in_eu = int(any(season_start <= d < season_start + pd.DateOffset(years=1) for d in eu))
            cc = ext["coach"].get(team_name, [])
            j = bisect.bisect_left(cc, day)
            coach_days = (day - cc[j - 1]).days if j > 0 else NAN
            rest = (m.date - t.dates[-1]).days if t.dates else NAN
            last_any = min([x for x in (rest, last_eu) if not math.isnan(x)], default=NAN)
            return {"euro_last_days": last_eu, "euro_next_days": next_eu,
                    "euro_next_soon": int(not math.isnan(next_eu) and next_eu <= 4),
                    "euro_n_30d": n_eu_30, "in_europe": in_eu,
                    "matches_14d": n_lg_14 + (i - bisect.bisect_left(eu, day - pd.Timedelta(days=14))),
                    "rest": rest, "rest_any": last_any,
                    "after_break": int(not math.isnan(rest) and 12 <= rest <= 20),
                    "coach_days": min(coach_days, 1500) if not math.isnan(coach_days) else NAN,
                    "coach_new": int(not math.isnan(coach_days) and coach_days <= 60)}

        vh, va = ext["venue"].get(m.home), ext["venue"].get(m.away)
        travel = haversine(vh, va) if vh and va else NAN
        temp = rain = wind = NAN
        if vh and vh[2] in ext["weather"]:
            w = ext["weather"][vh[2]]
            hour = m.date.floor("h")
            if hour in w.index:
                temp, rain, wind = w.loc[hour, ["temperature_2m", "precipitation", "wind_speed_10m"]]
        ref_name = m.Referee if isinstance(m.Referee, str) else None
        ref_f = {"ref_cards": ref[ref_name]["cards"].get() if ref_name else NAN,
                 "ref_fouls": ref[ref_name]["fouls"].get() if ref_name else NAN}

        # marché : cotes d'ouverture, marge retirée (variante « modèle + marché »)
        def market(side):
            o = [m.PSH, m.PSD, m.PSA] if not np.isnan(m.PSH) else [m.AvgH, m.AvgD, m.AvgA]
            if any(np.isnan(o)):
                return {"mkt_win": NAN, "mkt_draw": NAN, "mkt_lose": NAN}
            p = np.array([1 / x for x in o]); p /= p.sum()
            win, lose = (p[0], p[2]) if side == "h" else (p[2], p[0])
            return {"mkt_win": win, "mkt_draw": p[1], "mkt_lose": lose}

        # ---------------------------------------------------------- lignes
        for side, att, dfn, katt, kdef, goals in (("h", th, ta, kh, ka, m.hg),
                                                  ("a", ta, th, ka, kh, m.ag)):
            other = "a" if side == "h" else "h"
            fa, fd = forms[side], forms[other]
            fv = form_vs[(fa, fd)] if fa and fd else [0, 0]
            fb = form_base[fa] if fa else [0, 0]
            k = 30.0
            base = fb[0] / fb[1] if fb[1] else 1.35
            hh = h2h[(m.league, katt[1], kdef[1])]
            row = {"match_id": m.id, "date": m.date, "league": m.league, "season": m.season,
                   "team": katt[1], "opp": kdef[1], "is_home": int(side == "h"),
                   "goals": goals, "lg_avg_goals": lg_avg,
                   "elo_diff": elo[katt] - elo[kdef] + (60 if side == "h" else -60),
                   "elo_att": elo[katt], "elo_def": elo[kdef],
                   "form_att": fa or "?", "form_def": fd or "?",
                   "form_matchup_xg": (fv[0] + k * base) / (fv[1] + k) - base if fa and fd else NAN,
                   "form_att_xg": (fb[0] + k * 1.35) / (fb[1] + k) if fa else NAN,
                   "h2h_n": len(hh), "h2h_xg": np.mean(hh) if hh else NAN,
                   "travel_km": travel, "temp": temp, "rain": rain, "wind": wind,
                   "weekday": m.date.weekday(), "hour": m.date.hour, **ref_f, **market(side)}
            row.update({f"att_{k2}": v for k2, v in team_feats(att).items()})
            row.update({f"def_{k2}": v for k2, v in team_feats(dfn).items()})
            row.update({f"att_{k2}": v for k2, v in lf[side].items()})
            row.update({f"def_{k2}": v for k2, v in lf[other].items()})
            row.update({f"att_{k2}": v for k2, v in standing(katt[1]).items()})
            row.update({f"def_{k2}": v for k2, v in standing(kdef[1]).items()})
            row.update({f"att_{k2}": v for k2, v in calendar(katt[1], att).items()})
            row.update({f"def_{k2}": v for k2, v in calendar(kdef[1], dfn).items()})
            rows.append(row)

        # ================= mises à jour APRÈS avoir écrit le match (anti-fuite)
        # Elles sont différées : tous les matchs qui commencent à la même heure sont
        # écrits AVANT qu'aucun d'eux ne mette à jour les historiques.
        def _update(m=m, roster=roster, shots=shots, kh=kh, ka=ka, th=th, ta=ta,
                    form_h=form_h, form_a=form_a, forms=forms, ref_name=ref_name, tab=tab):
            e = elo_expect(elo[kh] + 60, elo[ka])
            res = 1.0 if m.hg > m.ag else 0.5 if m.hg == m.ag else 0.0
            kk = 20 * math.log(abs(m.hg - m.ag) + 1) + 20
            elo[kh] += kk * (res - e)
            elo[ka] -= kk * (res - e)
            league_goals[m.league].append((m.hg + m.ag) / 2)
            for t_, pts in ((m.home, 3 if m.hg > m.ag else 1 if m.hg == m.ag else 0),
                            (m.away, 3 if m.ag > m.hg else 1 if m.hg == m.ag else 0)):
                tab[t_][0] += pts; tab[t_][1] += 1
            tab[m.home][2] += m.hg - m.ag; tab[m.away][2] += m.ag - m.hg
            h2h[(m.league, m.home, m.away)].append(m.hxg)
            h2h[(m.league, m.away, m.home)].append(m.axg)
            if form_h and form_a:
                for fa, fd, xg in ((form_h, form_a, m.hxg), (form_a, form_h, m.axg)):
                    form_vs[(fa, fd)][0] += xg; form_vs[(fa, fd)][1] += 1
                    form_base[fa][0] += xg; form_base[fa][1] += 1
            if ref_name:
                ref[ref_name]["cards"].add((m.HY + m.AY + 2 * (m.HR + m.AR)))
                ref[ref_name]["fouls"].add(m.HF + m.AF)

            gs = game_state(shots["h"], shots["a"]) if shots else None
            for side, key, t, xgf, xga, gf, ga, team_name in (
                    ("h", kh, th, m.hxg, m.axg, m.hg, m.ag, m.home),
                    ("a", ka, ta, m.axg, m.hxg, m.ag, m.hg, m.away)):
                other = "a" if side == "h" else "h"
                pts = 3 if gf > ga else 1 if gf == ga else 0
                hh = hist.get((team_name, m.dt), {})
                vals = {"xgf": xgf, "xga": xga, "gf": gf, "ga": ga, "pts": pts,
                        "draw": float(gf == ga), "xpts": hh.get("xpts", NAN)}
                if hh:
                    pp, pa = hh["ppda"], hh["ppda_allowed"]
                    vals.update({"ppda": pp["att"] / pp["def"] if pp["def"] else NAN,
                                 "ppda_allowed": pa["att"] / pa["def"] if pa["def"] else NAN,
                                 "deep": hh["deep"], "deep_allowed": hh["deep_allowed"]})
                if shots:
                    sf, sa = shot_profile(shots[side]), shot_profile(shots[other])
                    vals.update({"npxgf": sf["npxg"], "npxga": sa["npxg"], "sp_xgf": sf["sp"],
                                 "sp_xga": sa["sp"], "op_xgf": sf["op"], "op_xga": sa["op"],
                                 "xg_per_shot_f": sf["per_shot"], "xg_per_shot_a": sa["per_shot"],
                                 "box_share_f": sf["box"], "head_share_f": sf["head"],
                                 "big_ch_f": sf["big"], "big_ch_a": sa["big"]})
                    (lvl_h, cl_h), (lvl_a, cl_a) = gs[0]["h"], gs[0]["a"]
                    mine, theirs = (gs[0][side], gs[0][other])
                    t.level_f.add(mine[0], gs[1]); t.level_a.add(theirs[0], gs[1])
                    t.close_f.add(mine[1], gs[2]); t.close_a.add(theirs[1], gs[2])
                hs = {"h": ("HS", "HST", "HC", "HF", "HY", "HR"), "a": ("AS", "AST", "AC", "AF", "AY", "AR")}
                me_c, op_c = hs[side], hs[other]
                g = lambda c: getattr(m, c)
                vals.update({"shots_f": g(me_c[0]), "shots_a": g(op_c[0]), "sot_f": g(me_c[1]),
                             "sot_a": g(op_c[1]), "corners_f": g(me_c[2]), "corners_a": g(op_c[2]),
                             "fouls_f": g(me_c[3]), "fouls_a": g(op_c[3]), "yellow_f": g(me_c[4]),
                             "red_f": g(me_c[5])})
                for k2, v in vals.items():
                    if k2 in t.mid:
                        t.mid[k2].add(v)
                    if k2 in t.short:
                        t.short[k2].add(v)
                    if k2 in t.long:
                        t.long[k2].add(v)
                (t.home_xgd if side == "h" else t.away_xgd).add(xgf - xga)

                if roster:
                    ps = [p for p in roster[side] if p["min"] > 0]
                    for p in ps:
                        players[p["pid"]].update(p, m.date, key, xgf, xga)
                    for a, b in combinations(ps, 2):
                        t.add_chem(a["pid"], b["pid"], min(a["min"], b["min"]))
                    gk = next((p["pid"] for p in roster[side] if p["pos"] == "GK" and p["start"]), None)
                    if gk and shots:
                        prev, n_on = gk_faced(shots[other])
                        players[gk].gk_prev = 0.98 * players[gk].gk_prev + prev
                        players[gk].gk_shots = 0.98 * players[gk].gk_shots + n_on
                    t.last_xi = {p["pid"] for p in roster[side] if p["start"]}
                    t.recent_minutes.append({p["pid"]: p["min"] for p in roster[side]})
                    t.last_formation = forms[side]
                t.n += 1
                t.dates.append(m.date)
        pending.append(_update)
    for f in pending:
        f()

    df = pd.DataFrame(rows)
    suffix = "_cut" if os.environ.get("CUTOFF") else "_drop" if os.environ.get("DROP_ID") else ""
    df.to_parquet(f"{DATA}/features{suffix}.parquet")
    pd.DataFrame(tokens).to_parquet(f"{DATA}/player_tokens{suffix}.parquet")
    if not suffix:
        matches.to_parquet(f"{DATA}/matches.parquet")
    print(df.shape, len(tokens), df["date"].min(), df["date"].max())


if __name__ == "__main__":
    build()
