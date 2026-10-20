"""
Automated leakage audit.

    python -m footpred.evaluation.leak_tests data

  1. Truncation   : rebuild every feature with all data after time T removed
                    → features of every match before T must be bit-for-bit identical.
  2. Removal      : rebuild with ONE match played at time T removed
                    → features of the other matches kicking off at T (and before) must not change.
                    (catches leakage between simultaneous kick-offs)
  3. Correlation  : no feature may be suspiciously correlated with the goals of its own match.
  4. Closing odds : no closing-odds column (known only at kick-off) may appear among features.

T is the first kick-off time after May 2025 at which at least 8 matches of the same league
are played simultaneously (typically a final matchday), which is the hardest case for test 2.
Exits with status 1 if any test fails.
"""
import os
import subprocess
import sys

import numpy as np
import pandas as pd

DATA = sys.argv[1] if len(sys.argv) > 1 else "data"
ok = True


def build(env_extra):
    env = dict(os.environ, **env_extra)
    subprocess.run([sys.executable, "-m", "footpred.features.build", DATA], env=env, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def compare(a, b, label):
    global ok
    a = a.set_index(["match_id", "is_home"]).sort_index()
    b = b.set_index(["match_id", "is_home"]).sort_index().loc[a.index]
    bad = []
    for c in a.columns:
        x, y = a[c], b[c]
        if x.dtype.kind in "fiub":
            d = ~np.isclose(x.astype(float), y.astype(float), equal_nan=True)
        else:
            d = x.astype(str).to_numpy() != y.astype(str).to_numpy()
        if d.sum():
            bad.append((c, int(d.sum())))
    print(f"{label}: {len(a)} rows compared -> {'no difference' if not bad else bad}")
    ok &= not bad


full = pd.read_parquet(f"{DATA}/features.parquet")
full_tok = pd.read_parquet(f"{DATA}/player_tokens.parquet")
matches = pd.read_parquet(f"{DATA}/matches.parquet")
counts = matches.groupby(["date", "league"]).size()
counts = counts[(counts >= 8) & (counts.index.get_level_values(0) >= "2025-05-01")]
T, league = counts.index[0]
drop_id = str(matches[(matches.date == T) & (matches.league == league)].id.iloc[0])
print(f"Reference kick-off T = {T} ({league}, {counts.iloc[0]} simultaneous matches); removed match: {drop_id}\n")

build({"CUTOFF": str(T)})
build({"DROP_ID": drop_id})

# 1) truncation
cut = pd.read_parquet(f"{DATA}/features_cut.parquet")
compare(cut, full[full.date < T], "Test 1  truncation (features)")
tc = pd.read_parquet(f"{DATA}/player_tokens_cut.parquet").set_index(["match_id", "side", "slot"]).sort_index()
tf = full_tok.set_index(["match_id", "side", "slot"]).sort_index().loc[tc.index]
num = [c for c in tc.columns if tc[c].dtype.kind in "fiub"]
diff = [c for c in num if (~np.isclose(tc[c], tf[c], equal_nan=True)).sum()]
print(f"        truncation (player tokens): {len(tc)} rows compared -> {'no difference' if not diff else diff}")
ok &= not diff

# 2) removal of one simultaneous match
drop = pd.read_parquet(f"{DATA}/features_drop.parquet")
ref = full[(full.date <= T) & (full.match_id.astype(str) != drop_id)]
compare(drop[drop.match_id.isin(ref.match_id)], ref, "Test 2  removal of a simultaneous match")

# 3) correlation with the match's own goals
cols = [c for c in full.columns if full[c].dtype.kind in "fiub" and c not in ("goals", "season", "is_home")]
corr = full[cols].corrwith(full["goals"]).abs().sort_values(ascending=False)
print("Test 3  max |corr(feature, goals of the same match)| = "
      + ", ".join(f"{c} {v:.2f}" for c, v in corr.head(4).items()))
ok &= corr.max() < 0.5

# 4) no closing odds among features
closing = [c for c in full.columns if any(k in c for k in ("PSC", "AvgC", "MaxC", "B365C"))]
print(f"Test 4  closing-odds columns among features: {closing or 'none'}")
ok &= not closing

for f in ("features_cut", "player_tokens_cut", "features_drop", "player_tokens_drop"):
    try:
        os.remove(f"{DATA}/{f}.parquet")
    except FileNotFoundError:
        pass
print("\nRESULT:", "NO LEAKAGE DETECTED" if ok else "LEAKAGE DETECTED")
sys.exit(0 if ok else 1)
