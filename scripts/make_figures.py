"""Figures used in the README (static PNG, light theme).

    python scripts/make_figures.py

Reads results/*.csv produced by the pipeline and writes docs/figures/*.png.
Palette validated for colour-vision deficiency (blue #2a78d6 / orange #eb6834).
"""
import os

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RES, OUT = "results", "docs/figures"
os.makedirs(OUT, exist_ok=True)
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
BLUE, ORANGE, NEUTRAL = "#2a78d6", "#eb6834", "#8a8984"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK,
    "text.color": INK, "font.size": 10.5, "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "axes.titlesize": 12.5, "axes.titleweight": "semibold",
    "axes.titlelocation": "left", "axes.titlepad": 12,
})


def dot_ci(ax, labels, mean, lo, hi, color, offset=0.0, label=None):
    ys = [i + offset for i in range(len(labels))]
    ax.hlines(ys, lo, hi, color=color, linewidth=2.2, zorder=2)
    ax.scatter(mean, ys, s=64, color=color, edgecolor=SURFACE, linewidth=2, zorder=3, label=label)
    return ys


def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"{OUT}/{name}", dpi=170)
    plt.close(fig)
    print("écrit", f"{OUT}/{name}")


# ---------------------------------------------------------------- 1. gap to the market
cmp_ = pd.read_csv(f"{RES}/wf_comparisons.csv")
LABELS = {
    ("LightGBM sans compositions", "Bookmakers (ouverture)"): "Model, no lineups  vs  opening odds",
    ("LightGBM sans compositions + cotes d'ouverture en variables", "Bookmakers (ouverture)"):
        "Model + odds as features  vs  opening odds",
    ("Mélange sans compositions + marché d'ouverture", "Bookmakers (ouverture)"): "Model ⊕ market blend  vs  opening odds",
    ("LightGBM avec compositions", "Bookmakers (clôture)"): "Model, with lineups  vs  closing odds",
    ("Mélange avec compositions + marché de clôture", "Bookmakers (clôture)"): "Model ⊕ market blend  vs  closing odds",
}
c = cmp_[[(r.model, r.reference) in LABELS for r in cmp_.itertuples()]].iloc[::-1]
labels = [LABELS[(r.model, r.reference)] for r in c.itertuples()]
fig, ax = plt.subplots(figsize=(8.6, 3.6))
ax.axvline(0, color=INK2, linewidth=1.2, zorder=1)
ys = dot_ci(ax, labels, c["diff"], c["lo"], c["hi"], BLUE)
ax.set_yticks(ys, labels)
ax.set_xlabel("Log-loss difference with the bookmakers (95 % CI)  ·  < 0 = better than the market")
ax.text(0, len(labels) - 0.35, " market", color=INK2, va="bottom", fontsize=9.5)
ax.set_title("No model beats the market it is fairly compared with  (3,799 matches, Jul 2024 – Oct 2026)")
ax.grid(axis="y", visible=False)
save(fig, "market_gap.png")

# ---------------------------------------------------------------- 2. feature families
ab = pd.read_csv(f"{RES}/ablation_families.csv").sort_values("gain_add")
fam = [f[2:] for f in ab.family]
fig, ax = plt.subplots(figsize=(8.6, 4.6))
ax.axvline(0, color=INK2, linewidth=1.2, zorder=1)
dot_ci(ax, fam, ab.gain_add, ab.gain_add_lo, ab.gain_add_hi, BLUE, offset=0.15,
       label="added to team strength alone")
ys = dot_ci(ax, fam, ab.gain_unique, ab.gain_unique_lo, ab.gain_unique_hi, ORANGE, offset=-0.15,
            label="removed from the full model (unique information)")
ax.set_yticks(range(len(fam)), fam)
ax.set_xlabel("Log-loss improvement (95 % CI)  ·  > 0 = the family helps")
ax.set_title("Which kinds of football statistics carry predictive signal?  (tuning period, 2,819 matches)")
ax.legend(loc="lower right", frameon=False, fontsize=9.5)
ax.grid(axis="y", visible=False)
save(fig, "feature_families.png")

# ---------------------------------------------------------------- 3. favourite-longshot bias
ms = pd.read_csv(f"{RES}/market_study.csv")
b = ms[ms["stratégie"].str.startswith("B tranche")].copy()
fig, ax = plt.subplots(figsize=(5.6, 5.0))
ax.plot([0, 0.9], [0, 0.9], color=NEUTRAL, linewidth=1.2, linestyle=(0, (4, 3)), zorder=1)
ax.text(0.62, 0.66, "perfect calibration", color=INK2, rotation=41, fontsize=9)
ax.plot(b.proba, b.freq, color=BLUE, linewidth=2, zorder=2)
ax.scatter(b.proba, b.freq, s=56, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=3)
for r in b.itertuples():
    if r.proba < 0.2 or r.proba > 0.6:
        ax.annotate(f"ROI {r.roi:+.0%}", (r.proba, r.freq), textcoords="offset points",
                    xytext=(8, -12), fontsize=9, color=INK2)
ax.set_xlabel("Probability implied by closing odds (proportional de-vig)")
ax.set_ylabel("Observed frequency")
ax.set_title("Favourite-longshot bias  (9,200 matches)")
ax.set_xlim(0, 0.9); ax.set_ylim(0, 0.9)
save(fig, "favourite_longshot.png")

# ---------------------------------------------------------------- 4. by league
lg = cmp_[cmp_.model.str.startswith("league:")].copy()
lg["name"] = lg.model.str[7:].replace({"EPL": "Premier League", "La_liga": "La Liga", "Serie_A": "Serie A",
                                       "Ligue_1": "Ligue 1"})
lg = lg.sort_values("diff", ascending=False)
d = ms[ms["stratégie"].str.startswith("D ligue")].copy()
d["name"] = d["stratégie"].str.replace("D ligue : ", "", regex=False).replace(
    {"EPL": "Premier League", "La_liga": "La Liga", "Serie_A": "Serie A", "Ligue_1": "Ligue 1"})
d = d.set_index("name").loc[lg.name]
fig, axes = plt.subplots(1, 2, figsize=(10, 3.3), sharey=True)
ax = axes[0]
ax.axvline(0, color=INK2, linewidth=1.2)
ys = dot_ci(ax, list(lg.name), lg["diff"], lg.lo, lg.hi, BLUE)
ax.set_yticks(ys, list(lg.name))
ax.set_xlabel("Model − opening odds (log-loss, 95 % CI)")
ax.set_title("Model gap to the market, by league")
ax.grid(axis="y", visible=False)
ax = axes[1]
ax.axvline(0, color=INK2, linewidth=1.2)
dot_ci(ax, list(d.index), d["diff"], d.diff_lo, d.diff_hi, ORANGE)
ax.set_xlabel("Opening − closing odds (log-loss, 95 % CI)")
ax.set_title("How much the market learns before kick-off")
ax.grid(axis="y", visible=False)
save(fig, "by_league.png")
