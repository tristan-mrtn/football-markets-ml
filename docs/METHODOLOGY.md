# Methodology

This document describes how the data are built, how models are evaluated, and every
safeguard against information leakage. The guiding principle is simple: **every number
in this repository must be reproducible by someone who only had access to the past.**

## 1. Data

| Source | Content | Coverage | Licence / terms |
|---|---|---|---|
| [Understat](https://understat.com) | match xG, every shot (minute, xG, situation, body part, location), every player's minutes / position / xG / xA / xGChain / xGBuildup, team PPDA and deep completions | 9,253 matches, Big-5 leagues, Aug 2021 → Oct 2026 | scraped politely (≈ 2 req/s); data **not** redistributed |
| [football-data.co.uk](https://www.football-data.co.uk) | 1X2 odds of each bookmaker, average and maximum, at opening and closing; shots, corners, fouls, cards; referee (England) | 99.4 % of matches joined | free for personal / research use; not redistributed |
| Wikipedia | European cup fixtures (UCL / UEL / UECL, all rounds), managerial changes, club pages | 1,409 European matches, 448 managerial changes | CC BY-SA |
| Wikidata | stadium coordinates (138 / 138 clubs), players' birth dates (65 % of starters) | | CC0 |
| Open-Meteo | hourly weather at the stadium | partial (rate-limited) | CC BY 4.0 |

Team names differ between sources. They are reconciled automatically (fuzzy matching with a
full-name tie-break, then majority vote on matches sharing date and score) and the least
certain matches are printed for manual review.

## 2. Leakage-free feature construction

All 220 features are produced by a single **chronological replay** of the 9,253 matches
(`footpred/features/build.py`). For each match the features are written from the current
state of the world, and only *then* is the state updated with the match result.

**Simultaneous kick-offs.** Matches that start at the same minute (e.g. a final matchday)
are all written before any of them updates the state; otherwise the second match processed
would "see" the first one's result through league tables or formation statistics.

**Information set at prediction time.**

| Information | Known when | Used by |
|---|---|---|
| everything about past matches | always | all models |
| fixtures (incl. future European matches), referee appointment, managerial appointments | days before | all models |
| starting XI | ≈ 1 h before kick-off | "with lineups" models only |
| opening odds | days before | "+ odds" variants, opening-market comparison |
| closing odds | at kick-off | **evaluation only, never a feature** |
| weather | observed values used as a proxy for the forecast available hours before | context family |

### Feature families

| Family | Examples |
|---|---|
| A team strength | Elo, exponentially-weighted xG / goals for and against, non-penalty xG, home/away split |
| B form / mean reversion | short (≈ 2 matches) vs long (≈ 23 matches) averages and their difference |
| S game-state xG | xG per 90 min while the score is level / within one goal |
| C lineup attack | sum of the starters' xG+xA, xGChain, xGBuildup per 90 (shrunk towards a prior) |
| D defence / goalkeeper | team xG conceded when each defender plays (on/off), goals prevented by the goalkeeper |
| E availability | missing regulars, players back from ≥ 3 missed matches or from suspension, 30-day minutes load, bench strength |
| F chemistry | minutes played together by every pair of starters (decaying), defensive-line chemistry |
| G tactics / style | formation (inferred from positions), formation-vs-formation xG, PPDA, deep completions, set-piece vs open-play xG, shot quality profile |
| H head-to-head | xG in the last meetings |
| I context | table position and stakes (gap to title / top 4 / relegation, nothing to play for), days of rest, European match before/after, managerial change, travel distance, weather, referee tendencies, squad age |
| J luck | goals − xG, points − expected points |
| K match stats | shots, shots on target, corners, fouls, cards |
| M market | opening-odds implied probabilities (only in the "+ odds" variants) |

## 3. Models

* **Dixon-Coles (1997)** — attack/defence Poisson model with low-score correction and time decay,
  refitted every month per league on the previous three seasons. Baseline using goals only.
* **LightGBM, Poisson objective** — two rows per match (each team attacking the other), predicting
  goals; the two expected-goal rates are combined into a score matrix with a Dixon-Coles
  correction ρ fitted on the tuning period, giving home/draw/away probabilities.
  Hyper-parameters: 60-trial random search on the tuning period only; the selected configuration
  is in `results/wf_lgb_best.json`.
* **Lineup transformer (experimental)** — learned player embeddings + per-player state, two
  self-attention layers over the 22 starters. It never beat LightGBM on the tuning period
  (log-loss 0.990 vs 0.979) and is not part of the final evaluation; code kept in
  `footpred/models/transformer.py`.

## 4. Evaluation protocol

| Period | Role |
|---|---|
| Aug 2021 → Dec 2021 | warm-up of the rolling statistics (never scored) |
| Jan 2022 → | training data (expanding window) |
| Jan 2023 → Jun 2024 | **tuning**: hyper-parameters, ρ, blend weights, feature-family ablation |
| **Jul 2024 → Oct 2026** | **evaluation**: 3,849 matches, never used for any choice |

**Walk-forward.** The model is refitted every two months on all matches completed before the
window and predicts the next two months. Two assertions in the code guarantee that the latest
training match strictly precedes the first predicted match.

**Metrics.** Multi-class log-loss (primary) and Brier score of the home/draw/away probabilities.
Differences between two forecasters are computed match by match and given with a 95 % paired
bootstrap confidence interval.

## 5. Fair comparison with the market

A model is only compared with the market that had **the same information**:

* models **without lineups** → **opening** odds; bets placed at opening prices;
* models **with lineups** → **closing** odds (the market also knows the line-ups); bets at closing prices.

Comparing a lineup model with opening odds — a mistake made in an earlier version of this
project — gives the model information the market did not yet have and inflates any "edge".

**Removing the bookmaker margin.** Implied probabilities are de-vigged with the **power
method** (find *k* such that Σ (1/oᵢ)ᵏ = 1). The common proportional method spreads the margin
evenly, whereas bookmakers load more margin on longshots; proportional de-vigging therefore
overstates longshot probabilities (12.8 % announced vs 10.9 % observed on the bets studied).

**Closing-line value (CLV)** — odds taken divided by the de-vigged closing odds, minus one — is a
much lower-variance signal of skill than profit. With proportional de-vigging, a simple
"outlier bookmaker" strategy shows a spurious **CLV of +10.7 %** while losing money; with the
power method its CLV is **−0.4 %**, consistent with its returns. This pitfall is easy to fall into.

**Model ⊕ market blends.** p ∝ p_model^w₁ · p_market^w₂, with (w₁, w₂) fitted on the tuning
period. If the model held information the market lacks, w₁ would be clearly positive and the
blend would beat the market.

## 6. Leakage audit

`python -m footpred.evaluation.leak_tests data` runs four automated tests:

1. **Truncation** — rebuild all features with every match after time T deleted; the features of
   all earlier matches (and the 154k player tokens) must be identical.
2. **Removal** — delete one match played at T, where T is a final matchday with ≥ 8 simultaneous
   matches; the features of the other matches at T must not change.
3. **Correlation** — no feature may correlate above 0.5 with the goals of its own match
   (the maximum is the opening-odds feature, ≈ 0.38).
4. **Closing odds** — no closing-odds column may exist among features.

### Bugs found and fixed during the project

| Bug | Effect | How it was found |
|---|---|---|
| a market-odds column (`mkt_draw`) captured by the regex of the "luck" family | a "pure" model silently used bookmaker odds; the luck family looked useful | inspection of feature importances |
| results of simultaneous kick-offs visible to each other | small leak through tables and pooled statistics | code review → automated test 2 |
| lineup models compared with opening odds | unfair advantage over the market | review of information sets |
| shot that scored assigned to the post-goal score state | game-state xG slightly biased | unit test |
| proportional de-vigging in CLV | spurious +10 % CLV on longshots | CLV inconsistent with returns |

All results in this repository were produced after these fixes.

## 7. Limitations

* Understat has no defensive event data (tackles, interceptions, pressures); defensive quality is
  only measured indirectly.
* Domestic cups are missing from the fatigue features; European matches are included.
* French-licensed bookmakers are not in the odds data; the "maximum" odds may include exchange
  prices whose commission is not modelled, so outlier-bookmaker results are indicative only.
* Weather coverage is partial; birth dates cover 65 % of starters.
* The evaluation spans ≈ 2.3 seasons: effects smaller than ≈ 0.003 log-loss cannot be resolved.
* Hyper-parameters were tuned before the final bug fixes, but only on the tuning period, so the
  evaluation period remains untouched.
