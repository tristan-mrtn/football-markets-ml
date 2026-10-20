"""Feature families: each feature column belongs to exactly one family (first match wins).

Market odds are matched first so that no odds column can ever end up in a "pure" model.
"""
import re

ID_COLS = ("match_id", "date", "league", "season", "team", "opp", "goals", "form_att", "form_def", "ok")

FAMILIES = {
    "M market odds (opening)": r"^mkt_",
    "A team strength": r"^(is_home|lg_avg_goals|elo_.*|league_cat|(att|def)_(xgf|xga|gf|ga|npxgf|npxga|n_played|home_xgd|away_xgd))$",
    "B form / mean reversion": r"_(short|long|trend)$",
    "S game-state xG": r"_(level|close)_xg[fa]90$",
    "C lineup attack": r"_(xi_off|xi_chain|xi_buildup|xi_top3_off)$",
    "D defence / goalkeeper": r"_(def_line_onxga|gk_prev|gk_new|xi_onxgf|missing_def_onxga)$",
    "E availability": r"_(n_missing_regulars|missing_off|bench_off|xi_new_players|xi_returning_n|xi_returning_off|"
                      r"xi_back_from_red|xi_absence_rate|xi_load_30d|xi_low_load_n|continuity)$",
    "F chemistry": r"_chem_",
    "G tactics / style": r"^(form_(att|def)_cat|form_matchup_xg|form_att_xg|.*_formation_change|.*_(ppda|ppda_allowed|deep|"
                        r"deep_allowed|sp_xg[fa]|op_xg[fa]|xg_per_shot_[fa]|box_share_f|head_share_f|big_ch_[fa]))$",
    "H head-to-head": r"^h2h_",
    "I context": r"^(travel_km|temp|rain|wind|weekday|hour|ref_.*|.*_(rank|played|remaining|season_progress|"
                 r"nothing_to_play|gap_top|gap_ucl|gap_safe|euro_.*|in_europe|matches_14d|rest|rest_any|after_break|"
                 r"coach_days|coach_new|xi_age|xi_age_std))$",
    "J luck": r"^(att|def)_(finishing|keeping|luck_pts|pts|xpts|draw)$",
    "K match stats": r"^(att|def)_(shots_[fa]|sot_[fa]|corners_[fa]|fouls_[fa]|yellow_f|red_f)$",
}

# Features that depend on the starting XI (published ~1 h before kick-off only)
LINEUP_KEYS = ("xi_", "chem_", "missing", "bench_off", "continuity", "def_line", "gk_prev", "gk_new",
               "n_missing_regulars", "formation_change", "form_")


def assign(columns):
    """Return {family: [columns]} with every usable column in exactly one family."""
    out, taken = {}, set()
    for fam, pat in FAMILIES.items():
        out[fam] = [c for c in columns if c not in ID_COLS and c not in taken and re.search(pat, c)]
        taken.update(out[fam])
    return out
