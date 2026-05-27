"""Simulación completa del Mundial 2026 usando el motor de predicción."""

from itertools import combinations

import numpy as np

from prediction_utils import (
    get_team_stats_combined,
    predict_from_stats,
    resolve_model_team,
)
from worldcup_2026 import WORLD_CUP_2026, WORLD_CUP_2026_TEAMS

STAGE_ENC = {
    "Group Stage": 1,
    "Round of 32": 2,
    "Round of 16": 3,
    "Quarter-finals": 4,
    "Semi-finals": 5,
    "Final": 6,
}


def _groups_from_teams():
    groups = {}
    for name, _code, grp, _conf, _host in WORLD_CUP_2026_TEAMS:
        groups.setdefault(grp, []).append(name)
    return groups


def _predict_probs(conn, model, le, label_classes, home, away, stage, season=2026):
    hs = get_team_stats_combined(conn, home, season)
    as_ = get_team_stats_combined(conn, away, season)
    home_model = resolve_model_team(home, label_classes)
    away_model = resolve_model_team(away, label_classes)

    if model is not None and le is not None and home_model and away_model:
        stage_enc = STAGE_ENC.get(stage, 1)
        home_enc = le.transform([home_model])[0]
        away_enc = le.transform([away_model])[0]
        features = np.array([[
            home_enc, away_enc, stage_enc, season,
            hs["win_rate"], as_["win_rate"],
            hs["goal_diff"], as_["goal_diff"],
            hs["wins"], as_["wins"],
            hs["draws"], as_["draws"],
        ]])
        probs = model.predict_proba(features)[0]
        return {
            "home_win": float(probs[0]) * 100,
            "draw": float(probs[1]) * 100,
            "away_win": float(probs[2]) * 100,
            "method": "ml_model",
        }

    p = predict_from_stats(hs, as_)
    p["method"] = "database_stats"
    return p


def _scores_from_outcome(outcome, probs):
    if outcome == "home":
        return 2, 0 if probs["home_win"] > 55 else 1
    if outcome == "away":
        return 0 if probs["away_win"] > 55 else 1, 2
    return 1, 1


def _pick_outcome(probs, allow_draw=True):
    h, d, a = probs["home_win"], probs["draw"], probs["away_win"]
    if allow_draw and d >= h and d >= a:
        return "draw"
    if h >= a:
        return "home"
    return "away"


def _knockout_winner(probs, home, away):
    h, a = probs["home_win"], probs["away_win"]
    if h >= a:
        return home
    return away


def _new_standing(team):
    return {
        "team": team,
        "played": 0,
        "wins": 0,
        "draws": 0,
        "losses": 0,
        "gf": 0,
        "ga": 0,
        "points": 0,
    }


def _apply_result(standings, home, away, hs, as_):
    sh = standings[home]
    sa = standings[away]
    sh["played"] += 1
    sa["played"] += 1
    sh["gf"] += hs
    sh["ga"] += as_
    sa["gf"] += as_
    sa["ga"] += hs

    if hs > as_:
        sh["wins"] += 1
        sh["points"] += 3
        sa["losses"] += 1
    elif hs < as_:
        sa["wins"] += 1
        sa["points"] += 3
        sh["losses"] += 1
    else:
        sh["draws"] += 1
        sa["draws"] += 1
        sh["points"] += 1
        sa["points"] += 1


def _rank_standings(standings_dict):
    rows = list(standings_dict.values())
    for r in rows:
        r["gd"] = r["gf"] - r["ga"]
    rows.sort(key=lambda x: (-x["points"], -x["gd"], -x["gf"], x["team"]))
    return rows


def simulate_worldcup_2026(conn, model=None, le=None, label_classes=None):
    label_classes = label_classes or set()
    groups = _groups_from_teams()
    all_matches = []
    group_results = []
    third_places = []

    for grp, teams in sorted(groups.items()):
        standings = {t: _new_standing(t) for t in teams}
        for home, away in combinations(teams, 2):
            probs = _predict_probs(
                conn, model, le, label_classes, home, away, "Group Stage"
            )
            outcome = _pick_outcome(probs, allow_draw=True)
            hs, as_ = _scores_from_outcome(outcome, probs)
            _apply_result(standings, home, away, hs, as_)
            all_matches.append({
                "stage": "Group Stage",
                "group": grp,
                "home": home,
                "away": away,
                "home_score": hs,
                "away_score": as_,
                "home_win": probs["home_win"],
                "draw": probs["draw"],
                "away_win": probs["away_win"],
            })

        ranked = _rank_standings(standings)
        first, second = ranked[0], ranked[1]
        third = ranked[2]
        third_places.append({**third, "group": grp})
        group_results.append({
            "group": grp,
            "standings": ranked,
            "qualified": [first["team"], second["team"]],
        })

    third_places.sort(key=lambda x: (-x["points"], -x["gd"], -x["gf"]))
    best_thirds = [t["team"] for t in third_places[:8]]

    knockout_teams = []
    for gr in group_results:
        knockout_teams.extend(gr["qualified"])
    knockout_teams.extend(best_thirds)

    seed_rows = []
    for gr in group_results:
        for row in gr["standings"]:
            if row["team"] in knockout_teams:
                seed_rows.append(row)
    seed_rows.sort(key=lambda x: (-x["points"], -x["gd"], -x["gf"]))
    ordered = [r["team"] for r in seed_rows]

    if len(ordered) < 32:
        for t in knockout_teams:
            if t not in ordered:
                ordered.append(t)
    ordered = ordered[:32]

    rounds = [
        ("Round of 32", 16),
        ("Round of 16", 8),
        ("Quarter-finals", 4),
        ("Semi-finals", 2),
        ("Final", 1),
    ]
    knockout_matches = []
    current = ordered

    for stage_name, _n_matches in rounds:
        next_round = []
        for i in range(0, len(current), 2):
            home, away = current[i], current[i + 1]
            probs = _predict_probs(
                conn, model, le, label_classes, home, away, stage_name
            )
            winner = _knockout_winner(probs, home, away)
            hs, as_ = (2, 1) if winner == home else (1, 2)
            knockout_matches.append({
                "stage": stage_name,
                "home": home,
                "away": away,
                "home_score": hs,
                "away_score": as_,
                "winner": winner,
                "home_win": probs["home_win"],
                "draw": probs["draw"],
                "away_win": probs["away_win"],
            })
            next_round.append(winner)
        current = next_round

    champion = current[0]
    final = knockout_matches[-1]
    runner_up = final["home"] if final["winner"] == final["away"] else final["away"]

    return {
        "tournament": WORLD_CUP_2026["tournament_name"],
        "champion": champion,
        "runner_up": runner_up,
        "total_matches": len(all_matches) + len(knockout_matches),
        "group_stage_matches": len(all_matches),
        "knockout_matches": len(knockout_matches),
        "groups": group_results,
        "best_third_places": best_thirds,
        "knockout_bracket": knockout_matches,
        "group_matches": all_matches,
    }
