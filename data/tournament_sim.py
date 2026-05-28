"""Simulación completa del Mundial 2026 usando el motor de predicción."""

from __future__ import annotations

import random
from itertools import combinations

import numpy as np

from prediction_utils import (
    apply_champion_boost,
    encode_stage,
    get_team_stats_combined,
    knockout_win_weights,
    predict_from_stats,
    resolve_model_team,
    simulate_group_match,
    simulate_knockout_match,
)
from worldcup_2026 import WORLD_CUP_2026, WORLD_CUP_2026_TEAMS
from worldcup_2026_bracket import (
    build_group_qualifiers,
    build_knockout_round_fixtures,
    resolve_r32_pairing,
    validate_r32_pairings,
)


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
        stage_enc = encode_stage(stage)
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
        raw = {
            "home_win": float(probs[0]) * 100,
            "draw": float(probs[1]) * 100,
            "away_win": float(probs[2]) * 100,
            "method": "ml_model",
            "predicted_result": ["Local", "Empate", "Visitante"][int(np.argmax(probs))],
        }
        return apply_champion_boost(home, away, raw)

    p = predict_from_stats(hs, as_)
    p["method"] = "database_stats"
    return apply_champion_boost(home, away, p)


def _match_rng(home: str, away: str, stage: str) -> random.Random:
    return random.Random(f"{home}|{away}|{stage}|{random.random()}")


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
            hs_stats = get_team_stats_combined(conn, home)
            as_stats = get_team_stats_combined(conn, away)
            probs = _predict_probs(
                conn, model, le, label_classes, home, away, "Group Stage"
            )
            rng = _match_rng(home, away, f"Group Stage|{grp}")
            hs, as_ = simulate_group_match(
                probs, hs_stats, as_stats, "Group Stage", rng
            )
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
                "method": probs.get("method", "ml_model"),
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

    qual = build_group_qualifiers(group_results, best_thirds)
    team_group = qual["team_group"]
    r32_pairings = resolve_r32_pairing(qual)
    validate_r32_pairings(r32_pairings, team_group)

    rounds = [
        ("Round of 32", r32_pairings),
        ("Round of 16", None),
        ("Quarter-finals", None),
        ("Semi-finals", None),
        ("Final", None),
    ]
    knockout_matches = []
    match_winners: dict[int, str] = {}

    for stage_name, fixtures in rounds:
        if fixtures is None:
            fixtures = build_knockout_round_fixtures(stage_name, match_winners)

        for home, away, match_id in fixtures:
            hs_stats = get_team_stats_combined(conn, home)
            as_stats = get_team_stats_combined(conn, away)
            probs = _predict_probs(
                conn, model, le, label_classes, home, away, stage_name
            )
            rng = _match_rng(home, away, f"{stage_name}|{match_id}")
            hs, as_, winner = simulate_knockout_match(
                home, away, probs, hs_stats, as_stats, rng
            )
            h_eff, a_eff = knockout_win_weights(probs)
            match_winners[match_id] = winner
            knockout_matches.append({
                "stage": stage_name,
                "match_id": match_id,
                "home": home,
                "away": away,
                "home_group": team_group.get(home),
                "away_group": team_group.get(away),
                "home_score": hs,
                "away_score": as_,
                "winner": winner,
                "home_win": probs["home_win"],
                "draw": probs["draw"],
                "away_win": probs["away_win"],
                "ko_home_win": round(h_eff, 1),
                "ko_away_win": round(a_eff, 1),
                "method": probs.get("method", "ml_model"),
            })

    final = knockout_matches[-1]
    champion = final["winner"]
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
