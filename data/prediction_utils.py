"""Utilidades de predicción: alias de equipos, estadísticas desde BD y fallback."""

import hashlib
import random

from sqlalchemy import text

from worldcup_2026 import WORLD_CUP_CHAMPIONS

# +10% relativo a la probabilidad de victoria de un campeón del mundo
CHAMPION_WIN_BOOST = 1.10

# Nombre en worldcup_2026 / UI -> nombre en label_encoder.pkl
MODEL_TEAM_ALIASES = {
    "South Korea": "Korea Republic",
    "United States": "USA",
    "Iran": "IR Iran",
    "Bosnia and Herzegovina": "Bosnia and Herzegovina",
    "DR Congo": "Zaire",
}

DEFAULT_STATS = {
    "wins": 0,
    "draws": 0,
    "losses": 0,
    "goals_for": 0,
    "goals_against": 0,
    "matches_played": 0,
    "win_rate": 0.33,
    "goal_diff": 0,
}


def resolve_model_team(team: str, label_classes: set) -> str | None:
    candidates = [team, MODEL_TEAM_ALIASES.get(team)]
    for name in candidates:
        if name and name in label_classes:
            return name
    return None


def _stats_from_row(row) -> dict:
    mp = int(row["matches_played"] or 0)
    wins = int(row["wins"] or 0)
    draws = int(row["draws"] or 0)
    gf = int(row["goals_for"] or 0)
    ga = int(row["goals_against"] or 0)
    return {
        "wins": wins,
        "draws": draws,
        "losses": int(row["losses"] or 0),
        "goals_for": gf,
        "goals_against": ga,
        "matches_played": mp,
        "win_rate": wins / (mp + 1),
        "goal_diff": gf - ga,
    }


def get_team_stats_from_team_stats(conn, team: str, season: int) -> dict | None:
    names = [team, MODEL_TEAM_ALIASES.get(team)]
    names = [n for n in names if n]
    for name in names:
        row = conn.execute(
            text("""
                SELECT wins, draws, losses, goals_for, goals_against, matches_played
                FROM team_stats
                WHERE team = :team
                ORDER BY ABS(season - :season) ASC
                LIMIT 1
            """),
            {"team": name, "season": season},
        ).fetchone()
        if row:
            return _stats_from_row(dict(row._mapping))
    return None


def get_team_stats_from_worldcup(conn, team: str) -> dict | None:
    row = conn.execute(
        text("""
            SELECT
                COUNT(*) AS matches_played,
                SUM(CASE
                    WHEN home_team_name = :team AND home_team_score > away_team_score THEN 1
                    WHEN away_team_name = :team AND away_team_score > home_team_score THEN 1
                    ELSE 0
                END) AS wins,
                SUM(CASE
                    WHEN home_team_score = away_team_score THEN 1
                    ELSE 0
                END) AS draws,
                SUM(CASE
                    WHEN home_team_name = :team AND home_team_score < away_team_score THEN 1
                    WHEN away_team_name = :team AND away_team_score < home_team_score THEN 1
                    ELSE 0
                END) AS losses,
                SUM(CASE
                    WHEN home_team_name = :team THEN home_team_score
                    ELSE away_team_score
                END) AS goals_for,
                SUM(CASE
                    WHEN home_team_name = :team THEN away_team_score
                    ELSE home_team_score
                END) AS goals_against
            FROM worldcup_matches
            WHERE home_team_name = :team OR away_team_name = :team
        """),
        {"team": team},
    ).fetchone()
    if not row or not row.matches_played:
        return None
    return _stats_from_row(dict(row._mapping))


def get_team_stats_combined(conn, team: str, season: int = 2026) -> dict:
    ts = get_team_stats_from_team_stats(conn, team, season)
    wc = get_team_stats_from_worldcup(conn, team)
    if ts and wc:
        return ts if ts["matches_played"] >= wc["matches_played"] else wc
    if ts and ts["matches_played"] > 0:
        return ts
    if wc:
        return wc
    return dict(DEFAULT_STATS)


def predict_from_stats(home_stats: dict, away_stats: dict) -> dict:
    """Probabilidades a partir de historial en BD (sin modelo ML)."""
    h_exp = home_stats["matches_played"]
    a_exp = away_stats["matches_played"]
    if h_exp == 0 and a_exp > 0:
        return _fixed_probs(22.0, 28.0, 50.0, "Visitante")
    if a_exp == 0 and h_exp > 0:
        return _fixed_probs(50.0, 28.0, 22.0, "Local")
    if h_exp == 0 and a_exp == 0:
        return _fixed_probs(38.0, 30.0, 32.0, "Local")

    h_mp = max(h_exp, 1)
    a_mp = max(a_exp, 1)
    h_strength = (
        home_stats["win_rate"] * 1.5
        + home_stats["goal_diff"] / h_mp * 0.15
        + 0.12
    )
    a_strength = (
        away_stats["win_rate"] * 1.5
        + away_stats["goal_diff"] / a_mp * 0.15
    )
    diff = h_strength - a_strength
    home_win = 0.34 + diff * 0.22
    away_win = 0.34 - diff * 0.22
    draw = 0.32
    total = home_win + draw + away_win
    home_win, draw, away_win = home_win / total, draw / total, away_win / total
    home_win = max(0.08, min(0.75, home_win))
    away_win = max(0.08, min(0.75, away_win))
    draw = max(0.12, 1.0 - home_win - away_win)
    total = home_win + draw + away_win
    probs = [home_win / total, draw / total, away_win / total]
    labels = ["Local", "Empate", "Visitante"]
    return _fixed_probs(
        round(probs[0] * 100, 1),
        round(probs[1] * 100, 1),
        round(probs[2] * 100, 1),
        labels[max(range(3), key=lambda i: probs[i])],
    )


def _fixed_probs(home_pct: float, draw_pct: float, away_pct: float, result: str) -> dict:
    return {
        "home_win": home_pct,
        "draw": draw_pct,
        "away_win": away_pct,
        "predicted_result": result,
        "method": "database_stats",
    }


def is_world_champion(team: str) -> bool:
    """True si la selección ha ganado al menos un Mundial masculino FIFA."""
    if not team:
        return False
    if team in WORLD_CUP_CHAMPIONS:
        return True
    alias = MODEL_TEAM_ALIASES.get(team)
    return bool(alias and alias in WORLD_CUP_CHAMPIONS)


def apply_champion_boost(home_team: str, away_team: str, probs: dict) -> dict:
    """Aumenta +10% la prob. de victoria de cada campeón del mundo y renormaliza."""
    if not is_world_champion(home_team) and not is_world_champion(away_team):
        return probs

    h = float(probs["home_win"])
    d = float(probs["draw"])
    a = float(probs["away_win"])
    if is_world_champion(home_team):
        h *= CHAMPION_WIN_BOOST
    if is_world_champion(away_team):
        a *= CHAMPION_WIN_BOOST

    total = h + d + a
    if total <= 0:
        return probs

    h, d, a = (h / total) * 100, (d / total) * 100, (a / total) * 100
    labels = ["Local", "Empate", "Visitante"]
    values = [h, d, a]
    result = {
        **probs,
        "home_win": round(h, 1),
        "draw": round(d, 1),
        "away_win": round(a, 1),
    }
    if "predicted_result" in probs:
        result["predicted_result"] = labels[max(range(3), key=lambda i: values[i])]
    return result


KNOCKOUT_STAGES = frozenset({
    "Round of 32", "Round of 16", "Quarter-finals", "Semi-finals", "Final",
})

# Misma codificación con la que se entrenó el modelo ML
STAGE_ENCODING = {
    "Group Stage": 1,
    "Round of 32": 2,
    "Round of 16": 2,
    "Quarter-finals": 3,
    "Semi-finals": 4,
    "Third place": 5,
    "Final": 6,
}

KNOCKOUT_HOME_EDGE = 3.0
STRENGTH_WEIGHT = 8.0


def encode_stage(stage: str) -> int:
    return STAGE_ENCODING.get(stage, 1)


def _team_strength(stats: dict | None) -> float:
    if not stats:
        return 0.5
    mp = max(int(stats.get("matches_played") or 0), 1)
    wr = float(stats.get("win_rate") or 0.33)
    if wr > 1:
        wr /= 100.0
    gd = float(stats.get("goal_diff") or 0) / mp
    return wr * 0.6 + (0.5 + gd * 0.15) * 0.4


def knockout_win_weights(probs: dict) -> tuple[float, float]:
    """Reparte la probabilidad de empate entre local y visitante (90'+ prórroga/penales)."""
    h = float(probs["home_win"])
    d = float(probs["draw"])
    a = float(probs["away_win"])
    if d <= 0:
        return max(h, 0.01), max(a, 0.01)
    ha = h + a
    if ha <= 0:
        return 50.0, 50.0
    h_eff = h + d * (h / ha)
    a_eff = a + d * (a / ha)
    return max(h_eff, 0.01), max(a_eff, 0.01)


def pick_outcome_from_probs(probs: dict, stage: str = "Group Stage") -> str:
    """'home' | 'draw' | 'away' — resultado más probable (determinista)."""
    h, d, a = probs["home_win"], probs["draw"], probs["away_win"]
    allow_draw = stage not in KNOCKOUT_STAGES
    if allow_draw and d >= h and d >= a:
        return "draw"
    if h >= a:
        return "home"
    return "away"


def sample_outcome_from_probs(
    probs: dict, stage: str = "Group Stage", rng: random.Random | None = None
) -> str:
    """Muestrea resultado según probabilidades (más realista en simulación)."""
    rng = rng or random.Random()
    if stage in KNOCKOUT_STAGES:
        h, a = knockout_win_weights(probs)
        return rng.choices(["home", "away"], weights=[h, a], k=1)[0]
    h = float(probs["home_win"])
    d = float(probs["draw"])
    a = float(probs["away_win"])
    return rng.choices(["home", "draw", "away"], weights=[h, d, a], k=1)[0]


def winner_from_score(home_team: str, away_team: str, hs: int, as_: int) -> str | None:
    if hs > as_:
        return home_team
    if as_ > hs:
        return away_team
    return None


def pick_knockout_winner(
    probs: dict,
    home_team: str,
    away_team: str,
    home_stats: dict | None = None,
    away_stats: dict | None = None,
    rng: random.Random | None = None,
) -> str:
    """Ganador en eliminatorias: probabilidades + historial + ventaja local."""
    rng = rng or random.Random()
    h, a = knockout_win_weights(probs)
    h += (_team_strength(home_stats) - _team_strength(away_stats)) * STRENGTH_WEIGHT
    h += KNOCKOUT_HOME_EDGE
    h = max(h, 0.01)
    a = max(a, 0.01)
    return home_team if rng.random() < h / (h + a) else away_team


def simulate_group_match(
    probs: dict,
    home_stats: dict | None,
    away_stats: dict | None,
    stage: str,
    rng: random.Random,
) -> tuple[int, int]:
    outcome = sample_outcome_from_probs(probs, stage, rng)
    return simulate_match_score(outcome, probs, home_stats, away_stats, rng=rng)


def simulate_knockout_match(
    home_team: str,
    away_team: str,
    probs: dict,
    home_stats: dict | None,
    away_stats: dict | None,
    rng: random.Random,
) -> tuple[int, int, str]:
    """Marcador y ganador coherentes en eliminatorias."""
    winner = pick_knockout_winner(
        probs, home_team, away_team, home_stats, away_stats, rng
    )
    outcome = "home" if winner == home_team else "away"
    for _ in range(8):
        hs, as_ = simulate_match_score(
            outcome, probs, home_stats, away_stats, rng=rng
        )
        if winner_from_score(home_team, away_team, hs, as_) == winner:
            return hs, as_, winner
    if winner == home_team:
        return 1, 0, winner
    return 0, 1, winner


def _match_seed(home: str, away: str, stage: str) -> int:
    key = f"{home}|{away}|{stage}".encode("utf-8")
    return int(hashlib.md5(key).hexdigest()[:8], 16)


def _goal_rates(stats: dict | None) -> tuple[float, float]:
    """Goles a favor/en contra por partido (estimados si faltan datos)."""
    if not stats:
        return 1.25, 1.25
    mp = max(int(stats.get("matches_played") or 0), 1)
    if stats.get("goals_for") is not None and stats.get("goals_against") is not None:
        return (
            max(0.4, float(stats["goals_for"]) / mp),
            max(0.4, float(stats["goals_against"]) / mp),
        )
    gd = float(stats.get("goal_diff") or 0) / mp
    wr = float(stats.get("win_rate") or 0.33)
    if wr > 1:
        wr /= 100.0
    gf = 1.05 + gd * 0.35 + wr * 0.55
    ga = max(0.55, 1.05 - gd * 0.3 + (1.0 - wr) * 0.2)
    return max(0.4, gf), max(0.4, ga)


def simulate_match_score(
    outcome: str,
    probs: dict,
    home_stats: dict | None = None,
    away_stats: dict | None = None,
    *,
    rng: random.Random | None = None,
) -> tuple[int, int]:
    """Marcador aleatorio pero realista, coherente con el resultado del partido."""
    rng = rng or random.Random()
    h_p = float(probs.get("home_win", 33))
    a_p = float(probs.get("away_win", 33))
    margin = abs(h_p - a_p)

    h_gf, h_ga = _goal_rates(home_stats)
    a_gf, a_ga = _goal_rates(away_stats)
    attack_home = h_gf + a_ga * 0.45 + 0.12
    attack_away = a_gf + h_ga * 0.45

    if outcome == "draw":
        lines = [(0, 0), (1, 1), (2, 2), (3, 3)]
        weights = [16.0, 52.0, 24.0, 8.0]
        if attack_home + attack_away > 2.8:
            weights[2] += 6
            weights[3] += 2
        return rng.choices(lines, weights=weights, k=1)[0]

    if outcome == "home":
        lines = [
            (1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2), (4, 1), (4, 2), (5, 1),
        ]
        weights = [26.0, 14.0, 22.0, 9.0, 10.0, 6.0, 4.0, 2.0, 1.0]
        if margin < 12:
            weights[0] += 12
            weights[2] += 4
        elif margin > 28:
            weights[1] += 8
            weights[3] += 5
            weights[4] += 4
        if attack_home > attack_away + 0.35:
            weights[1] += 5
            weights[3] += 3
        return rng.choices(lines, weights=weights, k=1)[0]

    lines = [
        (0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 3), (1, 4), (2, 4), (1, 5),
    ]
    weights = [26.0, 14.0, 22.0, 9.0, 10.0, 6.0, 4.0, 2.0, 1.0]
    if margin < 12:
        weights[0] += 12
        weights[2] += 4
    elif margin > 28:
        weights[1] += 8
        weights[3] += 5
        weights[4] += 4
    if attack_away > attack_home + 0.35:
        weights[1] += 5
        weights[3] += 3
    return rng.choices(lines, weights=weights, k=1)[0]


def _outcome_from_predicted(predicted_result: str | None, probs: dict, stage: str) -> str:
    if predicted_result == "Empate":
        return "draw"
    if predicted_result == "Local":
        return "home"
    if predicted_result == "Visitante":
        return "away"
    h, d, a = probs["home_win"], probs["draw"], probs["away_win"]
    if d >= h and d >= a and stage not in KNOCKOUT_STAGES:
        return "draw"
    return pick_outcome_from_probs(probs, stage)


def estimate_score_from_probs(
    probs: dict,
    stage: str = "Group Stage",
    predicted_result: str | None = None,
    home_stats: dict | None = None,
    away_stats: dict | None = None,
    home_team: str = "",
    away_team: str = "",
) -> tuple[int, int]:
    """Marcador estimado coherente con probabilidades, fase y resultado predicho."""
    outcome = _outcome_from_predicted(predicted_result, probs, stage)
    seed = _match_seed(home_team, away_team, stage)
    return simulate_match_score(
        outcome, probs, home_stats, away_stats, rng=random.Random(seed)
    )


def attach_predicted_score(result: dict, stage: str = "Group Stage") -> dict:
    probs = {
        "home_win": result["home_win"],
        "draw": result["draw"],
        "away_win": result["away_win"],
    }
    hs, as_ = estimate_score_from_probs(
        probs,
        stage,
        result.get("predicted_result"),
        result.get("home_stats"),
        result.get("away_stats"),
        result.get("home_team", ""),
        result.get("away_team", ""),
    )
    result["predicted_score"] = {"home": hs, "away": as_}
    result["score_display"] = f"{hs} - {as_}"
    return result
