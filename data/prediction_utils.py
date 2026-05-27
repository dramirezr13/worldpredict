"""Utilidades de predicción: alias de equipos, estadísticas desde BD y fallback."""

from sqlalchemy import text

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


KNOCKOUT_STAGES = frozenset({
    "Round of 32", "Round of 16", "Quarter-finals", "Semi-finals", "Final",
})


def pick_outcome_from_probs(probs: dict, stage: str = "Group Stage") -> str:
    """'home' | 'draw' | 'away' — misma lógica que la simulación del torneo."""
    h, d, a = probs["home_win"], probs["draw"], probs["away_win"]
    allow_draw = stage not in KNOCKOUT_STAGES
    if allow_draw and d >= h and d >= a:
        return "draw"
    if h >= a:
        return "home"
    return "away"


def estimate_score_from_probs(probs: dict, stage: str = "Group Stage") -> tuple[int, int]:
    """Marcador estimado coherente con probabilidades y fase."""
    outcome = pick_outcome_from_probs(probs, stage)
    if outcome == "home":
        return (2, 0) if probs["home_win"] > 55 else (2, 1)
    if outcome == "away":
        return (0, 2) if probs["away_win"] > 55 else (1, 2)
    return (1, 1)


def attach_predicted_score(result: dict, stage: str = "Group Stage") -> dict:
    hs, as_ = estimate_score_from_probs(result, stage)
    result["predicted_score"] = {"home": hs, "away": as_}
    result["score_display"] = f"{hs} - {as_}"
    return result
