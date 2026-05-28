import os
import pickle
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

sys_path = os.path.join(os.path.dirname(__file__), "..", "data")
if sys_path not in __import__("sys").path:
    __import__("sys").path.insert(0, os.path.abspath(sys_path))

try:
    from worldcup_2026 import FIFA_DISPLAY_NAMES
    from prediction_utils import (
        apply_champion_boost,
        attach_predicted_score,
        get_team_stats_combined,
        predict_from_stats,
        resolve_model_team,
    )
    from tournament_sim import simulate_worldcup_2026
except ImportError:
    FIFA_DISPLAY_NAMES = {}
    get_team_stats_combined = predict_from_stats = resolve_model_team = None
    simulate_worldcup_2026 = None

load_dotenv()

app = Flask(__name__)
CORS(app)

engine = create_engine(os.getenv("DATABASE_URL"))

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

with open(os.path.join(BASE_DIR, "models/model.pkl"), "rb") as f:
    model = pickle.load(f)
with open(os.path.join(BASE_DIR, "models/label_encoder.pkl"), "rb") as f:
    le = pickle.load(f)

LABEL_CLASSES = set(le.classes_)

STAGE_MAP = {
    "Group Stage": 1, "Round of 16": 2, "Quarter-finals": 3,
    "Semi-finals": 4, "Third place": 5, "Final": 6
}

@app.route("/health")
def health():
    return jsonify({"status": "ok", "message": "WorldPredict API corriendo!"})

@app.route("/matches")
def get_matches():
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT match_id, home_team, away_team, home_score,
                   away_score, status, match_date, stage, season
            FROM matches ORDER BY match_date DESC LIMIT 50
        """))
        matches = [dict(row._mapping) for row in result]
    return jsonify(matches)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json or {}
    home = data.get("home_team")
    away = data.get("away_team")
    stage = data.get("stage", "Group Stage")
    season = int(data.get("season", 2026))

    if not home or not away:
        return jsonify({"error": "home_team y away_team son obligatorios"}), 400
    if home == away:
        return jsonify({"error": "Selecciona equipos diferentes"}), 400

    with engine.connect() as conn:
        hs = get_team_stats_combined(conn, home, season)
        as_ = get_team_stats_combined(conn, away, season)

        home_model = resolve_model_team(home, LABEL_CLASSES)
        away_model = resolve_model_team(away, LABEL_CLASSES)

        if not home_model or not away_model:
            result = apply_champion_boost(home, away, predict_from_stats(hs, as_))
            result.update({
                "home_team": home,
                "away_team": away,
                "home_stats": _public_stats(hs),
                "away_stats": _public_stats(as_),
                "note": "Predicción por historial en base de datos (equipo sin modelo ML).",
            })
            return jsonify(attach_predicted_score(result, stage))

        home_enc = le.transform([home_model])[0]
        away_enc = le.transform([away_model])[0]
        stage_enc = STAGE_MAP.get(stage, 1)
        features = np.array([[
            home_enc, away_enc, stage_enc, season,
            hs["win_rate"], as_["win_rate"],
            hs["goal_diff"], as_["goal_diff"],
            hs["wins"], as_["wins"],
            hs["draws"], as_["draws"],
        ]])
        probs = model.predict_proba(features)[0]

    ml_probs = apply_champion_boost(home, away, {
        "home_win": round(float(probs[0]) * 100, 1),
        "draw": round(float(probs[1]) * 100, 1),
        "away_win": round(float(probs[2]) * 100, 1),
        "predicted_result": ["Local", "Empate", "Visitante"][int(np.argmax(probs))],
        "method": "ml_model",
    })
    return jsonify(attach_predicted_score({
        "home_team": home,
        "away_team": away,
        "home_stats": _public_stats(hs),
        "away_stats": _public_stats(as_),
        **ml_probs,
    }, stage))


def _public_stats(s):
    return {
        "matches_played": s["matches_played"],
        "wins": s["wins"],
        "win_rate": round(s["win_rate"] * 100, 1),
        "goal_diff": s["goal_diff"],
    }

@app.route("/history/<team>")
def team_history(team):
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT home_team, away_team, home_score, away_score, stage, season
            FROM matches WHERE home_team = :team OR away_team = :team
            ORDER BY season DESC LIMIT 20
        """), {"team": team})
        matches = [dict(row._mapping) for row in result]
    return jsonify(matches)

@app.route("/worldcup/2026/teams")
def worldcup_2026_teams():
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT team_name, team_code, group_name, confederation, is_host
            FROM worldcup_qualified_teams
            WHERE tournament_id = 'WC-2026'
            ORDER BY group_name, team_name
        """)).fetchall()

    if not rows:
        return jsonify({
            "tournament_id": "WC-2026",
            "tournament_name": "2026 FIFA Men's World Cup",
            "count": 0,
            "teams": [],
            "groups": {},
            "message": "Ejecuta python data/load_worldcup.py para cargar los equipos.",
        })

    teams = []
    groups = {}
    for r in rows:
        display = FIFA_DISPLAY_NAMES.get(r.team_name, r.team_name)
        entry = {
            "team_name": r.team_name,
            "display_name": display,
            "team_code": r.team_code,
            "group": r.group_name,
            "confederation": r.confederation,
            "is_host": bool(r.is_host),
        }
        teams.append(entry)
        groups.setdefault(r.group_name, []).append(display)

    return jsonify({
        "tournament_id": "WC-2026",
        "tournament_name": "2026 FIFA Men's World Cup",
        "count": len(teams),
        "teams": teams,
        "groups": groups,
    })


@app.route("/worldcup/2026/simulate", methods=["POST"])
def simulate_worldcup():
    if simulate_worldcup_2026 is None:
        return jsonify({"error": "Módulo de simulación no disponible"}), 500
    with engine.connect() as conn:
        result = simulate_worldcup_2026(conn, model, le, LABEL_CLASSES)
    return jsonify(result)


@app.route("/stats")
def get_stats():
    with engine.connect() as conn:
        goals = conn.execute(text("""
            SELECT season,
                   ROUND(AVG(home_score + away_score)::numeric, 2) as avg_goals,
                   COUNT(*) as total_matches,
                   SUM(home_score + away_score) as total_goals
            FROM matches WHERE status = 'FINISHED'
            GROUP BY season ORDER BY season DESC
        """))
        goals_data = [dict(row._mapping) for row in goals]

        top_teams = conn.execute(text("""
            SELECT team, SUM(wins) as wins FROM team_stats
            GROUP BY team ORDER BY wins DESC LIMIT 5
        """))
        top_data = [dict(row._mapping) for row in top_teams]

        totals = conn.execute(text("""
            SELECT COUNT(*) as total_matches,
                   SUM(home_score + away_score) as total_goals
            FROM matches WHERE status = 'FINISHED'
        """))
        totals_data = dict(totals.fetchone()._mapping)

    return jsonify({"by_season": goals_data, "top_teams": top_data, "totals": totals_data})

if __name__ == "__main__":
    app.run(debug=True)
