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
except ImportError:
    FIFA_DISPLAY_NAMES = {}

load_dotenv()

app = Flask(__name__)
CORS(app)

engine = create_engine(os.getenv("DATABASE_URL"))

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

with open(os.path.join(BASE_DIR, "models/model.pkl"), "rb") as f:
    model = pickle.load(f)
with open(os.path.join(BASE_DIR, "models/label_encoder.pkl"), "rb") as f:
    le = pickle.load(f)

STAGE_MAP = {
    "Group Stage": 1, "Round of 16": 2, "Quarter-finals": 3,
    "Semi-finals": 4, "Third place": 5, "Final": 6
}

def get_team_stats(conn, team, season=2022):
    result = conn.execute(text("""
        SELECT wins, draws, losses, goals_for, goals_against, matches_played
        FROM team_stats WHERE team = :team
        ORDER BY ABS(season - :season) ASC LIMIT 1
    """), {"team": team, "season": season}).fetchone()
    if result:
        s = dict(result._mapping)
        s["win_rate"] = s["wins"] / (s["matches_played"] + 1)
        s["goal_diff"] = s["goals_for"] - s["goals_against"]
        return s
    return {"wins": 0, "draws": 0, "losses": 0, "goals_for": 0,
            "goals_against": 0, "matches_played": 0, "win_rate": 0, "goal_diff": 0}

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
    data = request.json
    home = data.get("home_team")
    away = data.get("away_team")
    stage = data.get("stage", "Group Stage")
    season = data.get("season", 2026)

    try:
        home_enc = le.transform([home])[0]
        away_enc = le.transform([away])[0]
    except ValueError:
        return jsonify({"error": "Equipo no reconocido en el modelo"}), 400

    with engine.connect() as conn:
        hs = get_team_stats(conn, home, season)
        as_ = get_team_stats(conn, away, season)

    stage_enc = STAGE_MAP.get(stage, 1)
    features = np.array([[
        home_enc, away_enc, stage_enc, season,
        hs["win_rate"], as_["win_rate"],
        hs["goal_diff"], as_["goal_diff"],
        hs["wins"], as_["wins"],
        hs["draws"], as_["draws"]
    ]])

    probs = model.predict_proba(features)[0]
    return jsonify({
        "home_team": home,
        "away_team": away,
        "home_win": round(float(probs[0]) * 100, 1),
        "draw": round(float(probs[1]) * 100, 1),
        "away_win": round(float(probs[2]) * 100, 1),
        "predicted_result": ["Local", "Empate", "Visitante"][np.argmax(probs)]
    })

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