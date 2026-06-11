import os
import sys
import pickle
import traceback

import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
if DATA_DIR not in sys.path:
    sys.path.insert(0, DATA_DIR)

load_dotenv()

FIFA_DISPLAY_NAMES = {}
apply_champion_boost = attach_predicted_score = encode_stage = None
get_team_stats_combined = predict_from_stats = resolve_model_team = None
simulate_worldcup_2026 = None
SIM_IMPORT_ERROR = None

try:
    from worldcup_2026 import FIFA_DISPLAY_NAMES
    from prediction_utils import (
        apply_champion_boost,
        attach_predicted_score,
        encode_stage,
        get_team_stats_combined,
        predict_from_stats,
        resolve_model_team,
    )
except Exception as exc:
    print(f"[worldpredict] Error cargando prediction_utils: {exc}", flush=True)
    traceback.print_exc()

try:
    from tournament_sim import simulate_worldcup_2026
except Exception as exc:
    SIM_IMPORT_ERROR = str(exc)
    print(f"[worldpredict] Error cargando tournament_sim: {exc}", flush=True)
    traceback.print_exc()
    simulate_worldcup_2026 = None

app = Flask(__name__)
CORS(app)

_db_url = os.getenv("DATABASE_URL")
if _db_url and _db_url.startswith("postgres://"):
    _db_url = _db_url.replace("postgres://", "postgresql://", 1)

engine = create_engine(
    _db_url,
    pool_pre_ping=True,
    pool_recycle=300,
) if _db_url else None

model = le = None
LABEL_CLASSES = set()
MODEL_LOAD_ERROR = None

try:
    with open(os.path.join(BASE_DIR, "models/model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(BASE_DIR, "models/label_encoder.pkl"), "rb") as f:
        le = pickle.load(f)
    LABEL_CLASSES = set(le.classes_)
except Exception as exc:
    MODEL_LOAD_ERROR = str(exc)
    print(f"[worldpredict] Error cargando modelos: {exc}", flush=True)
    traceback.print_exc()


def _db_error_response():
    return jsonify({
        "error": "No hay conexión a la base de datos. Configura DATABASE_URL en Render (Environment).",
    }), 503


def _require_db():
    if engine is None:
        return _db_error_response()
    return None


@app.route("/health")
def health():
    db_ok = False
    db_error = None
    if engine is not None:
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            db_ok = True
        except Exception as exc:
            db_error = str(exc)
    return jsonify({
        "status": "ok",
        "message": "WorldPredict API corriendo!",
        "database_connected": db_ok,
        "database_error": db_error,
        "simulation_available": simulate_worldcup_2026 is not None,
        "simulation_error": SIM_IMPORT_ERROR,
        "model_loaded": model is not None and le is not None,
        "model_error": MODEL_LOAD_ERROR,
    })


@app.route("/matches")
def get_matches():
    err = _require_db()
    if err:
        return err
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
    if get_team_stats_combined is None:
        return jsonify({"error": "Motor de predicción no disponible en el servidor"}), 500

    err = _require_db()
    if err:
        return err

    with engine.connect() as conn:
        hs = get_team_stats_combined(conn, home, season)
        as_ = get_team_stats_combined(conn, away, season)

        home_model = resolve_model_team(home, LABEL_CLASSES)
        away_model = resolve_model_team(away, LABEL_CLASSES)

        if not home_model or not away_model or model is None or le is None:
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
        stage_enc = encode_stage(stage)
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
    err = _require_db()
    if err:
        return err
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
    err = _require_db()
    if err:
        return err
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
        msg = "Módulo de simulación no disponible"
        if SIM_IMPORT_ERROR:
            msg = f"{msg}: {SIM_IMPORT_ERROR}"
        return jsonify({"error": msg}), 500
    if get_team_stats_combined is None:
        return jsonify({"error": "Motor de predicción no disponible"}), 500
    err = _require_db()
    if err:
        return err
    try:
        with engine.connect() as conn:
            result = simulate_worldcup_2026(conn, model, le, LABEL_CLASSES)
        return jsonify(result)
    except Exception as exc:
        print(f"[worldpredict] simulate error: {exc}", flush=True)
        traceback.print_exc()
        return jsonify({"error": f"Error en simulación: {exc}"}), 500


@app.route("/stats")
def get_stats():
    err = _require_db()
    if err:
        return err
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

        top_teams = []
        try:
            top_result = conn.execute(text("""
                SELECT team, SUM(wins) as wins FROM team_stats
                GROUP BY team ORDER BY wins DESC LIMIT 5
            """))
            top_teams = [dict(row._mapping) for row in top_result]
        except Exception:
            pass

        totals_row = conn.execute(text("""
            SELECT COUNT(*) as total_matches,
                   COALESCE(SUM(home_score + away_score), 0) as total_goals
            FROM matches WHERE status = 'FINISHED'
        """)).fetchone()
        totals_data = dict(totals_row._mapping) if totals_row else {
            "total_matches": 0,
            "total_goals": 0,
        }

    return jsonify({"by_season": goals_data, "top_teams": top_teams, "totals": totals_data})


if __name__ == "__main__":
    app.run(debug=True)
