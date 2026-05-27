"""
Carga datos de worldcup-master en PostgreSQL/Supabase y registra los 48 equipos del Mundial 2026.
Uso: python data/load_worldcup.py
"""
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from worldcup_2026 import FIFA_DISPLAY_NAMES, WORLD_CUP_2026, WORLD_CUP_2026_TEAMS

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
WORLDCUP_DATA = BASE_DIR / "worldcup-master" / "worldcup-master" / "data-csv"
SCHEMA_FILE = Path(__file__).resolve().parent / "schema_worldcup.sql"

MENS_TOURNAMENT_PREFIX = "WC-"


def get_engine():
    url = os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL no está definida en .env")
    return create_engine(url)


def run_schema(conn):
  with open(SCHEMA_FILE, encoding="utf-8") as f:
    for stmt in f.read().split(";"):
      sql = stmt.strip()
      if sql:
        conn.execute(text(sql))


def load_tournaments(conn):
    path = WORLDCUP_DATA / "tournaments.csv"
    if not path.exists():
        print(f"AVISO: No se encontro {path}")
        return 0

    df = pd.read_csv(path)
    df = df[df["tournament_id"].str.startswith(MENS_TOURNAMENT_PREFIX, na=False)]
    count = 0
    for _, row in df.iterrows():
        conn.execute(text("""
            INSERT INTO worldcup_tournaments
                (tournament_id, tournament_name, year, start_date, end_date,
                 host_country, winner, count_teams)
            VALUES
                (:tournament_id, :tournament_name, :year, :start_date, :end_date,
                 :host_country, :winner, :count_teams)
            ON CONFLICT (tournament_id) DO UPDATE SET
                tournament_name = EXCLUDED.tournament_name,
                year = EXCLUDED.year,
                start_date = EXCLUDED.start_date,
                end_date = EXCLUDED.end_date,
                host_country = EXCLUDED.host_country,
                winner = EXCLUDED.winner,
                count_teams = EXCLUDED.count_teams
        """), {
            "tournament_id": row["tournament_id"],
            "tournament_name": row["tournament_name"],
            "year": int(row["year"]),
            "start_date": row["start_date"],
            "end_date": row["end_date"],
            "host_country": row["host_country"],
            "winner": row["winner"] if pd.notna(row["winner"]) else None,
            "count_teams": int(row["count_teams"]),
        })
        count += 1

    # Torneo 2026 (no está en el dataset original)
    t = WORLD_CUP_2026
    conn.execute(text("""
        INSERT INTO worldcup_tournaments
            (tournament_id, tournament_name, year, start_date, end_date,
             host_country, winner, count_teams)
        VALUES
            (:tournament_id, :tournament_name, :year, :start_date, :end_date,
             :host_country, NULL, :count_teams)
        ON CONFLICT (tournament_id) DO UPDATE SET
            tournament_name = EXCLUDED.tournament_name,
            year = EXCLUDED.year,
            start_date = EXCLUDED.start_date,
            end_date = EXCLUDED.end_date,
            host_country = EXCLUDED.host_country,
            count_teams = EXCLUDED.count_teams
    """), {
        "tournament_id": t["tournament_id"],
        "tournament_name": t["tournament_name"],
        "year": t["year"],
        "start_date": t["start_date"],
        "end_date": t["end_date"],
        "host_country": t["host_countries"],
        "count_teams": t["count_teams"],
    })
    return count + 1


def load_teams(conn):
    path = WORLDCUP_DATA / "teams.csv"
    df = pd.read_csv(path)
    df = df[df["mens_team"] == 1]
    count = 0
    for _, row in df.iterrows():
        conn.execute(text("""
            INSERT INTO worldcup_teams
                (team_id, team_name, team_code, confederation_code, region_name)
            VALUES
                (:team_id, :team_name, :team_code, :confederation_code, :region_name)
            ON CONFLICT (team_id) DO UPDATE SET
                team_name = EXCLUDED.team_name,
                team_code = EXCLUDED.team_code,
                confederation_code = EXCLUDED.confederation_code,
                region_name = EXCLUDED.region_name
        """), {
            "team_id": row["team_id"],
            "team_name": row["team_name"],
            "team_code": row["team_code"],
            "confederation_code": row["confederation_code"],
            "region_name": row["region_name"],
        })
        count += 1

    # Equipos nuevos del 2026 que no están en teams.csv
    extra = [
        ("T-CPV", "Cape Verde", "CPV", "CAF", "Africa"),
        ("T-CUW", "Curaçao", "CUW", "CONCACAF", "Caribbean"),
        ("T-JOR", "Jordan", "JOR", "AFC", "Middle East"),
        ("T-UZB", "Uzbekistan", "UZB", "AFC", "Central Asia"),
        ("T-IRQ", "Iraq", "IRQ", "AFC", "Middle East"),
        ("T-PAN", "Panama", "PAN", "CONCACAF", "Central America"),
        ("T-COD", "DR Congo", "COD", "CAF", "Africa"),
    ]
    for team_id, name, code, conf, region in extra:
        conn.execute(text("""
            INSERT INTO worldcup_teams
                (team_id, team_name, team_code, confederation_code, region_name)
            VALUES (:team_id, :team_name, :team_code, :confederation_code, :region_name)
            ON CONFLICT (team_id) DO NOTHING
        """), {
            "team_id": team_id,
            "team_name": name,
            "team_code": code,
            "confederation_code": conf,
            "region_name": region,
        })
        count += 1
    return count


def load_qualified_historical(conn):
    path = WORLDCUP_DATA / "qualified_teams.csv"
    df = pd.read_csv(path)
    df = df[df["tournament_id"].str.startswith(MENS_TOURNAMENT_PREFIX, na=False)]
    count = 0
    for _, row in df.iterrows():
        conn.execute(text("""
            INSERT INTO worldcup_qualified_teams
                (tournament_id, team_id, team_name, team_code, performance)
            VALUES
                (:tournament_id, :team_id, :team_name, :team_code, :performance)
            ON CONFLICT (tournament_id, team_name) DO UPDATE SET
                team_id = EXCLUDED.team_id,
                team_code = EXCLUDED.team_code,
                performance = EXCLUDED.performance
        """), {
            "tournament_id": row["tournament_id"],
            "team_id": row["team_id"],
            "team_name": row["team_name"],
            "team_code": row["team_code"],
            "performance": row["performance"],
        })
        count += 1
    return count


def load_qualified_2026(conn):
    conn.execute(text("""
        DELETE FROM worldcup_qualified_teams WHERE tournament_id = :tid
    """), {"tid": WORLD_CUP_2026["tournament_id"]})

    teams_df = pd.read_csv(WORLDCUP_DATA / "teams.csv")
    name_to_id = dict(zip(teams_df["team_name"], teams_df["team_id"]))
    extra_ids = {
        "Cape Verde": "T-CPV", "Curaçao": "T-CUW", "Jordan": "T-JOR",
        "Uzbekistan": "T-UZB", "Iraq": "T-IRQ", "Panama": "T-PAN", "DR Congo": "T-COD",
    }
    name_to_id.update(extra_ids)

    for team_name, code, group, conf, is_host in WORLD_CUP_2026_TEAMS:
        conn.execute(text("""
            INSERT INTO worldcup_qualified_teams
                (tournament_id, team_id, team_name, team_code,
                 group_name, confederation, is_host, performance)
            VALUES
                (:tournament_id, :team_id, :team_name, :team_code,
                 :group_name, :confederation, :is_host, 'qualified')
        """), {
            "tournament_id": WORLD_CUP_2026["tournament_id"],
            "team_id": name_to_id.get(team_name),
            "team_name": team_name,
            "team_code": code,
            "group_name": group,
            "confederation": conf,
            "is_host": is_host,
        })
    return len(WORLD_CUP_2026_TEAMS)


def load_matches(conn):
    path = WORLDCUP_DATA / "matches.csv"
    df = pd.read_csv(path)
    df = df[df["tournament_id"].str.startswith(MENS_TOURNAMENT_PREFIX, na=False)]
    count = 0
    for _, row in df.iterrows():
        conn.execute(text("""
            INSERT INTO worldcup_matches
                (match_id, tournament_id, tournament_year, match_date, stage_name,
                 group_name, home_team_name, away_team_name,
                 home_team_score, away_team_score, knockout_stage)
            VALUES
                (:match_id, :tournament_id, :tournament_year, :match_date, :stage_name,
                 :group_name, :home_team_name, :away_team_name,
                 :home_team_score, :away_team_score, :knockout_stage)
            ON CONFLICT (match_id) DO UPDATE SET
                home_team_score = EXCLUDED.home_team_score,
                away_team_score = EXCLUDED.away_team_score
        """), {
            "match_id": row["match_id"],
            "tournament_id": row["tournament_id"],
            "tournament_year": int(str(row["tournament_id"]).split("-")[1]),
            "match_date": row["match_date"],
            "stage_name": row["stage_name"],
            "group_name": row["group_name"] if pd.notna(row.get("group_name")) else None,
            "home_team_name": row["home_team_name"],
            "away_team_name": row["away_team_name"],
            "home_team_score": int(row["home_team_score"]),
            "away_team_score": int(row["away_team_score"]),
            "knockout_stage": bool(row["knockout_stage"]),
        })
        count += 1
    return count


def main():
    if not WORLDCUP_DATA.exists():
        raise FileNotFoundError(f"Carpeta de datos no encontrada: {WORLDCUP_DATA}")

    engine = get_engine()
    with engine.connect() as conn:
        print("[1/5] Creando tablas worldcup_*...")
        run_schema(conn)

        print("[2/5] Cargando torneos...")
        n_t = load_tournaments(conn)
        print(f"      {n_t} torneos")

        print("[3/5] Cargando equipos...")
        n_teams = load_teams(conn)
        print(f"      {n_teams} equipos")

        print("[4/5] Cargando clasificados historicos...")
        n_q = load_qualified_historical(conn)
        print(f"      {n_q} registros")

        print("[5/5] Cargando 48 equipos del Mundial 2026...")
        n_2026 = load_qualified_2026(conn)
        print(f"      {n_2026} equipos")

        print("      Cargando partidos historicos (worldcup)...")
        n_m = load_matches(conn)
        print(f"      {n_m} partidos")

        conn.commit()

    print("\nCarga completada.")
    print(f"   Mundial 2026: {len(WORLD_CUP_2026_TEAMS)} equipos en {len({t[2] for t in WORLD_CUP_2026_TEAMS})} grupos")
    if FIFA_DISPLAY_NAMES:
        print("   (Nombres FIFA en la API: Korea Republic, Czechia, etc.)")


if __name__ == "__main__":
    main()
