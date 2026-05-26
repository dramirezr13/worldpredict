import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()
engine = create_engine(os.getenv("DATABASE_URL"))

def build_team_stats():
    with engine.connect() as conn:
        # Limpiar tabla primero
        conn.execute(text("DELETE FROM team_stats"))
        
        # Obtener todos los partidos finalizados
        matches = conn.execute(text("""
            SELECT home_team, away_team, home_score, away_score, season
            FROM matches WHERE status = 'FINISHED'
        """)).fetchall()

        stats = {}

        for m in matches:
            home, away, hs, as_, season = m.home_team, m.away_team, m.home_score, m.away_score, m.season

            for team in [home, away]:
                key = (team, season)
                if key not in stats:
                    stats[key] = {"wins": 0, "draws": 0, "losses": 0, "goals_for": 0, "goals_against": 0, "matches_played": 0}

            stats[(home, season)]["matches_played"] += 1
            stats[(away, season)]["matches_played"] += 1
            stats[(home, season)]["goals_for"] += hs
            stats[(home, season)]["goals_against"] += as_
            stats[(away, season)]["goals_for"] += as_
            stats[(away, season)]["goals_against"] += hs

            if hs > as_:
                stats[(home, season)]["wins"] += 1
                stats[(away, season)]["losses"] += 1
            elif hs == as_:
                stats[(home, season)]["draws"] += 1
                stats[(away, season)]["draws"] += 1
            else:
                stats[(away, season)]["wins"] += 1
                stats[(home, season)]["losses"] += 1

        # Insertar en BD
        for (team, season), s in stats.items():
            conn.execute(text("""
                INSERT INTO team_stats (team, season, wins, draws, losses, goals_for, goals_against, matches_played)
                VALUES (:team, :season, :wins, :draws, :losses, :goals_for, :goals_against, :matches_played)
            """), {"team": team, "season": season, **s})

        conn.commit()
        print(f"✅ Estadísticas calculadas para {len(stats)} combinaciones equipo/torneo")

if __name__ == "__main__":
    build_team_stats()