import os
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

engine = create_engine(os.getenv("DATABASE_URL"))

def load_from_csv():
    df = pd.read_csv("data/raw/WorldCupMatches.csv")
    df = df.dropna(subset=["Home Team Name", "Away Team Name"])

    with engine.connect() as conn:
        for i, row in df.iterrows():
            conn.execute(text("""
                INSERT INTO matches (match_id, home_team, away_team, home_score, away_score, status, match_date, stage, season)
                VALUES (:match_id, :home_team, :away_team, :home_score, :away_score, :status, :match_date, :stage, :season)
                ON CONFLICT (match_id) DO NOTHING
            """), {
                "match_id": int(row["MatchID"]),
                "home_team": row["Home Team Name"],
                "away_team": row["Away Team Name"],
                "home_score": int(row["Home Team Goals"]),
                "away_score": int(row["Away Team Goals"]),
                "status": "FINISHED",
                "match_date": pd.to_datetime(row["Datetime"]),
                "stage": row["Stage"],
                "season": int(row["Year"])
            })
        conn.commit()
    print(f"✅ {len(df)} partidos cargados en la base de datos!")

if __name__ == "__main__":
    load_from_csv()