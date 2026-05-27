import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()
engine = create_engine(os.getenv("DATABASE_URL"))
with engine.connect() as conn:
    n = conn.execute(text(
        "SELECT COUNT(*) FROM worldcup_qualified_teams WHERE tournament_id = 'WC-2026'"
    )).scalar()
    print(f"Equipos 2026: {n}")
