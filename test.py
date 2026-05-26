import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

engine = create_engine(os.getenv("DATABASE_URL"))

try:
    with engine.connect() as conn:
        result = conn.execute(text("SELECT 1"))
        print("✅ Conexión exitosa a Supabase!")
except Exception as e:
    print(f"❌ Error: {e}")