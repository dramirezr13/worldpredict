"""Muestra datos no sensibles de DATABASE_URL para comparar local vs Render."""
import os
from urllib.parse import urlparse

from dotenv import load_dotenv

load_dotenv()

url = os.getenv("DATABASE_URL", "")
if not url:
    print("DATABASE_URL no definida en .env")
    raise SystemExit(1)

parsed = urlparse(url)
user = parsed.username or ""
# Supabase pooler: postgres.PROJECT_REF
project_ref = user.split(".")[-1] if "." in user else user

print("Compara estos valores con Render > Environment > DATABASE_URL:")
print(f"  host:         {parsed.hostname}")
print(f"  port:         {parsed.port or 5432}")
print(f"  database:     {(parsed.path or '/').lstrip('/')}")
print(f"  user:         {user.split(':')[0] if user else '(vacío)'}")
print(f"  project_ref:  {project_ref}")
