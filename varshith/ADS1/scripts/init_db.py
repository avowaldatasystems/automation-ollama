from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import create_engine, text

from app.config import get_settings


def main() -> None:
    settings = get_settings()
    schema = Path("sql/schema.sql").read_text(encoding="utf-8")
    engine = create_engine(settings.mysql_server_url, pool_pre_ping=True)

    statements = [part.strip() for part in schema.split(";") if part.strip()]
    with engine.begin() as conn:
        for statement in statements:
            conn.execute(text(statement))

    print("Database schema created and sample data inserted.")


if __name__ == "__main__":
    main()
