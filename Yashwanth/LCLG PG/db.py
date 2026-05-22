from sqlalchemy import create_engine


DATABASE_URL = (
    "postgresql://postgres:postgres123@localhost:5432/LCLG2"
)

engine = create_engine(
    DATABASE_URL
)