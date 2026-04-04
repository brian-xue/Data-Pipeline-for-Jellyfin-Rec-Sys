from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from config import PostgresConfig


def create_db_engine(pg: PostgresConfig):
	url = f"postgresql+psycopg2://{pg.user}:{pg.password}@{pg.host}:{pg.port}/{pg.dbname}"
	return create_engine(url, future=True, pool_pre_ping=True)


def create_session_factory(pg: PostgresConfig) -> sessionmaker[Session]:
	engine = create_db_engine(pg)
	return sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)

