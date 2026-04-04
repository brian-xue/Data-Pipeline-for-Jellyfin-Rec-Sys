from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import psycopg2
from psycopg2.extensions import connection as PgConnection

from scripts.config.config import OnlineServiceConfig, load_online_service_config


def create_connection(config: OnlineServiceConfig | None = None) -> PgConnection:
	cfg = config or load_online_service_config()
	postgres = cfg.postgres
	return psycopg2.connect(
		host=postgres.host,
		port=postgres.port,
		dbname=postgres.dbname,
		user=postgres.user,
		password=postgres.password,
		connect_timeout=postgres.connect_timeout_seconds,
	)


@contextmanager
def get_connection(config: OnlineServiceConfig | None = None) -> Iterator[PgConnection]:
	conn = create_connection(config)
	try:
		yield conn
		conn.commit()
	except Exception:
		conn.rollback()
		raise
	finally:
		conn.close()


@contextmanager
def get_cursor(config: OnlineServiceConfig | None = None):
	with get_connection(config) as conn:
		with conn.cursor() as cur:
			yield cur
