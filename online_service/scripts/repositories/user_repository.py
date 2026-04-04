from __future__ import annotations

from datetime import datetime

from psycopg2.extras import RealDictCursor


class UserRepository:
	def __init__(self, conn):
		self.conn = conn

	def ensure_user_exists(self, user_id: int, source_type: str = "simulator") -> None:
		with self.conn.cursor() as cur:
			cur.execute(
				"""
				INSERT INTO users (user_id, status, first_seen_at, last_seen_at, metadata_json)
				VALUES (%s, 'active', NOW(), NOW(), jsonb_build_object('source_type', %s))
				ON CONFLICT (user_id) DO NOTHING
				""",
				(user_id, source_type),
			)

	def increment_login_count(self, user_id: int) -> None:
		with self.conn.cursor() as cur:
			cur.execute(
				"""
				UPDATE users
				SET login_count = login_count + 1,
					updated_at = NOW()
				WHERE user_id = %s
				""",
				(user_id,),
			)

	def update_last_login(self, user_id: int, last_login_at: datetime) -> None:
		with self.conn.cursor() as cur:
			cur.execute(
				"""
				UPDATE users
				SET last_login_at = %s,
					updated_at = NOW()
				WHERE user_id = %s
				""",
				(last_login_at, user_id),
			)

	def update_last_seen(self, user_id: int, last_seen_at: datetime) -> None:
		with self.conn.cursor() as cur:
			cur.execute(
				"""
				UPDATE users
				SET last_seen_at = %s,
					updated_at = NOW()
				WHERE user_id = %s
				""",
				(last_seen_at, user_id),
			)

	def update_user_embedding(
		self,
		user_id: int,
		embedding_uri: str,
		embedding_version: str,
		embedding_updated_at: datetime,
	) -> None:
		with self.conn.cursor() as cur:
			cur.execute(
				"""
				UPDATE users
				SET embedding_uri = %s,
					embedding_version = %s,
					embedding_updated_at = %s,
					updated_at = NOW()
				WHERE user_id = %s
				""",
				(embedding_uri, embedding_version, embedding_updated_at, user_id),
			)

	def get_user_by_id(self, user_id: int) -> dict | None:
		with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
			cur.execute(
				"""
				SELECT user_id, status, embedding_uri, embedding_version,
					   embedding_updated_at, login_count, last_login_at,
					   first_seen_at, last_seen_at, updated_at, metadata_json
				FROM users
				WHERE user_id = %s
				""",
				(user_id,),
			)
			row = cur.fetchone()
			return None if row is None else dict(row)
