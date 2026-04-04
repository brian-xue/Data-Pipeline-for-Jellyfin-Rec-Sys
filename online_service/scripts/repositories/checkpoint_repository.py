from __future__ import annotations

from datetime import datetime
from typing import Any

from psycopg2.extras import RealDictCursor


class CheckpointRepository:
	def __init__(self, conn):
		self.conn = conn

	def get_checkpoint(self, job_name: str) -> dict[str, Any]:
		with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
			cur.execute(
				"""
				SELECT job_name, last_auth_event_id, last_user_event_id,
					   last_auth_event_time, last_user_event_time, status, updated_at, created_at
				FROM service_checkpoints
				WHERE job_name = %s
				""",
				(job_name,),
			)
			row = cur.fetchone()
			if row is None:
				self.upsert_checkpoint(job_name=job_name)
				return self.get_checkpoint(job_name)
			return dict(row)

	def upsert_checkpoint(
		self,
		job_name: str,
		last_auth_event_id: int = 0,
		last_user_event_id: int = 0,
		last_auth_event_time: datetime | None = None,
		last_user_event_time: datetime | None = None,
		status: str = "idle",
	) -> None:
		with self.conn.cursor() as cur:
			cur.execute(
				"""
				INSERT INTO service_checkpoints (
					job_name,
					last_auth_event_id,
					last_user_event_id,
					last_auth_event_time,
					last_user_event_time,
					status,
					updated_at
				)
				VALUES (%s, %s, %s, %s, %s, %s, NOW())
				ON CONFLICT (job_name)
				DO UPDATE SET
					last_auth_event_id = EXCLUDED.last_auth_event_id,
					last_user_event_id = EXCLUDED.last_user_event_id,
					last_auth_event_time = EXCLUDED.last_auth_event_time,
					last_user_event_time = EXCLUDED.last_user_event_time,
					status = EXCLUDED.status,
					updated_at = NOW()
				""",
				(
					job_name,
					last_auth_event_id,
					last_user_event_id,
					last_auth_event_time,
					last_user_event_time,
					status,
				),
			)

	def update_auth_checkpoint(
		self,
		job_name: str,
		last_auth_event_id: int,
		last_auth_event_time: datetime | None,
		status: str = "idle",
	) -> None:
		with self.conn.cursor() as cur:
			cur.execute(
				"""
				UPDATE service_checkpoints
				SET last_auth_event_id = %s,
					last_auth_event_time = %s,
					status = %s,
					updated_at = NOW()
				WHERE job_name = %s
				""",
				(last_auth_event_id, last_auth_event_time, status, job_name),
			)

	def update_user_event_checkpoint(
		self,
		job_name: str,
		last_user_event_id: int,
		last_user_event_time: datetime | None,
		status: str = "idle",
	) -> None:
		with self.conn.cursor() as cur:
			cur.execute(
				"""
				UPDATE service_checkpoints
				SET last_user_event_id = %s,
					last_user_event_time = %s,
					status = %s,
					updated_at = NOW()
				WHERE job_name = %s
				""",
				(last_user_event_id, last_user_event_time, status, job_name),
			)
