from __future__ import annotations

from datetime import datetime


class SessionRepository:
	def __init__(self, conn):
		self.conn = conn

	def create_or_activate_session(
		self,
		session_id: str,
		user_id: int,
		session_start_time: datetime,
	) -> None:
		with self.conn.cursor() as cur:
			cur.execute(
				"""
				INSERT INTO online_sessions (
					session_id,
					user_id,
					session_start_time,
					status,
					updated_at
				)
				VALUES (%s, %s, %s, 'active', NOW())
				ON CONFLICT (session_id)
				DO UPDATE SET
					user_id = EXCLUDED.user_id,
					status = 'active',
					session_end_time = NULL,
					updated_at = NOW()
				""",
				(session_id, user_id, session_start_time),
			)

	def update_session_on_finish(
		self,
		session_id: str,
		event_time: datetime,
		watch_duration_seconds: float,
	) -> None:
		with self.conn.cursor() as cur:
			cur.execute(
				"""
				UPDATE online_sessions
				SET last_event_time = %s,
					event_count = event_count + 1,
					total_watch_duration_seconds = total_watch_duration_seconds + %s,
					updated_at = NOW()
				WHERE session_id = %s
				""",
				(event_time, watch_duration_seconds, session_id),
			)

	def close_session(self, session_id: str, session_end_time: datetime) -> None:
		with self.conn.cursor() as cur:
			cur.execute(
				"""
				UPDATE online_sessions
				SET status = 'closed',
					session_end_time = %s,
					updated_at = NOW()
				WHERE session_id = %s
				""",
				(session_end_time, session_id),
			)
