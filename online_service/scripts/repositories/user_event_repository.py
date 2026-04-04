from __future__ import annotations

from typing import Any

from psycopg2.extras import RealDictCursor


class UserEventRepository:
	def __init__(self, conn):
		self.conn = conn

	def fetch_user_events_after(self, last_user_event_id: int, limit: int | None = None) -> list[dict[str, Any]]:
		with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
			if limit is None:
				cur.execute(
					"""
					SELECT event_id, user_id, movie_id, session_id, event_type,
						   event_time, watch_duration_seconds, created_at
					FROM user_events
					WHERE event_id > %s
					ORDER BY event_id ASC
					""",
					(last_user_event_id,),
				)
			else:
				cur.execute(
					"""
					SELECT event_id, user_id, movie_id, session_id, event_type,
						   event_time, watch_duration_seconds, created_at
					FROM user_events
					WHERE event_id > %s
					ORDER BY event_id ASC
					LIMIT %s
					""",
					(last_user_event_id, limit),
				)
			return [dict(row) for row in cur.fetchall()]

	def fetch_recent_user_events(self, user_id: int, limit: int) -> list[dict[str, Any]]:
		with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
			cur.execute(
				"""
				SELECT event_id, user_id, movie_id, session_id, event_type,
					   event_time, watch_duration_seconds, created_at
				FROM user_events
				WHERE user_id = %s
				ORDER BY event_time DESC, event_id DESC
				LIMIT %s
				""",
				(user_id, limit),
			)
			return [dict(row) for row in cur.fetchall()]

	def fetch_recent_user_events_within_window(
		self,
		user_id: int,
		limit: int,
		window_hours: int,
	) -> list[dict[str, Any]]:
		with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
			cur.execute(
				"""
				SELECT event_id, user_id, movie_id, session_id, event_type,
					   event_time, watch_duration_seconds, created_at
				FROM user_events
				WHERE user_id = %s
				  AND event_time >= NOW() - (%s || ' hours')::interval
				ORDER BY event_time DESC, event_id DESC
				LIMIT %s
				""",
				(user_id, window_hours, limit),
			)
			return [dict(row) for row in cur.fetchall()]

	def fetch_affected_user_ids_after(self, last_user_event_id: int) -> list[int]:
		with self.conn.cursor() as cur:
			cur.execute(
				"""
				SELECT DISTINCT user_id
				FROM user_events
				WHERE event_id > %s
				ORDER BY user_id ASC
				""",
				(last_user_event_id,),
			)
			return [row[0] for row in cur.fetchall()]

	def fetch_top_movies_by_watch_time(self, limit: int = 50) -> list[dict[str, Any]]:
		with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
			cur.execute(
				"""
				SELECT
					movie_id,
					SUM(watch_duration_seconds) AS total_watch_duration_seconds,
					COUNT(*) AS finish_count
				FROM user_events
				GROUP BY movie_id
				ORDER BY total_watch_duration_seconds DESC, finish_count DESC, movie_id ASC
				LIMIT %s
				""",
				(limit,),
			)
			return [dict(row) for row in cur.fetchall()]
