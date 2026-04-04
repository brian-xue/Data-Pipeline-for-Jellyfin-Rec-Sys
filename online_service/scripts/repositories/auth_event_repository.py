from __future__ import annotations

from typing import Any

from psycopg2.extras import RealDictCursor


class AuthEventRepository:
	def __init__(self, conn):
		self.conn = conn

	def fetch_auth_events_after(self, last_auth_event_id: int, limit: int | None = None) -> list[dict[str, Any]]:
		with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
			if limit is None:
				cur.execute(
					"""
					SELECT auth_event_id, user_id, session_id, event_type, event_time, created_at, metadata_json
					FROM auth_events
					WHERE auth_event_id > %s
					ORDER BY auth_event_id ASC
					""",
					(last_auth_event_id,),
				)
			else:
				cur.execute(
					"""
					SELECT auth_event_id, user_id, session_id, event_type, event_time, created_at, metadata_json
					FROM auth_events
					WHERE auth_event_id > %s
					ORDER BY auth_event_id ASC
					LIMIT %s
					""",
					(last_auth_event_id, limit),
				)
			return [dict(row) for row in cur.fetchall()]
