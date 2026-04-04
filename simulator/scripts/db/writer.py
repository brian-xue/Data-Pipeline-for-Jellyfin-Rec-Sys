from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session


class EventWriter:
	def __init__(self, session: Session):
		self.session = session

	def _ensure_user_exists(self, user_id: int) -> None:
		self.session.execute(
			text(
				"""
				INSERT INTO users (user_id, status, first_seen_at, last_seen_at)
				VALUES (:user_id, 'active', NOW(), NOW())
				ON CONFLICT (user_id) DO NOTHING
				"""
			),
			{"user_id": user_id},
		)

	def _ensure_session_exists(self, user_id: int, session_id: str, event_time: datetime) -> None:
		self.session.execute(
			text(
				"""
				INSERT INTO online_sessions (
					session_id,
					user_id,
					session_start_time,
					status,
					event_count,
					total_watch_duration_seconds
				)
				VALUES (:session_id, :user_id, :session_start_time, 'active', 0, 0.0)
				ON CONFLICT (session_id) DO NOTHING
				"""
			),
			{
				"session_id": session_id,
				"user_id": user_id,
				"session_start_time": event_time,
			},
		)

		bound_user_id = self.session.execute(
			text("SELECT user_id FROM online_sessions WHERE session_id = :session_id"),
			{"session_id": session_id},
		).scalar_one()

		if int(bound_user_id) != int(user_id):
			raise ValueError(
				f"session_id '{session_id}' is already bound to user_id={bound_user_id}, "
				f"cannot use with user_id={user_id}"
			)

	def insert_auth_event(
		self,
		user_id: int,
		session_id: str,
		event_type: str,
		event_time: datetime,
		metadata_json: dict[str, Any] | None = None,
	) -> None:
		self._ensure_user_exists(user_id)

		if event_type == "login":
			self._ensure_session_exists(user_id=user_id, session_id=session_id, event_time=event_time)
		elif event_type == "logout":
			self.session.execute(
				text(
					"""
					UPDATE online_sessions
					SET status = 'closed',
						session_end_time = :event_time,
						updated_at = NOW()
					WHERE session_id = :session_id
					"""
				),
				{"session_id": session_id, "event_time": event_time},
			)

		self.session.execute(
			text(
				"""
				INSERT INTO auth_events (user_id, session_id, event_type, event_time, metadata_json)
				VALUES (:user_id, :session_id, :event_type, :event_time, CAST(:metadata_json AS jsonb))
				"""
			),
			{
				"user_id": user_id,
				"session_id": session_id,
				"event_type": event_type,
				"event_time": event_time,
				"metadata_json": "{}" if metadata_json is None else __import__("json").dumps(metadata_json),
			},
		)

	def insert_user_event(
		self,
		user_id: int,
		movie_id: int,
		session_id: str,
		event_time: datetime,
		watch_duration_seconds: float,
	) -> None:
		self._ensure_user_exists(user_id)
		self._ensure_session_exists(user_id=user_id, session_id=session_id, event_time=event_time)

		self.session.execute(
			text(
				"""
				INSERT INTO user_events (
					user_id,
					movie_id,
					session_id,
					event_type,
					event_time,
					watch_duration_seconds
				)
				VALUES (:user_id, :movie_id, :session_id, 'finish', :event_time, :watch_duration_seconds)
				"""
			),
			{
				"user_id": user_id,
				"movie_id": movie_id,
				"session_id": session_id,
				"event_time": event_time,
				"watch_duration_seconds": watch_duration_seconds,
			},
		)

		self.session.execute(
			text(
				"""
				UPDATE online_sessions
				SET last_event_time = :event_time,
					event_count = event_count + 1,
					total_watch_duration_seconds = total_watch_duration_seconds + :watch_duration_seconds,
					updated_at = NOW()
				WHERE session_id = :session_id
				"""
			),
			{
				"session_id": session_id,
				"event_time": event_time,
				"watch_duration_seconds": watch_duration_seconds,
			},
		)

		self.session.execute(
			text(
				"""
				UPDATE users
				SET last_seen_at = :event_time,
					updated_at = NOW()
				WHERE user_id = :user_id
				"""
			),
			{"user_id": user_id, "event_time": event_time},
		)

	def commit(self) -> None:
		self.session.commit()

