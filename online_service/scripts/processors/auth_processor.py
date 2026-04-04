from __future__ import annotations

import threading
import time
from datetime import datetime

from scripts.db.connection import get_connection
from scripts.repositories.auth_event_repository import AuthEventRepository
from scripts.repositories.checkpoint_repository import CheckpointRepository
from scripts.repositories.session_repository import SessionRepository
from scripts.repositories.user_repository import UserRepository
from scripts.utils.logger import get_logger


LOGGER = get_logger("online_service.auth_processor")
JOB_NAME = "online_service_auth"


class AuthProcessor:
	def __init__(self, config):
		self.config = config
		self.interval_seconds = config.processor_intervals.auth_processor_seconds

	def run_once(self) -> int:
		processed = 0
		with get_connection(self.config) as conn:
			checkpoint_repo = CheckpointRepository(conn)
			auth_repo = AuthEventRepository(conn)
			user_repo = UserRepository(conn)
			session_repo = SessionRepository(conn)

			checkpoint = checkpoint_repo.get_checkpoint(JOB_NAME)
			last_auth_event_id = int(checkpoint["last_auth_event_id"])
			events = auth_repo.fetch_auth_events_after(last_auth_event_id=last_auth_event_id)

			if not events:
				return 0

			events = sorted(
				events,
				key=lambda item: (int(item["user_id"]), int(item["auth_event_id"])),
			)

			latest_event_id = last_auth_event_id
			latest_event_time: datetime | None = checkpoint.get("last_auth_event_time")

			for event in events:
				processed += 1
				user_id = int(event["user_id"])
				event_type = event["event_type"]
				event_time = event["event_time"]
				session_id = event.get("session_id") or f"fallback-{user_id}-{int(event['auth_event_id'])}"

				user_repo.ensure_user_exists(user_id)

				if event_type == "login":
					user_repo.increment_login_count(user_id)
					user_repo.update_last_login(user_id, event_time)
					user_repo.update_last_seen(user_id, event_time)
					session_repo.create_or_activate_session(session_id, user_id, event_time)
				elif event_type == "logout":
					session_repo.close_session(session_id, event_time)
					user_repo.update_last_seen(user_id, event_time)

				latest_event_id = max(latest_event_id, int(event["auth_event_id"]))
				latest_event_time = event_time

			checkpoint_repo.update_auth_checkpoint(
				job_name=JOB_NAME,
				last_auth_event_id=latest_event_id,
				last_auth_event_time=latest_event_time,
				status="idle",
			)

		return processed

	def run_loop(self, stop_event: threading.Event) -> None:
		LOGGER.info("AuthProcessor loop started")
		while not stop_event.is_set():
			try:
				count = self.run_once()
				if count > 0:
					LOGGER.info("AuthProcessor processed %s events", count)
			except Exception:
				LOGGER.exception("AuthProcessor run_once failed")
			stop_event.wait(self.interval_seconds)
