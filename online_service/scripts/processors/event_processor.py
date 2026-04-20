from __future__ import annotations

import threading
from datetime import datetime, timedelta

from scripts.db.connection import get_connection
from scripts.repositories.checkpoint_repository import CheckpointRepository
from scripts.repositories.online_event_stats_repository import OnlineEventStatsRepository
from scripts.repositories.session_repository import SessionRepository
from scripts.repositories.user_event_repository import UserEventRepository
from scripts.repositories.user_repository import UserRepository
from scripts.utils.logger import get_logger


LOGGER = get_logger("online_service.event_processor")
JOB_NAME = "online_service_event"


class EventProcessor:
	def __init__(self, config):
		self.config = config
		self.interval_seconds = config.processor_intervals.event_processor_seconds
		self.window_minutes = max(int(config.monitoring.window_minutes), 1)
		self.short_watch_threshold_seconds = float(config.monitoring.short_watch_threshold_seconds)

	def _resolve_window_bounds(self, event_time: datetime) -> tuple[datetime, datetime]:
		base_time = event_time.replace(second=0, microsecond=0)
		offset_minutes = base_time.minute % self.window_minutes
		window_start = base_time - timedelta(minutes=offset_minutes)
		window_end = window_start + timedelta(minutes=self.window_minutes)
		return window_start, window_end

	def run_once(self) -> int:
		processed = 0
		with get_connection(self.config) as conn:
			checkpoint_repo = CheckpointRepository(conn)
			event_repo = UserEventRepository(conn)
			stats_repo = OnlineEventStatsRepository(conn)
			user_repo = UserRepository(conn)
			session_repo = SessionRepository(conn)

			checkpoint = checkpoint_repo.get_checkpoint(JOB_NAME)
			last_user_event_id = int(checkpoint["last_user_event_id"])
			events = event_repo.fetch_user_events_after(last_user_event_id=last_user_event_id)

			if not events:
				return 0

			events = sorted(
				events,
				key=lambda item: (int(item["user_id"]), int(item["event_id"])),
			)

			latest_event_id = last_user_event_id
			latest_event_time: datetime | None = checkpoint.get("last_user_event_time")
			window_stats: dict[datetime, dict[str, int | datetime]] = {}

			for event in events:
				processed += 1
				if event["event_type"] != "finish":
					continue

				user_id = int(event["user_id"])
				event_time = event["event_time"]
				session_id = event.get("session_id")
				watch_duration = float(event["watch_duration_seconds"])

				user_repo.ensure_user_exists(user_id)
				user_repo.update_last_seen(user_id, event_time)

				if session_id:
					session_repo.update_session_on_finish(
						session_id=session_id,
						event_time=event_time,
						watch_duration_seconds=watch_duration,
					)

				window_start_time, window_end_time = self._resolve_window_bounds(event_time)
				stats_bucket = window_stats.get(window_start_time)
				if stats_bucket is None:
					stats_bucket = {
						"window_end_time": window_end_time,
						"total_count": 0,
						"abnormal_count": 0,
					}
					window_stats[window_start_time] = stats_bucket
				stats_bucket["total_count"] = int(stats_bucket["total_count"]) + 1
				if watch_duration < self.short_watch_threshold_seconds:
					stats_bucket["abnormal_count"] = int(stats_bucket["abnormal_count"]) + 1

				latest_event_id = max(latest_event_id, int(event["event_id"]))
				latest_event_time = event_time

			for window_start_time in sorted(window_stats):
				stats_bucket = window_stats[window_start_time]
				stats_repo.upsert_window_stats(
					window_start_time=window_start_time,
					window_end_time=stats_bucket["window_end_time"],
					total_count=int(stats_bucket["total_count"]),
					abnormal_count=int(stats_bucket["abnormal_count"]),
				)

			checkpoint_repo.update_user_event_checkpoint(
				job_name=JOB_NAME,
				last_user_event_id=latest_event_id,
				last_user_event_time=latest_event_time,
				status="idle",
			)

		return processed

	def run_loop(self, stop_event: threading.Event) -> None:
		LOGGER.info("EventProcessor loop started")
		while not stop_event.is_set():
			try:
				count = self.run_once()
				if count > 0:
					LOGGER.info("EventProcessor processed %s events", count)
			except Exception:
				LOGGER.exception("EventProcessor run_once failed")
			stop_event.wait(self.interval_seconds)
