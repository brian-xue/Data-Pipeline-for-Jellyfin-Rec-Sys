from __future__ import annotations

import threading

from scripts.db.connection import get_connection
from scripts.repositories.popular_movie_repository import PopularMovieRepository
from scripts.repositories.user_event_repository import UserEventRepository
from scripts.utils.logger import get_logger


LOGGER = get_logger("online_service.popular_movie_updater")


class PopularMovieUpdater:
	def __init__(self, config):
		self.config = config
		self.interval_seconds = config.processor_intervals.popular_movie_updater_seconds

	def compute_top_movies(self, event_repo: UserEventRepository, limit: int = 50) -> list[dict]:
		rows = event_repo.fetch_top_movies_by_watch_time(limit=limit)
		movies: list[dict] = []
		for index, row in enumerate(rows, start=1):
			movies.append(
				{
					"rank_position": index,
					"movie_id": int(row["movie_id"]),
					"score": float(row["total_watch_duration_seconds"]),
				}
			)
		return movies

	def refresh_popular_movies(self, popular_repo: PopularMovieRepository, movies: list[dict]) -> None:
		popular_repo.replace_popular_movies(movies)

	def run_once(self) -> int:
		with get_connection(self.config) as conn:
			event_repo = UserEventRepository(conn)
			popular_repo = PopularMovieRepository(conn)
			movies = self.compute_top_movies(event_repo=event_repo, limit=50)
			self.refresh_popular_movies(popular_repo=popular_repo, movies=movies)
			return len(movies)

	def run_loop(self, stop_event: threading.Event) -> None:
		LOGGER.info("PopularMovieUpdater loop started")
		while not stop_event.is_set():
			try:
				count = self.run_once()
				LOGGER.info("PopularMovieUpdater refreshed %s rows", count)
			except Exception:
				LOGGER.exception("PopularMovieUpdater run_once failed")
			stop_event.wait(self.interval_seconds)
