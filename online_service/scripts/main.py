from __future__ import annotations

import argparse
import signal
import threading
from typing import Callable

from scripts.api.candidate_api import create_app
from scripts.config.config import load_online_service_config
from scripts.processors.auth_processor import AuthProcessor
from scripts.processors.event_processor import EventProcessor
from scripts.processors.exporter import ExporterProcessor
from scripts.processors.popular_movie_updater import PopularMovieUpdater
from scripts.processors.user_embedding_updater import UserEmbeddingUpdater
from scripts.utils.logger import get_logger, setup_logging


LOGGER = get_logger("online_service.main")


def _start_worker(name: str, target: Callable[[threading.Event], None], stop_event: threading.Event) -> threading.Thread:
	thread = threading.Thread(target=target, args=(stop_event,), name=name, daemon=True)
	thread.start()
	return thread


def main() -> None:
	parser = argparse.ArgumentParser(description="Online service main entry")
	parser.add_argument("--config", default=None, help="Path to online service YAML config")
	args = parser.parse_args()

	setup_logging()
	config = load_online_service_config(args.config)

	stop_event = threading.Event()

	def _shutdown(*_):
		LOGGER.info("Received shutdown signal")
		stop_event.set()

	signal.signal(signal.SIGINT, _shutdown)
	signal.signal(signal.SIGTERM, _shutdown)

	auth_processor = AuthProcessor(config)
	event_processor = EventProcessor(config)
	popular_updater = PopularMovieUpdater(config)
	embedding_updater = UserEmbeddingUpdater(config)
	exporter = ExporterProcessor(config)

	threads: list[threading.Thread] = [
		_start_worker("auth_processor", auth_processor.run_loop, stop_event),
		_start_worker("event_processor", event_processor.run_loop, stop_event),
		_start_worker("popular_movie_updater", popular_updater.run_loop, stop_event),
		_start_worker("user_embedding_updater", embedding_updater.run_loop, stop_event),
		_start_worker("exporter", exporter.run_loop, stop_event),
	]

	if config.api.enabled:
		app = create_app(config)

		def _run_api(_: threading.Event) -> None:
			LOGGER.info("Starting candidate API at %s:%s", config.api.host, config.api.port)
			app.run(host=config.api.host, port=config.api.port, debug=False, use_reloader=False)

		threads.append(_start_worker("candidate_api", _run_api, stop_event))

	LOGGER.info("Online service started with %s worker threads", len(threads))

	while not stop_event.is_set():
		for thread in threads:
			if not thread.is_alive():
				LOGGER.error("Thread %s stopped unexpectedly", thread.name)
		stop_event.wait(1)

	for thread in threads:
		thread.join(timeout=3)

	LOGGER.info("Online service stopped")


if __name__ == "__main__":
	main()
