from __future__ import annotations

import random

from api_writer import ApiEventWriter
from config import AppConfig
from db.client import create_session_factory
from db.writer import EventWriter
from generators.user_pool import generate_user_pool

from scheduler import TickScheduler
from service.simulator_service import SimulatorService
from state import SimulatorState
from incremental_request import incremental_candidate_request


def run_simulation(cfg: AppConfig) -> None:
	random.seed(cfg.random_seed)

	user_pool = generate_user_pool(
		user_pool_size=cfg.simulator.user_pool_size,
		min_user_id=cfg.simulator.min_user_id,
		max_user_id=cfg.simulator.max_user_id,
		random_seed=cfg.random_seed,
	)

	def _run_with_writer(writer: EventWriter | ApiEventWriter) -> None:
		state = SimulatorState(
			online_users=set(),
			offline_users=set(user_pool),
			runtime_by_user={},
		)
		service = SimulatorService(cfg=cfg.simulator, state=state, writer=writer)
		scheduler = TickScheduler(cfg.simulator.tick_seconds)

		inc_cfg = cfg.incremental_request or {}
		if inc_cfg.get("enabled", False):
			import threading
			threading.Thread(
				target=incremental_candidate_request,
				args=(
					state,
					inc_cfg.get("interval_seconds", 2.0),
					inc_cfg.get("total_duration", 60.0),
					inc_cfg.get("uri", "http://localhost:8000/api/candidates")
				),
				kwargs={
					"min_interval": inc_cfg.get("min_interval"),
					"max_interval": inc_cfg.get("max_interval")
				},
				daemon=True
			).start()

		def _tick(tick_index: int) -> None:
			stats = service.run_tick()
			writer.commit()
			print(
				f"[tick={tick_index}] logged_in={stats['logged_in']} "
				f"emitted={stats['emitted_events']} "
				f"logged_out={stats['logged_out']} online={stats['online_users']}"
			)

		try:
			scheduler.run(total_ticks=cfg.simulator.total_ticks, tick_fn=_tick)
		except KeyboardInterrupt:
			print("Received termination signal, shutting down simulator...")
		finally:
			forced_logout_count = service.logout_all_active_users()
			if forced_logout_count > 0:
				writer.commit()
				print(f"Flushed {forced_logout_count} active users with logout events.")

	if cfg.ingest_api.enabled:
		print(f"Using API ingest writer: {cfg.ingest_api.endpoint}")
		writer = ApiEventWriter(
			endpoint=cfg.ingest_api.endpoint,
			timeout_seconds=cfg.ingest_api.timeout_seconds,
		)
		_run_with_writer(writer)
		return

	session_factory = create_session_factory(cfg.postgres)
	with session_factory() as db_session:
		writer = EventWriter(db_session)
		_run_with_writer(writer)

