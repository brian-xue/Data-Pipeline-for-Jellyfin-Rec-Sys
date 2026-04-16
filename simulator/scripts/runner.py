from __future__ import annotations

import gc
import random

from api_writer import ApiEventWriter
from config import AppConfig
from db.client import create_session_factory
from db.writer import EventWriter
from generators.user_pool import build_simulator_user_pool

from scheduler import TickScheduler
from service.simulator_service import SimulatorService
from state import SimulatorState


def run_simulation(cfg: AppConfig) -> None:
	random.seed(cfg.random_seed)

	def _run_with_writer(writer: EventWriter | ApiEventWriter) -> None:
		online_users, offline_users, user_embeddings_by_id = build_simulator_user_pool(
			profile_path=cfg.simulator.base_profile_path,
			online_user_sample_size=cfg.simulator.online_user_sample_size,
			fallback_user_pool_size=cfg.simulator.user_pool_size,
			min_user_id=cfg.simulator.min_user_id,
			max_user_id=cfg.simulator.max_user_id,
			random_seed=cfg.random_seed,
		)
		state = SimulatorState(
			online_users=set(online_users),
			offline_users=set(offline_users),
			runtime_by_user={},
			user_embeddings_by_id=user_embeddings_by_id,
		)
		service = SimulatorService(
			cfg=cfg.simulator,
			state=state,
			writer=writer,
			incremental_request_cfg=cfg.incremental_request,
		)
		scheduler = TickScheduler(cfg.simulator.tick_seconds)

		def _tick(tick_index: int) -> None:
			stats = service.run_tick()
			writer.commit()
			print(
				f"[tick={tick_index}] logged_in={stats['logged_in']} "
				f"emitted={stats['emitted_events']} "
				f"logged_out={stats['logged_out']} online={stats['online_users']}"
			)
			if cfg.simulator.memory_cleanup_every_ticks > 0 and tick_index % cfg.simulator.memory_cleanup_every_ticks == 0:
				gc.collect()

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

