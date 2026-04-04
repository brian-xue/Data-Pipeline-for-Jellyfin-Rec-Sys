from __future__ import annotations

import random
import uuid
from typing import Iterable

from config import SimulatorConfig
from db.writer import EventWriter
from generators.auth_generator import make_auth_event
from generators.event_generator import make_finish_event
from state import SimulatorState, UserRuntime


class SimulatorService:
	def __init__(self, cfg: SimulatorConfig, state: SimulatorState, writer: EventWriter):
		self.cfg = cfg
		self.state = state
		self.writer = writer

	def _new_session_id(self, user_id: int) -> str:
		return f"sim-{user_id}-{uuid.uuid4().hex[:12]}"

	def login_user(self, user_id: int) -> None:
		if user_id in self.state.online_users:
			return
		session_id = self._new_session_id(user_id)
		payload = make_auth_event(user_id=user_id, session_id=session_id, event_type="login")
		self.writer.insert_auth_event(**payload)

		self.state.online_users.add(user_id)
		self.state.offline_users.discard(user_id)
		self.state.runtime_by_user[user_id] = UserRuntime(session_id=session_id, events_in_session=0)

	def logout_user(self, user_id: int) -> None:
		runtime = self.state.runtime_by_user.get(user_id)
		if runtime is None:
			return
		payload = make_auth_event(user_id=user_id, session_id=runtime.session_id, event_type="logout")
		self.writer.insert_auth_event(**payload)

		self.state.online_users.discard(user_id)
		self.state.offline_users.add(user_id)
		self.state.runtime_by_user.pop(user_id, None)

	def logout_all_active_users(self) -> int:
		active_user_ids = list(self.state.online_users)
		if not active_user_ids:
			return 0

		for user_id in active_user_ids:
			self.logout_user(user_id)
		return len(active_user_ids)

	def emit_one_user_event(self, user_id: int) -> bool:
		runtime = self.state.runtime_by_user.get(user_id)
		if runtime is None:
			return False

		payload = make_finish_event(
			user_id=user_id,
			session_id=runtime.session_id,
			min_movie_id=self.cfg.min_movie_id,
			max_movie_id=self.cfg.max_movie_id,
			min_watch_duration_seconds=self.cfg.min_watch_duration_seconds,
			max_watch_duration_seconds=self.cfg.max_watch_duration_seconds,
		)
		self.writer.insert_user_event(
			user_id=payload["user_id"],
			movie_id=payload["movie_id"],
			session_id=payload["session_id"],
			event_time=payload["event_time"],
			watch_duration_seconds=payload["watch_duration_seconds"],
		)

		runtime.events_in_session += 1
		return True

	def ensure_target_online_users(self) -> int:
		deficit = self.cfg.target_online_users - len(self.state.online_users)
		if deficit <= 0:
			return 0

		capacity_left = self.cfg.max_online_users - len(self.state.online_users)
		to_login = max(0, min(deficit, capacity_left, self.cfg.login_rate_per_tick, len(self.state.offline_users)))
		if to_login == 0:
			return 0

		selected = random.sample(list(self.state.offline_users), to_login)
		for user_id in selected:
			self.login_user(user_id)
		return to_login

	def emit_user_events_for_tick(self) -> int:
		if not self.state.online_users:
			return 0

		emitted = 0
		budget = self.cfg.global_event_rate_per_tick
		online_users = list(self.state.online_users)
		random.shuffle(online_users)

		for user_id in online_users:
			if emitted >= budget:
				break

			runtime = self.state.runtime_by_user.get(user_id)
			if runtime is None:
				continue

			if runtime.events_in_session >= self.cfg.max_events_per_session:
				continue

			if random.random() > self.cfg.per_user_event_prob:
				continue

			if self.emit_one_user_event(user_id):
				emitted += 1

		return emitted

	def logout_some_users(self) -> int:
		if not self.state.online_users:
			return 0

		candidates: list[int] = []
		for user_id in self.state.online_users:
			runtime = self.state.runtime_by_user.get(user_id)
			if runtime is None:
				continue
			if runtime.events_in_session >= self.cfg.max_events_per_session:
				candidates.append(user_id)
			elif runtime.events_in_session >= self.cfg.min_events_per_session and random.random() < 0.1:
				candidates.append(user_id)

		if not candidates:
			return 0

		count = min(self.cfg.logout_rate_per_tick, len(candidates))
		selected = random.sample(candidates, count)
		for user_id in selected:
			self.logout_user(user_id)
		return count

	def run_tick(self) -> dict[str, int]:
		logged_in = self.ensure_target_online_users()
		emitted = self.emit_user_events_for_tick()
		logged_out = self.logout_some_users()

		return {
			"logged_in": logged_in,
			"emitted_events": emitted,
			"logged_out": logged_out,
			"online_users": len(self.state.online_users),
		}

