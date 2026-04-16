from __future__ import annotations

import random
import uuid
from typing import Any

import numpy as np

from config import SimulatorConfig
from db.writer import EventWriter
from generators.auth_generator import make_auth_event
from generators.event_generator import make_finish_event
from incremental_request import fetch_incremental_candidates
from state import SimulatorState, UserRuntime


class SimulatorService:
	def __init__(self, cfg: SimulatorConfig, state: SimulatorState, writer: EventWriter, incremental_request_cfg: dict[str, Any] | None = None):
		self.cfg = cfg
		self.state = state
		self.writer = writer
		self.incremental_request_cfg = incremental_request_cfg or {}
		self._movie_embeddings: np.ndarray | None = None
		self._movie_ids: np.ndarray | None = None
		self._movie_id_to_index: dict[int, int] = {}

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

	def _ensure_movie_embeddings_loaded(self) -> None:
		if self._movie_embeddings is not None and self._movie_ids is not None:
			return

		embeddings = np.load(self.cfg.movie_embeddings_npy_path).astype(np.float32)
		movie_ids = np.load(self.cfg.movie_ids_path)
		if embeddings.ndim != 2:
			raise ValueError(f"movie embeddings must be 2D, got shape={embeddings.shape}")
		if len(movie_ids) != embeddings.shape[0]:
			raise ValueError(
				f"movie ids size ({len(movie_ids)}) does not match embedding rows ({embeddings.shape[0]})"
			)

		movie_id_to_index: dict[int, int] = {}
		for index, movie_id in enumerate(movie_ids):
			movie_id_int = int(movie_id)
			if movie_id_int in movie_id_to_index:
				continue
			movie_id_to_index[movie_id_int] = index

		self._movie_embeddings = embeddings
		self._movie_ids = movie_ids
		self._movie_id_to_index = movie_id_to_index

	def _get_movie_embedding(self, movie_id: int) -> list[float] | None:
		self._ensure_movie_embeddings_loaded()
		index = self._movie_id_to_index.get(int(movie_id))
		if index is None or self._movie_embeddings is None:
			return None
		return self._movie_embeddings[index].tolist()

	def _cosine_similarity(self, left: list[float], right: list[float]) -> float:
		left_vec = np.asarray(left, dtype=np.float32)
		right_vec = np.asarray(right, dtype=np.float32)
		if left_vec.size == 0 or right_vec.size == 0 or left_vec.shape != right_vec.shape:
			return 0.0
		denominator = float(np.linalg.norm(left_vec) * np.linalg.norm(right_vec))
		if denominator <= 1e-12:
			return 0.0
		return float(np.dot(left_vec, right_vec) / denominator)

	def _random_movie_id(self) -> int:
		self._ensure_movie_embeddings_loaded()
		if self._movie_ids is not None and len(self._movie_ids) > 0:
			return int(random.choice(self._movie_ids.tolist()))
		return random.randint(self.cfg.min_movie_id, self.cfg.max_movie_id)

	def _rank_movie_ids_for_user(self, user_id: int) -> list[int]:
		candidate_payload = self.state.candidate_request_result or {}
		candidate_items = candidate_payload.get("items") if isinstance(candidate_payload, dict) else None
		if not isinstance(candidate_items, list) or not candidate_items:
			return []

		user_embedding = self.state.user_embeddings_by_id.get(user_id)
		scored_items: list[tuple[float, int]] = []
		for item in candidate_items:
			if not isinstance(item, dict):
				continue
			movie_id_raw = item.get("movie_id")
			if movie_id_raw is None:
				continue
			movie_id = int(movie_id_raw)
			base_score = float(item.get("score", 0.0))
			movie_embedding = self._get_movie_embedding(movie_id)
			if user_embedding is not None and movie_embedding is not None:
				base_score = self._cosine_similarity(user_embedding, movie_embedding)
			score = base_score + random.uniform(self.cfg.ranking_noise_min, self.cfg.ranking_noise_max)
			scored_items.append((score, movie_id))

		if not scored_items:
			return []

		scored_items.sort(key=lambda item: item[0], reverse=True)
		ranked_movie_ids: list[int] = []
		seen: set[int] = set()
		for _, movie_id in scored_items:
			if movie_id in seen:
				continue
			seen.add(movie_id)
			ranked_movie_ids.append(movie_id)
		return ranked_movie_ids

	def _select_movies_for_user(self, user_id: int, max_events_to_emit: int, remaining_session_events: int) -> list[tuple[int, float]]:
		event_count_cap = max(self.cfg.event_count_min, self.cfg.event_count_max)
		limit = min(event_count_cap, max_events_to_emit, remaining_session_events)
		if limit <= 0:
			return []

		event_count_floor = min(max(self.cfg.event_count_min, 1), limit)
		event_count = random.randint(event_count_floor, limit)
		short_min = max(0, self.cfg.short_event_count_min)
		short_cap = max(short_min, self.cfg.short_event_count_max)
		short_count = random.randint(short_min, min(short_cap, event_count))
		long_count = event_count - short_count

		ranked_movie_ids = self._rank_movie_ids_for_user(user_id)
		if not ranked_movie_ids:
			ranked_movie_ids = [
				self._random_movie_id()
				for _ in range(max(event_count * self.cfg.candidate_pool_multiplier, self.cfg.candidate_top_pool_min_size))
			]

		selected_movie_ids: list[int] = []
		selected_set: set[int] = set()

		pool_size = max(event_count * self.cfg.candidate_pool_multiplier, self.cfg.candidate_top_pool_min_size)
		top_pool = ranked_movie_ids[:pool_size]
		tail_size = max(event_count * self.cfg.candidate_pool_multiplier, self.cfg.candidate_tail_pool_min_size)
		tail_pool = ranked_movie_ids[-tail_size:]

		for movie_id in top_pool:
			if len(selected_movie_ids) >= long_count:
				break
			if movie_id in selected_set:
				continue
			selected_movie_ids.append(movie_id)
			selected_set.add(movie_id)

		short_movie_ids: list[int] = []
		for movie_id in tail_pool:
			if len(short_movie_ids) >= short_count:
				break
			if movie_id in selected_set:
				continue
			short_movie_ids.append(movie_id)
			selected_set.add(movie_id)

		while len(selected_movie_ids) < long_count:
			movie_id = self._random_movie_id()
			if movie_id in selected_set:
				continue
			selected_movie_ids.append(movie_id)
			selected_set.add(movie_id)

		while len(short_movie_ids) < short_count:
			movie_id = self._random_movie_id()
			if movie_id in selected_set:
				continue
			short_movie_ids.append(movie_id)
			selected_set.add(movie_id)

		for index, movie_id in enumerate(selected_movie_ids):
			if random.random() < self.cfg.random_movie_injection_ratio:
				replacement = self._random_movie_id()
				if replacement not in selected_set:
					selected_set.discard(movie_id)
					selected_set.add(replacement)
					selected_movie_ids[index] = replacement

		for index, movie_id in enumerate(short_movie_ids):
			if random.random() < self.cfg.random_movie_injection_ratio:
				replacement = self._random_movie_id()
				if replacement not in selected_set:
					selected_set.discard(movie_id)
					selected_set.add(replacement)
					short_movie_ids[index] = replacement

		selected_events: list[tuple[int, float]] = []
		for movie_id in selected_movie_ids:
			watch_duration = float(
				random.randint(
					self.cfg.long_watch_duration_min_seconds,
					self.cfg.long_watch_duration_max_seconds,
				)
			)
			selected_events.append((movie_id, watch_duration))

		for movie_id in short_movie_ids:
			watch_duration = float(
				random.randint(
					self.cfg.short_watch_duration_min_seconds,
					self.cfg.short_watch_duration_max_seconds,
				)
			)
			selected_events.append((movie_id, watch_duration))

		random.shuffle(selected_events)
		return selected_events

	def _maybe_request_candidates_once(self) -> None:
		if self.state.candidate_request_done:
			return

		if not self.incremental_request_cfg.get("enabled", False):
			return

		if not self.state.online_users:
			return

		candidate_url = self.incremental_request_cfg.get("uri", "http://online_service:18080/candidates")
		top_k = int(self.incremental_request_cfg.get("top_k", self.cfg.candidate_request_top_k))
		timeout_seconds = float(
			self.incremental_request_cfg.get("timeout_seconds", self.cfg.candidate_request_timeout_seconds)
		)
		user_id = random.choice(list(self.state.online_users))
		self.state.candidate_request_user_id = user_id
		self.state.candidate_request_result = fetch_incremental_candidates(
			user_id=user_id,
			online_service_url=candidate_url,
			top_k=top_k,
			timeout_seconds=timeout_seconds,
		)
		self.state.candidate_request_done = True

	def emit_user_events_for_user(self, user_id: int, max_events_to_emit: int) -> int:
		runtime = self.state.runtime_by_user.get(user_id)
		if runtime is None:
			return 0

		remaining_session_events = self.cfg.max_events_per_session - runtime.events_in_session
		if remaining_session_events <= 0:
			return 0

		selected_events = self._select_movies_for_user(
			user_id=user_id,
			max_events_to_emit=max_events_to_emit,
			remaining_session_events=remaining_session_events,
		)
		if not selected_events:
			return 0

		written = 0
		for movie_id, watch_duration_seconds in selected_events:
			payload = make_finish_event(
				user_id=user_id,
				session_id=runtime.session_id,
				min_movie_id=self.cfg.min_movie_id,
				max_movie_id=self.cfg.max_movie_id,
				min_watch_duration_seconds=self.cfg.min_watch_duration_seconds,
				max_watch_duration_seconds=self.cfg.max_watch_duration_seconds,
				movie_id=movie_id,
				watch_duration_seconds=watch_duration_seconds,
			)
			self.writer.insert_user_event(
				user_id=payload["user_id"],
				movie_id=payload["movie_id"],
				session_id=payload["session_id"],
				event_time=payload["event_time"],
				watch_duration_seconds=payload["watch_duration_seconds"],
			)
			written += 1

		runtime.events_in_session += written
		return written

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

			remaining_budget = budget - emitted
			written = self.emit_user_events_for_user(user_id=user_id, max_events_to_emit=remaining_budget)
			emitted += written

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
		self._maybe_request_candidates_once()
		emitted = self.emit_user_events_for_tick()
		logged_out = self.logout_some_users()

		return {
			"logged_in": logged_in,
			"emitted_events": emitted,
			"logged_out": logged_out,
			"online_users": len(self.state.online_users),
		}

