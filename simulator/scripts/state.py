from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class UserRuntime:
	session_id: str
	events_in_session: int = 0


@dataclass
class SimulatorState:
	online_users: set[int] = field(default_factory=set)
	offline_users: set[int] = field(default_factory=set)
	runtime_by_user: dict[int, UserRuntime] = field(default_factory=dict)
	user_embeddings_by_id: dict[int, list[float]] = field(default_factory=dict)
	candidate_request_done: bool = False
	candidate_request_user_id: int | None = None
	candidate_request_result: dict[str, Any] | None = None

