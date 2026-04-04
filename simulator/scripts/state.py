from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class UserRuntime:
	session_id: str
	events_in_session: int = 0


@dataclass
class SimulatorState:
	online_users: set[int] = field(default_factory=set)
	offline_users: set[int] = field(default_factory=set)
	runtime_by_user: dict[int, UserRuntime] = field(default_factory=dict)

