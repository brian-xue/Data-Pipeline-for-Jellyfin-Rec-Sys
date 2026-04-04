from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class AuthEvent:
	auth_event_id: int
	user_id: int
	session_id: str | None
	event_type: str
	event_time: datetime
	created_at: datetime | None = None
	metadata_json: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class UserEvent:
	event_id: int
	user_id: int
	movie_id: int
	session_id: str | None
	event_type: str
	event_time: datetime
	watch_duration_seconds: float
	created_at: datetime | None = None


@dataclass(frozen=True)
class ServiceCheckpoint:
	job_name: str
	last_auth_event_id: int = 0
	last_user_event_id: int = 0
	last_auth_event_time: datetime | None = None
	last_user_event_time: datetime | None = None
	status: str = "idle"
	updated_at: datetime | None = None
	created_at: datetime | None = None
	metadata_json: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CandidateMovie:
	movie_id: int
	score: float
	source: str
	rank: int | None = None
	metadata: dict[str, Any] = field(default_factory=dict)
