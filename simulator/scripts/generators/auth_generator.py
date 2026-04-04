from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def make_auth_event(
	user_id: int,
	session_id: str,
	event_type: str,
	event_time: datetime | None = None,
) -> dict[str, Any]:
	if event_type not in {"login", "logout"}:
		raise ValueError("event_type must be login or logout")
	ts = event_time or datetime.now(timezone.utc)
	return {
		"user_id": user_id,
		"session_id": session_id,
		"event_type": event_type,
		"event_time": ts,
	}

