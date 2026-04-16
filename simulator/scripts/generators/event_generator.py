from __future__ import annotations

import random
from datetime import datetime, timezone
from typing import Any


def make_finish_event(
	user_id: int,
	session_id: str,
	min_movie_id: int,
	max_movie_id: int,
	min_watch_duration_seconds: float,
	max_watch_duration_seconds: float,
	movie_id: int | None = None,
	watch_duration_seconds: float | None = None,
) -> dict[str, Any]:
	if max_movie_id < min_movie_id:
		raise ValueError("max_movie_id must be >= min_movie_id")
	if max_watch_duration_seconds < min_watch_duration_seconds:
		raise ValueError("max_watch_duration_seconds must be >= min_watch_duration_seconds")

	selected_movie_id = movie_id if movie_id is not None else random.randint(min_movie_id, max_movie_id)
	watch_duration = (
		float(watch_duration_seconds)
		if watch_duration_seconds is not None
		else random.uniform(min_watch_duration_seconds, max_watch_duration_seconds)
	)
	event_time = datetime.now(timezone.utc)
	return {
		"user_id": user_id,
		"movie_id": selected_movie_id,
		"session_id": session_id,
		"event_type": "finish",
		"event_time": event_time,
		"watch_duration_seconds": watch_duration,
	}

