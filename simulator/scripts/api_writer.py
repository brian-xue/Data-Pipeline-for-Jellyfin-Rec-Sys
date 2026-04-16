from __future__ import annotations

from datetime import datetime
from typing import Any

import requests


class ApiEventWriter:
    def __init__(self, endpoint: str, timeout_seconds: float = 5.0):
        self.endpoint = endpoint.rstrip("/")
        self.timeout_seconds = float(timeout_seconds)
        self.auth_events: list[dict[str, Any]] = []
        self.user_events: list[dict[str, Any]] = []

    def _normalize_event_time(self, event_time: datetime | str) -> str:
        if isinstance(event_time, datetime):
            return event_time.isoformat()
        return str(event_time)

    def insert_auth_event(
        self,
        user_id: int,
        session_id: str,
        event_type: str,
        event_time: datetime,
        metadata_json: dict[str, Any] | None = None,
    ) -> None:
        self.auth_events.append(
            {
                "user_id": int(user_id),
                "session_id": str(session_id),
                "event_type": str(event_type),
                "event_time": self._normalize_event_time(event_time),
                "metadata_json": metadata_json or {},
            }
        )

    def insert_user_event(
        self,
        user_id: int,
        movie_id: int,
        session_id: str,
        event_time: datetime,
        watch_duration_seconds: float,
    ) -> None:
        self.user_events.append(
            {
                "user_id": int(user_id),
                "movie_id": int(movie_id),
                "session_id": str(session_id),
                "event_type": "finish",
                "event_time": self._normalize_event_time(event_time),
                "watch_duration_seconds": float(watch_duration_seconds),
            }
        )

    def commit(self) -> None:
        if not self.auth_events and not self.user_events:
            return

        payload = {
            "auth_events": self.auth_events,
            "user_events": self.user_events,
        }

        def _post_once() -> None:
            response = requests.post(
                self.endpoint,
                json=payload,
                timeout=self.timeout_seconds,
            )
            if response.status_code >= 400:
                raise RuntimeError(f"Ingest API failed: {response.status_code} {response.text}")

        try:
            _post_once()
        except Exception as first_error:
            print(f"[ingest commit] first attempt failed: {first_error}; retrying once")
            try:
                _post_once()
            except Exception as second_error:
                print(f"[ingest commit] retry failed, dropping current batch: {second_error}")
                self.auth_events.clear()
                self.user_events.clear()
                return

        self.auth_events.clear()
        self.user_events.clear()
