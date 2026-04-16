from __future__ import annotations

from typing import Any

import requests


def fetch_incremental_candidates(
    user_id: int,
    online_service_url: str,
    top_k: int = 20,
    timeout_seconds: float = 10.0,
) -> dict[str, Any] | None:
    """Request candidate items once for a user and return the parsed payload.

    The caller decides when to invoke this helper and whether to cache the
    result. This module does not manage loops, sleeps, or retry policies.
    """
    try:
        response = requests.get(
            online_service_url,
            params={"user_id": int(user_id), "top_k": int(top_k)},
            timeout=float(timeout_seconds),
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError(f"unexpected candidate payload type: {type(payload)}")
        return payload
    except Exception as exc:
        print(f"[api request failed] user_id={user_id} error={exc}")
        return None
