import random
import time
import requests
from typing import Optional
from state import SimulatorState


def incremental_candidate_request(
    state: SimulatorState,
    interval_seconds: float,
    total_duration: float,
    online_service_url: str,
    min_interval: Optional[float] = None,
    max_interval: Optional[float] = None,
):
    """
    In spcified time interval, send request to online service
    interval_seconds: if min_interval and max_interval are not specified
    min_interval, max_interval: if specified, the interval between requests will be a random value between min_interval and max_interval
    online_service_url: eg http://localhost:8000/api/candidates
    """
    start_time = time.time()
    while time.time() - start_time < total_duration:
        if not state.online_users:
            print("[api request] no online users")
            time.sleep(1)
            continue
        user_id = random.choice(list(state.online_users))
        session_id = state.runtime_by_user[user_id].session_id
        payload = {"user_id": user_id, "session_id": session_id}
        try:
            resp = requests.get(
                    online_service_url,
                    params={"user_id": user_id, "top_k": 20},
                    timeout=10,
                )
            resp.raise_for_status()
            print(f"[api request succeeded] user_id={user_id} session_id={session_id} candidates={resp.json()}")
        except Exception as e:
            print(f"[api request failed] user_id={user_id} session_id={session_id} error={e}")
        if min_interval is not None and max_interval is not None:
            sleep_time = random.uniform(min_interval, max_interval)
        else:
            sleep_time = interval_seconds
        time.sleep(sleep_time)
