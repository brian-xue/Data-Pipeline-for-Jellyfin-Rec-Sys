from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class PostgresConfig:
    host: str
    port: int
    dbname: str
    user: str
    password: str


@dataclass(frozen=True)
class SimulatorConfig:
    user_pool_size: int
    min_user_id: int
    max_user_id: int
    target_online_users: int
    max_online_users: int
    tick_seconds: float
    total_ticks: int
    login_rate_per_tick: int
    logout_rate_per_tick: int
    global_event_rate_per_tick: int
    per_user_event_prob: float
    min_events_per_session: int
    max_events_per_session: int
    min_movie_id: int
    max_movie_id: int
    min_watch_duration_seconds: float
    max_watch_duration_seconds: float


@dataclass(frozen=True)
class AppConfig:
    postgres: PostgresConfig
    simulator: SimulatorConfig
    random_seed: int = 42
    incremental_request: dict = field(default_factory=dict)


def _deep_get(data: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def load_config(path: str) -> AppConfig:
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with cfg_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    postgres = PostgresConfig(
        host=str(_deep_get(raw, ["postgres", "host"], "postgres")),
        port=int(_deep_get(raw, ["postgres", "port"], 5432)),
        dbname=str(_deep_get(raw, ["postgres", "dbname"], "recsys")),
        user=str(_deep_get(raw, ["postgres", "user"], "recsys")),
        password=str(_deep_get(raw, ["postgres", "password"], "recsys")),
    )

    simulator = SimulatorConfig(
        user_pool_size=int(_deep_get(raw, ["simulator", "user_pool_size"], 5000)),
        min_user_id=int(_deep_get(raw, ["simulator", "min_user_id"], 10000000)),
        max_user_id=int(_deep_get(raw, ["simulator", "max_user_id"], 99999999)),
        target_online_users=int(_deep_get(raw, ["simulator", "target_online_users"], 100)),
        max_online_users=int(_deep_get(raw, ["simulator", "max_online_users"], 120)),
        tick_seconds=float(_deep_get(raw, ["simulator", "tick_seconds"], 1.0)),
        total_ticks=int(_deep_get(raw, ["simulator", "total_ticks"], 3600)),
        login_rate_per_tick=int(_deep_get(raw, ["simulator", "login_rate_per_tick"], 10)),
        logout_rate_per_tick=int(_deep_get(raw, ["simulator", "logout_rate_per_tick"], 5)),
        global_event_rate_per_tick=int(_deep_get(raw, ["simulator", "global_event_rate_per_tick"], 50)),
        per_user_event_prob=float(_deep_get(raw, ["simulator", "per_user_event_prob"], 0.3)),
        min_events_per_session=int(_deep_get(raw, ["simulator", "min_events_per_session"], 1)),
        max_events_per_session=int(_deep_get(raw, ["simulator", "max_events_per_session"], 10)),
        min_movie_id=int(_deep_get(raw, ["simulator", "min_movie_id"], 1)),
        max_movie_id=int(_deep_get(raw, ["simulator", "max_movie_id"], 292757)),
        min_watch_duration_seconds=float(_deep_get(raw, ["simulator", "min_watch_duration_seconds"], 60.0)),
        max_watch_duration_seconds=float(_deep_get(raw, ["simulator", "max_watch_duration_seconds"], 7200.0)),
    )

    random_seed = int(_deep_get(raw, ["random_seed"], 42))
    incremental_request = raw.get("incremental_request", {})
    return AppConfig(postgres=postgres, simulator=simulator, random_seed=random_seed, incremental_request=incremental_request)
