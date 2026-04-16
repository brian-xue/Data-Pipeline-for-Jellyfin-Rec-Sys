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
    base_profile_path: str | None
    online_user_sample_size: int
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
    movie_embeddings_npy_path: str
    movie_ids_path: str
    random_movie_injection_ratio: float
    candidate_request_top_k: int
    candidate_request_timeout_seconds: float
    memory_cleanup_every_ticks: int
    event_count_min: int
    event_count_max: int
    short_event_count_min: int
    short_event_count_max: int
    short_watch_duration_min_seconds: int
    short_watch_duration_max_seconds: int
    long_watch_duration_min_seconds: int
    long_watch_duration_max_seconds: int
    ranking_noise_min: float
    ranking_noise_max: float
    candidate_top_pool_min_size: int
    candidate_tail_pool_min_size: int
    candidate_pool_multiplier: int


@dataclass(frozen=True)
class IngestApiConfig:
    enabled: bool
    endpoint: str
    timeout_seconds: float


@dataclass(frozen=True)
class AppConfig:
    postgres: PostgresConfig
    simulator: SimulatorConfig
    random_seed: int = 42
    incremental_request: dict = field(default_factory=dict)
    ingest_api: IngestApiConfig = field(
        default_factory=lambda: IngestApiConfig(
            enabled=False,
            endpoint="http://api:8080/ingest/events",
            timeout_seconds=5.0,
        )
    )


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
        base_profile_path=_deep_get(raw, ["simulator", "base_profile_path"], "s3://artifacts/simulator_base_profile/simulator_base_profile.parquet"),
        online_user_sample_size=int(_deep_get(raw, ["simulator", "online_user_sample_size"], 100)),
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
        movie_embeddings_npy_path=str(_deep_get(raw, ["simulator", "movie_embeddings_npy_path"], "/data/embedding/embedding.npy")),
        movie_ids_path=str(_deep_get(raw, ["simulator", "movie_ids_path"], "/data/embedding/ids.npy")),
        random_movie_injection_ratio=float(_deep_get(raw, ["simulator", "random_movie_injection_ratio"], 0.1)),
        candidate_request_top_k=int(_deep_get(raw, ["simulator", "candidate_request_top_k"], 20)),
        candidate_request_timeout_seconds=float(_deep_get(raw, ["simulator", "candidate_request_timeout_seconds"], 10.0)),
        memory_cleanup_every_ticks=int(_deep_get(raw, ["simulator", "memory_cleanup_every_ticks"], 20)),
        event_count_min=int(_deep_get(raw, ["simulator", "event_count_min"], 1)),
        event_count_max=int(_deep_get(raw, ["simulator", "event_count_max"], 5)),
        short_event_count_min=int(_deep_get(raw, ["simulator", "short_event_count_min"], 0)),
        short_event_count_max=int(_deep_get(raw, ["simulator", "short_event_count_max"], 2)),
        short_watch_duration_min_seconds=int(_deep_get(raw, ["simulator", "short_watch_duration_min_seconds"], 1)),
        short_watch_duration_max_seconds=int(_deep_get(raw, ["simulator", "short_watch_duration_max_seconds"], 599)),
        long_watch_duration_min_seconds=int(_deep_get(raw, ["simulator", "long_watch_duration_min_seconds"], 600)),
        long_watch_duration_max_seconds=int(_deep_get(raw, ["simulator", "long_watch_duration_max_seconds"], 7200)),
        ranking_noise_min=float(_deep_get(raw, ["simulator", "ranking_noise_min"], -0.01)),
        ranking_noise_max=float(_deep_get(raw, ["simulator", "ranking_noise_max"], 0.01)),
        candidate_top_pool_min_size=int(_deep_get(raw, ["simulator", "candidate_top_pool_min_size"], 10)),
        candidate_tail_pool_min_size=int(_deep_get(raw, ["simulator", "candidate_tail_pool_min_size"], 10)),
        candidate_pool_multiplier=int(_deep_get(raw, ["simulator", "candidate_pool_multiplier"], 3)),
    )

    random_seed = int(_deep_get(raw, ["random_seed"], 42))
    incremental_request = raw.get("incremental_request", {})
    ingest_api = IngestApiConfig(
        enabled=bool(_deep_get(raw, ["ingest_api", "enabled"], False)),
        endpoint=str(_deep_get(raw, ["ingest_api", "endpoint"], "http://api:8080/ingest/events")),
        timeout_seconds=float(_deep_get(raw, ["ingest_api", "timeout_seconds"], 5.0)),
    )
    return AppConfig(
        postgres=postgres,
        simulator=simulator,
        random_seed=random_seed,
        incremental_request=incremental_request,
        ingest_api=ingest_api,
    )
