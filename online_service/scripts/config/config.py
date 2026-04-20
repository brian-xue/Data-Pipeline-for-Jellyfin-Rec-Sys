from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def _get_env_int(name: str, default: int) -> int:
	value = os.getenv(name)
	if value is None:
		return default
	return int(value)


def _get_env_float(name: str, default: float) -> float:
	value = os.getenv(name)
	if value is None:
		return default
	return float(value)


def _deep_get(data: dict[str, Any], keys: list[str], default: Any = None) -> Any:
	cur: Any = data
	for key in keys:
		if not isinstance(cur, dict) or key not in cur:
			return default
		cur = cur[key]
	return cur


def _load_yaml(path: str | None) -> dict[str, Any]:
	if not path:
		return {}
	config_file = Path(path)
	if not config_file.exists():
		raise FileNotFoundError(f"Config file not found: {path}")
	with config_file.open("r", encoding="utf-8") as handle:
		payload = yaml.safe_load(handle) or {}
	if not isinstance(payload, dict):
		raise ValueError("Online service config YAML must be a mapping/object")
	return payload


@dataclass(frozen=True)
class PostgresConfig:
    host: str
    port: int
    dbname: str
    user: str
    password: str
    connect_timeout_seconds: int = 5


@dataclass(frozen=True)
class ObjectStorageConfig:
    endpoint: str
    bucket: str
    access_key: str | None
    secret_key: str | None


@dataclass(frozen=True)
class ProcessorIntervals:
    auth_processor_seconds: int = 2
    event_processor_seconds: int = 2
    popular_movie_updater_seconds: int = 30
    user_embedding_updater_seconds: int = 600
    exporter_seconds: int = 600


@dataclass(frozen=True)
class EmbeddingConfig:
	recent_events_limit: int = 200
	min_watch_duration_seconds: float = 0.0
	version_prefix: str = "v"
	snapshot_root: str = "s3://artifacts/online-service-user-embeddings"


@dataclass(frozen=True)
class CandidateConfig:
	top_k_default: int = 100
	popular_fallback_k: int = 20
	movie_embeddings_npy_path: str = "movie_embeddings.npy"
	movie_embedding_index_path: str = "faiss.index"
	movie_ids_path: str | None = None
	hnsw_m: int = 32
	hnsw_ef_search: int = 64


@dataclass(frozen=True)
class UserHistoryQueryConfig:
	recent_limit: int = 200
	recent_window_hours: int = 24 * 30


@dataclass(frozen=True)
class ExporterConfig:
	snapshot_root: str = "s3://artifacts/online-service-export"
	auth_events_prefix: str = "auth_events"
	user_events_prefix: str = "user_events"
	write_parquet: bool = True


@dataclass(frozen=True)
class MonitoringConfig:
	window_minutes: int = 5
	short_watch_threshold_seconds: float = 10.0


@dataclass(frozen=True)
class ApiConfig:
	host: str = "0.0.0.0"
	port: int = 18080
	enabled: bool = True


@dataclass(frozen=True)
class OnlineServiceConfig:
	postgres: PostgresConfig
	object_storage: ObjectStorageConfig
	processor_intervals: ProcessorIntervals
	embedding: EmbeddingConfig
	candidate: CandidateConfig
	user_history_query: UserHistoryQueryConfig
	exporter: ExporterConfig
	monitoring: MonitoringConfig
	api: ApiConfig


def load_online_service_config(config_path: str | None = None) -> OnlineServiceConfig:
	resolved_path = config_path or os.getenv("ONLINE_SERVICE_CONFIG_PATH")
	yaml_cfg = _load_yaml(resolved_path)

	postgres = PostgresConfig(
		host=str(_deep_get(yaml_cfg, ["postgres", "host"], os.getenv("POSTGRES_HOST", "postgres"))),
		port=int(_deep_get(yaml_cfg, ["postgres", "port"], _get_env_int("POSTGRES_PORT", 5432))),
		dbname=str(_deep_get(yaml_cfg, ["postgres", "dbname"], os.getenv("POSTGRES_DB", "recsys"))),
		user=str(_deep_get(yaml_cfg, ["postgres", "user"], os.getenv("POSTGRES_USER", "recsys"))),
		password=str(_deep_get(yaml_cfg, ["postgres", "password"], os.getenv("POSTGRES_PASSWORD", "recsys"))),
		connect_timeout_seconds=int(
			_deep_get(
				yaml_cfg,
				["postgres", "connect_timeout_seconds"],
				_get_env_int("POSTGRES_CONNECT_TIMEOUT_SECONDS", 5),
			)
		),
	)

	object_storage = ObjectStorageConfig(
		endpoint=str(
			_deep_get(
				yaml_cfg,
				["object_storage", "endpoint"],
				os.getenv("OBJECT_STORAGE_ENDPOINT", "http://minio:9000"),
			)
		),
		bucket=str(
			_deep_get(
				yaml_cfg,
				["object_storage", "bucket"],
				os.getenv("OBJECT_STORAGE_BUCKET", "artifacts"),
			)
		),
		access_key=_deep_get(
			yaml_cfg,
			["object_storage", "access_key"],
			os.getenv("MINIO_ACCESS_KEY") or os.getenv("AWS_ACCESS_KEY_ID"),
		),
		secret_key=_deep_get(
			yaml_cfg,
			["object_storage", "secret_key"],
			os.getenv("MINIO_SECRET_KEY") or os.getenv("AWS_SECRET_ACCESS_KEY"),
		),
	)

	processor_intervals = ProcessorIntervals(
		auth_processor_seconds=int(
			_deep_get(
				yaml_cfg,
				["processor_intervals", "auth_processor_seconds"],
				_get_env_int("AUTH_PROCESSOR_INTERVAL_SECONDS", 2),
			)
		),
		event_processor_seconds=int(
			_deep_get(
				yaml_cfg,
				["processor_intervals", "event_processor_seconds"],
				_get_env_int("EVENT_PROCESSOR_INTERVAL_SECONDS", 2),
			)
		),
		popular_movie_updater_seconds=int(
			_deep_get(
				yaml_cfg,
				["processor_intervals", "popular_movie_updater_seconds"],
				_get_env_int("POPULAR_MOVIE_UPDATER_INTERVAL_SECONDS", 30),
			)
		),
		user_embedding_updater_seconds=int(
			_deep_get(
				yaml_cfg,
				["processor_intervals", "user_embedding_updater_seconds"],
				_get_env_int("USER_EMBEDDING_UPDATER_INTERVAL_SECONDS", 600),
			)
		),
		exporter_seconds=int(
			_deep_get(
				yaml_cfg,
				["processor_intervals", "exporter_seconds"],
				_get_env_int("EXPORTER_INTERVAL_SECONDS", 600),
			)
		),
	)

	embedding = EmbeddingConfig(
		recent_events_limit=int(
			_deep_get(
				yaml_cfg,
				["embedding", "recent_events_limit"],
				_get_env_int("EMBEDDING_RECENT_EVENTS_LIMIT", 200),
			)
		),
		min_watch_duration_seconds=float(
			_deep_get(
				yaml_cfg,
				["embedding", "min_watch_duration_seconds"],
				_get_env_float("EMBEDDING_MIN_WATCH_DURATION_SECONDS", 0.0),
			)
		),
		version_prefix=str(
			_deep_get(
				yaml_cfg,
				["embedding", "version_prefix"],
				os.getenv("EMBEDDING_VERSION_PREFIX", "v"),
			)
		),
		snapshot_root=str(
			_deep_get(
				yaml_cfg,
				["embedding", "snapshot_root"],
				os.getenv("EMBEDDING_SNAPSHOT_ROOT", "s3://artifacts/online-service-user-embeddings"),
			)
		),
	)

	candidate = CandidateConfig(
		top_k_default=int(
			_deep_get(
				yaml_cfg,
				["candidate", "top_k_default"],
				_get_env_int("CANDIDATE_TOP_K_DEFAULT", 100),
			)
		),
		popular_fallback_k=int(
			_deep_get(
				yaml_cfg,
				["candidate", "popular_fallback_k"],
				_get_env_int("CANDIDATE_POPULAR_FALLBACK_K", 20),
			)
		),
		movie_embeddings_npy_path=str(
			_deep_get(
				yaml_cfg,
				["candidate", "movie_embeddings_npy_path"],
				os.getenv("CANDIDATE_MOVIE_EMBEDDINGS_NPY_PATH", "movie_embeddings.npy"),
			)
		),
		movie_embedding_index_path=str(
			_deep_get(
				yaml_cfg,
				["candidate", "movie_embedding_index_path"],
				os.getenv("CANDIDATE_MOVIE_EMBEDDING_INDEX_PATH", "faiss.index"),
			)
		),
		movie_ids_path=_deep_get(
			yaml_cfg,
			["candidate", "movie_ids_path"],
			os.getenv("CANDIDATE_MOVIE_IDS_PATH"),
		),
		hnsw_m=int(
			_deep_get(
				yaml_cfg,
				["candidate", "hnsw_m"],
				_get_env_int("CANDIDATE_HNSW_M", 32),
			)
		),
		hnsw_ef_search=int(
			_deep_get(
				yaml_cfg,
				["candidate", "hnsw_ef_search"],
				_get_env_int("CANDIDATE_HNSW_EF_SEARCH", 64),
			)
		),
	)

	user_history_query = UserHistoryQueryConfig(
		recent_limit=int(
			_deep_get(
				yaml_cfg,
				["user_history_query", "recent_limit"],
				_get_env_int("USER_HISTORY_RECENT_LIMIT", 200),
			)
		),
		recent_window_hours=int(
			_deep_get(
				yaml_cfg,
				["user_history_query", "recent_window_hours"],
				_get_env_int("USER_HISTORY_RECENT_WINDOW_HOURS", 720),
			)
		),
	)

	exporter = ExporterConfig(
		snapshot_root=str(
			_deep_get(
				yaml_cfg,
				["exporter", "snapshot_root"],
				os.getenv("EXPORTER_SNAPSHOT_ROOT", "s3://artifacts/online-service-export"),
			)
		),
		auth_events_prefix=str(
			_deep_get(
				yaml_cfg,
				["exporter", "auth_events_prefix"],
				os.getenv("EXPORTER_AUTH_EVENTS_PREFIX", "auth_events"),
			)
		),
		user_events_prefix=str(
			_deep_get(
				yaml_cfg,
				["exporter", "user_events_prefix"],
				os.getenv("EXPORTER_USER_EVENTS_PREFIX", "user_events"),
			)
		),
		write_parquet=bool(
			_deep_get(
				yaml_cfg,
				["exporter", "write_parquet"],
				os.getenv("EXPORTER_WRITE_PARQUET", "true").lower() in {"1", "true", "yes"},
			)
		),
	)

	monitoring = MonitoringConfig(
		window_minutes=int(
			_deep_get(
				yaml_cfg,
				["monitoring", "window_minutes"],
				_get_env_int("MONITORING_WINDOW_MINUTES", 5),
			)
		),
		short_watch_threshold_seconds=float(
			_deep_get(
				yaml_cfg,
				["monitoring", "short_watch_threshold_seconds"],
				_get_env_float("MONITORING_SHORT_WATCH_THRESHOLD_SECONDS", 10.0),
			)
		),
	)

	api = ApiConfig(
		host=str(_deep_get(yaml_cfg, ["api", "host"], os.getenv("API_HOST", "0.0.0.0"))),
		port=int(_deep_get(yaml_cfg, ["api", "port"], _get_env_int("API_PORT", 18080))),
		enabled=bool(
			_deep_get(
				yaml_cfg,
				["api", "enabled"],
				os.getenv("API_ENABLED", "true").lower() in {"1", "true", "yes"},
			)
		),
	)

	return OnlineServiceConfig(
		postgres=postgres,
		object_storage=object_storage,
		processor_intervals=processor_intervals,
		embedding=embedding,
		candidate=candidate,
		user_history_query=user_history_query,
		exporter=exporter,
		monitoring=monitoring,
		api=api,
	)
