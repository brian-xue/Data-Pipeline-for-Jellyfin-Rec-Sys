import os
import math
import json
import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

try:
    from pyiceberg.catalog import load_catalog
except Exception:
    load_catalog = None


# ============================================================
# Config utils
# ============================================================
def load_yaml_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def deep_get(d: Dict[str, Any], keys: List[str], default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


# ============================================================
# Arrow schemas
# ============================================================
BASE_USER_EVENTS_SCHEMA = pa.schema([
    pa.field("user_id", pa.int64()),
    pa.field("movie_id", pa.int64()),
    pa.field("rating", pa.float64()),
    pa.field("timestamp", pa.int64()),
    pa.field("event_time", pa.timestamp("us", tz="UTC")),
    pa.field("event_order", pa.int32()),
    pa.field("is_positive", pa.bool_()),
    pa.field("rating_centered", pa.float64()),
    pa.field("time_weight", pa.float64()),
])

REMAINING_USER_EVENTS_SCHEMA = pa.schema([
    pa.field("user_id", pa.int64()),
    pa.field("movie_id", pa.int64()),
    pa.field("rating", pa.float64()),
    pa.field("timestamp", pa.int64()),
    pa.field("event_time", pa.timestamp("us", tz="UTC")),
    pa.field("future_event_order", pa.int32()),
])

BASE_USERS_SCHEMA = pa.schema([
    pa.field("user_id", pa.int64()),
    pa.field("num_total_interactions", pa.int32()),
    pa.field("num_bootstrap_interactions", pa.int32()),
    pa.field("num_remaining_interactions", pa.int32()),
    pa.field("first_timestamp", pa.int64()),
    pa.field("last_bootstrap_timestamp", pa.int64()),
    pa.field("last_total_timestamp", pa.int64()),
    pa.field("avg_rating_bootstrap", pa.float64()),
    pa.field("std_rating_bootstrap", pa.float64()),
    pa.field("activity_span_days", pa.float64()),
    pa.field("profile_confidence", pa.float64()),
    pa.field("built_at", pa.timestamp("us", tz="UTC")),
])

BASE_USER_PROFILES_SCHEMA = pa.schema([
    pa.field("user_id", pa.int64()),
    pa.field("long_term_embedding", pa.list_(pa.float32())),
    pa.field("short_term_embedding", pa.list_(pa.float32())),
    pa.field("embedding_dim", pa.int32()),
    pa.field("num_embedded_bootstrap_interactions", pa.int32()),
    pa.field("num_missing_embedding_movies", pa.int32()),
    pa.field("rating_bias", pa.float64()),
    pa.field("activity_level", pa.float64()),
    pa.field("profile_version", pa.string()),
    pa.field("built_at", pa.timestamp("us", tz="UTC")),
])


# ============================================================
# Incremental parquet writer
# ============================================================
class IncrementalParquetWriter:
    def __init__(self, path: str, schema: pa.Schema):
        self.path = path
        self.schema = schema
        self.writer: Optional[pq.ParquetWriter] = None
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def write_rows(self, rows: List[Dict[str, Any]]) -> None:
        if not rows:
            return
        table = rows_to_arrow(rows, self.schema)
        if self.writer is None:
            self.writer = pq.ParquetWriter(
                self.path,
                self.schema,
                compression="zstd",
            )
        self.writer.write_table(table)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None


def rows_to_arrow(rows: List[Dict[str, Any]], schema: pa.Schema) -> pa.Table:
    cols = {}
    for field in schema:
        vals = [row.get(field.name) for row in rows]
        cols[field.name] = pa.array(vals, type=field.type)
    return pa.Table.from_pydict(cols, schema=schema)


# ============================================================
# Input cleaning
# ============================================================
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for col in df.columns:
        c = col.strip()
        low = c.lower()
        if low == "userid":
            rename_map[col] = "user_id"
        elif low == "movieid":
            rename_map[col] = "movie_id"
        elif low == "rating":
            rename_map[col] = "rating"
        elif low == "timestamp":
            rename_map[col] = "timestamp"

    df = df.rename(columns=rename_map)

    required = ["user_id", "movie_id", "rating", "timestamp"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df[required].copy()


def clean_chunk(
    df: pd.DataFrame,
    allowed_rating_min: float,
    allowed_rating_max: float,
    drop_invalid_rows: bool,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    stats = {
        "rows_in": len(df),
        "rows_dropped_null": 0,
        "rows_dropped_invalid": 0,
        "rows_out": 0,
    }

    df = normalize_columns(df)

    for col in ["user_id", "movie_id", "rating", "timestamp"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    before_null = len(df)
    df = df.dropna(subset=["user_id", "movie_id", "rating", "timestamp"])
    stats["rows_dropped_null"] = before_null - len(df)

    df["user_id"] = df["user_id"].astype("int64")
    df["movie_id"] = df["movie_id"].astype("int64")
    df["rating"] = df["rating"].astype("float64")
    df["timestamp"] = df["timestamp"].astype("int64")

    if drop_invalid_rows:
        before_invalid = len(df)
        df = df[
            (df["timestamp"] > 0) &
            (df["rating"] >= allowed_rating_min) &
            (df["rating"] <= allowed_rating_max)
        ].copy()
        stats["rows_dropped_invalid"] = before_invalid - len(df)

    stats["rows_out"] = len(df)
    return df, stats


# ============================================================
# Movie embedding loading
# ============================================================
def load_movie_embeddings(parquet_path: str) -> Tuple[Dict[int, np.ndarray], int]:
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"movie embedding parquet not found: {parquet_path}")

    df = pd.read_parquet(parquet_path, columns=["movieId", "embedding"])
    df = df.dropna(subset=["movieId", "embedding"]).copy()

    movie_embedding_map: Dict[int, np.ndarray] = {}
    embedding_dim: Optional[int] = None

    for row in df.itertuples(index=False):
        try:
            movie_id = int(row.movieId)
            emb = np.asarray(row.embedding, dtype=np.float32)
        except Exception:
            continue

        if emb.ndim != 1:
            continue

        if embedding_dim is None:
            embedding_dim = int(emb.shape[0])

        if emb.shape[0] != embedding_dim:
            continue

        movie_embedding_map[movie_id] = emb

    if embedding_dim is None or not movie_embedding_map:
        raise ValueError("No valid embeddings found in movie embedding parquet")

    return movie_embedding_map, embedding_dim


# ============================================================
# User embedding aggregation
# ============================================================
def weighted_average_embedding(
    movie_ids: List[int],
    ratings: List[float],
    time_weights: List[float],
    movie_embedding_map: Dict[int, np.ndarray],
    preference_anchor_rating: float,
    embedding_dim: int,
    min_positive_weight_sum: float,
) -> Tuple[Optional[np.ndarray], int, int]:
    vec_sum = np.zeros(embedding_dim, dtype=np.float32)
    weight_sum = 0.0
    used_count = 0
    missing_count = 0

    for movie_id, rating, t_weight in zip(movie_ids, ratings, time_weights):
        emb = movie_embedding_map.get(int(movie_id))
        if emb is None:
            missing_count += 1
            continue

        pref_weight = max(float(rating) - preference_anchor_rating, 0.0)
        final_weight = pref_weight * float(t_weight)

        if final_weight <= 0:
            continue

        vec_sum += final_weight * emb
        weight_sum += final_weight
        used_count += 1

    if weight_sum <= min_positive_weight_sum:
        return None, used_count, missing_count

    return (vec_sum / weight_sum).astype(np.float32), used_count, missing_count


# ============================================================
# Per-user processing
# ============================================================
@dataclass
class UserBuildResult:
    base_user_events: List[Dict[str, Any]]
    remaining_user_events: List[Dict[str, Any]]
    base_users: List[Dict[str, Any]]
    base_user_profiles: List[Dict[str, Any]]
    stats: Dict[str, int]


def process_single_user(
    user_df: pd.DataFrame,
    min_interactions_per_user: int,
    bootstrap_ratio: float,
    min_bootstrap_interactions: int,
    min_remaining_interactions: int,
    deduplicate_user_movie: bool,
    positive_rating_threshold: float,
    recent_k: int,
    half_life_days: int,
    profile_version: str,
    movie_embedding_map: Dict[int, np.ndarray],
    embedding_dim: int,
    preference_anchor_rating: float,
    min_positive_weight_sum: float,
) -> UserBuildResult:
    result = UserBuildResult([], [], [], [], {
        "users_seen": 1,
        "users_kept": 0,
        "users_skipped_too_few": 0,
        "bootstrap_events": 0,
        "remaining_events": 0,
        "users_with_long_embedding": 0,
        "users_with_short_embedding": 0,
    })

    if user_df.empty:
        result.stats["users_skipped_too_few"] = 1
        return result

    user_df = user_df.sort_values("timestamp", kind="stable").copy()

    if deduplicate_user_movie:
        user_df = user_df.drop_duplicates(subset=["user_id", "movie_id"], keep="last").copy()
        user_df = user_df.sort_values("timestamp", kind="stable").copy()

    n = len(user_df)
    if n < min_interactions_per_user:
        result.stats["users_skipped_too_few"] = 1
        return result

    bootstrap_n = max(min_bootstrap_interactions, int(math.floor(n * bootstrap_ratio)))
    max_bootstrap_allowed = n - min_remaining_interactions
    bootstrap_n = min(bootstrap_n, max_bootstrap_allowed)

    if bootstrap_n < min_bootstrap_interactions or (n - bootstrap_n) < min_remaining_interactions:
        result.stats["users_skipped_too_few"] = 1
        return result

    boot = user_df.iloc[:bootstrap_n].copy()
    remain = user_df.iloc[bootstrap_n:].copy()

    if boot.empty or remain.empty:
        result.stats["users_skipped_too_few"] = 1
        return result

    user_id = int(boot["user_id"].iloc[0])

    boot["event_order"] = np.arange(1, len(boot) + 1, dtype=np.int32)
    boot["event_time"] = pd.to_datetime(boot["timestamp"], unit="s", utc=True)

    boot_avg_rating = float(boot["rating"].mean())
    boot["is_positive"] = boot["rating"] >= positive_rating_threshold
    boot["rating_centered"] = boot["rating"] - boot_avg_rating

    user_last_ts = int(boot["timestamp"].max())
    diff_days = (user_last_ts - boot["timestamp"]) / 86400.0
    decay_lambda = np.log(2) / max(half_life_days, 1)
    boot["time_weight"] = np.exp(-decay_lambda * diff_days)

    remain["future_event_order"] = np.arange(1, len(remain) + 1, dtype=np.int32)
    remain["event_time"] = pd.to_datetime(remain["timestamp"], unit="s", utc=True)

    # Long-term embedding
    long_emb, long_used, long_missing = weighted_average_embedding(
        movie_ids=boot["movie_id"].astype("int64").tolist(),
        ratings=boot["rating"].astype("float64").tolist(),
        time_weights=boot["time_weight"].astype("float64").tolist(),
        movie_embedding_map=movie_embedding_map,
        preference_anchor_rating=preference_anchor_rating,
        embedding_dim=embedding_dim,
        min_positive_weight_sum=min_positive_weight_sum,
    )

    # Short-term embedding
    recent_boot = boot.tail(recent_k).copy()
    short_emb, short_used, short_missing = weighted_average_embedding(
        movie_ids=recent_boot["movie_id"].astype("int64").tolist(),
        ratings=recent_boot["rating"].astype("float64").tolist(),
        time_weights=recent_boot["time_weight"].astype("float64").tolist(),
        movie_embedding_map=movie_embedding_map,
        preference_anchor_rating=preference_anchor_rating,
        embedding_dim=embedding_dim,
        min_positive_weight_sum=min_positive_weight_sum,
    )

    built_at = datetime.now(timezone.utc)

    result.base_user_events = [
        {
            "user_id": int(r.user_id),
            "movie_id": int(r.movie_id),
            "rating": float(r.rating),
            "timestamp": int(r.timestamp),
            "event_time": r.event_time.to_pydatetime(),
            "event_order": int(r.event_order),
            "is_positive": bool(r.is_positive),
            "rating_centered": float(r.rating_centered),
            "time_weight": float(r.time_weight),
        }
        for r in boot.itertuples(index=False)
    ]

    result.remaining_user_events = [
        {
            "user_id": int(r.user_id),
            "movie_id": int(r.movie_id),
            "rating": float(r.rating),
            "timestamp": int(r.timestamp),
            "event_time": r.event_time.to_pydatetime(),
            "future_event_order": int(r.future_event_order),
        }
        for r in remain.itertuples(index=False)
    ]

    std_rating = float(boot["rating"].std(ddof=1)) if len(boot) > 1 else 0.0

    result.base_users = [{
        "user_id": user_id,
        "num_total_interactions": int(n),
        "num_bootstrap_interactions": int(len(boot)),
        "num_remaining_interactions": int(len(remain)),
        "first_timestamp": int(user_df["timestamp"].min()),
        "last_bootstrap_timestamp": int(boot["timestamp"].max()),
        "last_total_timestamp": int(user_df["timestamp"].max()),
        "avg_rating_bootstrap": float(boot_avg_rating),
        "std_rating_bootstrap": std_rating,
        "activity_span_days": float((int(user_df["timestamp"].max()) - int(user_df["timestamp"].min())) / 86400.0),
        "profile_confidence": float(np.log1p(len(boot))),
        "built_at": built_at,
    }]

    result.base_user_profiles = [{
        "user_id": user_id,
        "long_term_embedding": long_emb.tolist() if long_emb is not None else None,
        "short_term_embedding": short_emb.tolist() if short_emb is not None else None,
        "embedding_dim": int(embedding_dim),
        "num_embedded_bootstrap_interactions": int(long_used),
        "num_missing_embedding_movies": int(long_missing),
        "rating_bias": float(boot_avg_rating),
        "activity_level": float(len(boot)),
        "profile_version": profile_version,
        "built_at": built_at,
    }]

    result.stats["users_kept"] = 1
    result.stats["bootstrap_events"] = len(boot)
    result.stats["remaining_events"] = len(remain)

    if long_emb is not None:
        result.stats["users_with_long_embedding"] = 1
    if short_emb is not None:
        result.stats["users_with_short_embedding"] = 1

    return result


# ============================================================
# Optional iceberg append (existing tables only)
# ============================================================
def append_parquet_to_existing_iceberg_table(
    catalog_name: str,
    table_identifier: str,
    parquet_path: str,
    batch_rows: int = 50000,
) -> None:
    if load_catalog is None:
        raise RuntimeError("pyiceberg is not installed or failed to import")

    catalog = load_catalog(catalog_name)
    table = catalog.load_table(table_identifier)
    parquet_file = pq.ParquetFile(parquet_path)

    for batch in parquet_file.iter_batches(batch_size=batch_rows):
        arrow_table = pa.Table.from_batches([batch])
        table.append(arrow_table)


# ============================================================
# Main
# ============================================================
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)

    job_name = deep_get(cfg, ["job", "name"], "bootstrap_base_users")
    profile_version = deep_get(cfg, ["job", "profile_version"], "bootstraper_v0_embedding")

    ratings_path = deep_get(cfg, ["input", "ratings_path"])
    movie_embedding_path = deep_get(cfg, ["input", "movie_embedding_path"])
    chunksize = int(deep_get(cfg, ["input", "chunksize"], 200000))

    allowed_rating_min = float(deep_get(cfg, ["cleaning", "allowed_rating_min"], 0.5))
    allowed_rating_max = float(deep_get(cfg, ["cleaning", "allowed_rating_max"], 5.0))
    drop_invalid_rows = bool(deep_get(cfg, ["cleaning", "drop_invalid_rows"], True))

    min_interactions_per_user = int(deep_get(cfg, ["filter", "min_interactions_per_user"], 5))

    bootstrap_ratio = float(deep_get(cfg, ["bootstrap", "ratio"], 0.7))
    min_bootstrap_interactions = int(deep_get(cfg, ["bootstrap", "min_bootstrap_interactions"], 3))
    min_remaining_interactions = int(deep_get(cfg, ["bootstrap", "min_remaining_interactions"], 1))
    deduplicate_user_movie = bool(deep_get(cfg, ["bootstrap", "deduplicate_user_movie"], False))

    positive_rating_threshold = float(deep_get(cfg, ["profile", "positive_rating_threshold"], 4.0))
    recent_k = int(deep_get(cfg, ["profile", "recent_k"], 10))
    half_life_days = int(deep_get(cfg, ["profile", "half_life_days"], 180))
    preference_anchor_rating = float(deep_get(cfg, ["profile", "preference_anchor_rating"], 3.5))
    min_positive_weight_sum = float(deep_get(cfg, ["profile", "min_positive_weight_sum"], 1e-8))

    base_dir = deep_get(cfg, ["output", "base_dir"], "/data/artifacts/bootstraper_v0_embedding")
    write_parquet = bool(deep_get(cfg, ["output", "write_parquet"], True))

    iceberg_enabled = bool(deep_get(cfg, ["iceberg", "enabled"], False))
    catalog_name = deep_get(cfg, ["iceberg", "catalog_name"], "default")
    namespace = deep_get(cfg, ["iceberg", "namespace"], "recsys")
    append_batch_rows = int(deep_get(cfg, ["iceberg", "append_batch_rows"], 50000))
    iceberg_tables = deep_get(cfg, ["iceberg", "tables"], {})

    manifest_path = deep_get(cfg, ["artifact", "manifest_path"], os.path.join(base_dir, "manifest.json"))
    log_every_users = int(deep_get(cfg, ["logging", "log_every_users"], 5000))

    if not ratings_path:
        raise ValueError("input.ratings_path is required")
    if not movie_embedding_path:
        raise ValueError("input.movie_embedding_path is required")

    os.makedirs(base_dir, exist_ok=True)

    base_user_events_path = os.path.join(base_dir, "base_user_events.parquet")
    base_users_path = os.path.join(base_dir, "base_users.parquet")
    base_user_profiles_path = os.path.join(base_dir, "base_user_profiles.parquet")
    remaining_user_events_path = os.path.join(base_dir, "remaining_user_events.parquet")

    print(f"[{job_name}] loading movie embeddings from {movie_embedding_path}")
    movie_embedding_map, embedding_dim = load_movie_embeddings(movie_embedding_path)
    print(f"[{job_name}] loaded movie embeddings: {len(movie_embedding_map)}, dim={embedding_dim}")

    writers = {}
    if write_parquet:
        writers["base_user_events"] = IncrementalParquetWriter(base_user_events_path, BASE_USER_EVENTS_SCHEMA)
        writers["base_users"] = IncrementalParquetWriter(base_users_path, BASE_USERS_SCHEMA)
        writers["base_user_profiles"] = IncrementalParquetWriter(base_user_profiles_path, BASE_USER_PROFILES_SCHEMA)
        writers["remaining_user_events"] = IncrementalParquetWriter(remaining_user_events_path, REMAINING_USER_EVENTS_SCHEMA)

    global_stats = {
        "chunks_seen": 0,
        "rows_in_raw": 0,
        "rows_after_cleaning": 0,
        "rows_dropped_null": 0,
        "rows_dropped_invalid": 0,
        "users_seen": 0,
        "users_kept": 0,
        "users_skipped_too_few": 0,
        "users_with_long_embedding": 0,
        "users_with_short_embedding": 0,
        "bootstrap_events": 0,
        "remaining_events": 0,
    }

    pending_user_df: Optional[pd.DataFrame] = None

    print(f"[{job_name}] start reading ratings: {ratings_path}")
    chunk_iter = pd.read_csv(ratings_path, chunksize=chunksize)

    for chunk_idx, raw_chunk in enumerate(chunk_iter, start=1):
        global_stats["chunks_seen"] += 1
        global_stats["rows_in_raw"] += len(raw_chunk)

        clean_df, clean_stats = clean_chunk(
            raw_chunk,
            allowed_rating_min=allowed_rating_min,
            allowed_rating_max=allowed_rating_max,
            drop_invalid_rows=drop_invalid_rows,
        )

        global_stats["rows_after_cleaning"] += clean_stats["rows_out"]
        global_stats["rows_dropped_null"] += clean_stats["rows_dropped_null"]
        global_stats["rows_dropped_invalid"] += clean_stats["rows_dropped_invalid"]

        if pending_user_df is not None and not pending_user_df.empty:
            clean_df = pd.concat([pending_user_df, clean_df], ignore_index=True)
            pending_user_df = None

        if clean_df.empty:
            continue

        # 原始文件按 user_id 排序，因此只需要把最后一个用户留到下一块
        last_user_id = int(clean_df["user_id"].iloc[-1])
        finalized_df = clean_df[clean_df["user_id"] != last_user_id].copy()
        pending_user_df = clean_df[clean_df["user_id"] == last_user_id].copy()

        if not finalized_df.empty:
            for _, user_df in finalized_df.groupby("user_id", sort=False):
                build_result = process_single_user(
                    user_df=user_df,
                    min_interactions_per_user=min_interactions_per_user,
                    bootstrap_ratio=bootstrap_ratio,
                    min_bootstrap_interactions=min_bootstrap_interactions,
                    min_remaining_interactions=min_remaining_interactions,
                    deduplicate_user_movie=deduplicate_user_movie,
                    positive_rating_threshold=positive_rating_threshold,
                    recent_k=recent_k,
                    half_life_days=half_life_days,
                    profile_version=profile_version,
                    movie_embedding_map=movie_embedding_map,
                    embedding_dim=embedding_dim,
                    preference_anchor_rating=preference_anchor_rating,
                    min_positive_weight_sum=min_positive_weight_sum,
                )

                for k, v in build_result.stats.items():
                    global_stats[k] += v

                if write_parquet:
                    writers["base_user_events"].write_rows(build_result.base_user_events)
                    writers["remaining_user_events"].write_rows(build_result.remaining_user_events)
                    writers["base_users"].write_rows(build_result.base_users)
                    writers["base_user_profiles"].write_rows(build_result.base_user_profiles)

                if global_stats["users_seen"] > 0 and global_stats["users_seen"] % log_every_users == 0:
                    print(
                        f"[{job_name}] users_seen={global_stats['users_seen']}, "
                        f"users_kept={global_stats['users_kept']}, "
                        f"users_with_long_embedding={global_stats['users_with_long_embedding']}, "
                        f"bootstrap_events={global_stats['bootstrap_events']}, "
                        f"remaining_events={global_stats['remaining_events']}"
                    )

        if chunk_idx % 10 == 0:
            print(
                f"[{job_name}] chunk={chunk_idx}, "
                f"rows_in_raw={global_stats['rows_in_raw']}, "
                f"rows_after_cleaning={global_stats['rows_after_cleaning']}"
            )

    if pending_user_df is not None and not pending_user_df.empty:
        build_result = process_single_user(
            user_df=pending_user_df,
            min_interactions_per_user=min_interactions_per_user,
            bootstrap_ratio=bootstrap_ratio,
            min_bootstrap_interactions=min_bootstrap_interactions,
            min_remaining_interactions=min_remaining_interactions,
            deduplicate_user_movie=deduplicate_user_movie,
            positive_rating_threshold=positive_rating_threshold,
            recent_k=recent_k,
            half_life_days=half_life_days,
            profile_version=profile_version,
            movie_embedding_map=movie_embedding_map,
            embedding_dim=embedding_dim,
            preference_anchor_rating=preference_anchor_rating,
            min_positive_weight_sum=min_positive_weight_sum,
        )

        for k, v in build_result.stats.items():
            global_stats[k] += v

        if write_parquet:
            writers["base_user_events"].write_rows(build_result.base_user_events)
            writers["remaining_user_events"].write_rows(build_result.remaining_user_events)
            writers["base_users"].write_rows(build_result.base_users)
            writers["base_user_profiles"].write_rows(build_result.base_user_profiles)

    for writer in writers.values():
        writer.close()

    if iceberg_enabled:
        if not write_parquet:
            raise ValueError("iceberg.enabled=true requires output.write_parquet=true")

        print(f"[{job_name}] appending parquet outputs into existing Iceberg tables")

        append_parquet_to_existing_iceberg_table(
            catalog_name=catalog_name,
            table_identifier=f"{namespace}.{iceberg_tables['base_user_events']}",
            parquet_path=base_user_events_path,
            batch_rows=append_batch_rows,
        )
        append_parquet_to_existing_iceberg_table(
            catalog_name=catalog_name,
            table_identifier=f"{namespace}.{iceberg_tables['base_users']}",
            parquet_path=base_users_path,
            batch_rows=append_batch_rows,
        )
        append_parquet_to_existing_iceberg_table(
            catalog_name=catalog_name,
            table_identifier=f"{namespace}.{iceberg_tables['base_user_profiles']}",
            parquet_path=base_user_profiles_path,
            batch_rows=append_batch_rows,
        )
        append_parquet_to_existing_iceberg_table(
            catalog_name=catalog_name,
            table_identifier=f"{namespace}.{iceberg_tables['remaining_user_events']}",
            parquet_path=remaining_user_events_path,
            batch_rows=append_batch_rows,
        )

    manifest = {
        "job_name": job_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "profile_version": profile_version,
        "input": {
            "ratings_path": ratings_path,
            "movie_embedding_path": movie_embedding_path,
            "chunksize": chunksize,
            "embedding_dim": embedding_dim,
            "num_movie_embeddings": len(movie_embedding_map),
        },
        "cleaning": {
            "allowed_rating_min": allowed_rating_min,
            "allowed_rating_max": allowed_rating_max,
            "drop_invalid_rows": drop_invalid_rows,
        },
        "filter": {
            "min_interactions_per_user": min_interactions_per_user,
        },
        "bootstrap": {
            "ratio": bootstrap_ratio,
            "min_bootstrap_interactions": min_bootstrap_interactions,
            "min_remaining_interactions": min_remaining_interactions,
            "deduplicate_user_movie": deduplicate_user_movie,
        },
        "profile": {
            "positive_rating_threshold": positive_rating_threshold,
            "recent_k": recent_k,
            "half_life_days": half_life_days,
            "preference_anchor_rating": preference_anchor_rating,
            "min_positive_weight_sum": min_positive_weight_sum,
        },
        "output": {
            "base_dir": base_dir,
            "parquet_files": {
                "base_user_events": base_user_events_path,
                "base_users": base_users_path,
                "base_user_profiles": base_user_profiles_path,
                "remaining_user_events": remaining_user_events_path,
            },
        },
        "iceberg": {
            "enabled": iceberg_enabled,
            "catalog_name": catalog_name,
            "namespace": namespace,
            "tables": iceberg_tables,
        },
        "stats": global_stats,
    }

    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"[{job_name}] done")
    print(json.dumps(global_stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()