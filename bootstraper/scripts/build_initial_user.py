import os
import gc
import math
import json
import argparse
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from minio_s3 import to_s3_uri, upload_bytes_to_path, upload_file_to_path, s3_filesystem

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
# IO
# ============================================================
def open_input_binary(path_like: str):
    path_str = str(path_like).strip()
    is_s3 = path_str.startswith("s3://")

    if not is_s3 or os.path.exists(path_str):
        local_path = path_str.replace("s3://", "/data/") if is_s3 else path_str
        if os.path.exists(local_path) and os.path.isfile(local_path):
            print(f"[Local IO] Opening physical file: {local_path}")
            return open(local_path, "rb")

    print(f"[S3 IO] Opening stream via s3fs: {path_str}")
    fs = s3_filesystem()
    try:
        s3_uri = to_s3_uri(path_str)
    except ValueError:
        s3_uri = path_str

    return fs.open(
        s3_uri,
        "rb",
        cache_type="readahead",
        block_size=8 * 1024 * 1024,   # 比原来更保守一点
    )


def csv_chunks_from_input(path_like: str, chunksize: int):
    with open_input_binary(path_like) as handle:
        yield from pd.read_csv(
            handle,
            chunksize=chunksize,
            low_memory=False,
        )


def estimate_chunks(path, chunksize):
    import pandas as pd

    total_rows = 0
    for chunk in pd.read_csv(path, chunksize=chunksize, usecols=[0]):  # 只读一列
        total_rows += len(chunk)

    total_chunks = (total_rows + chunksize - 1) // chunksize
    return total_rows, total_chunks

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
# Column-buffer parquet writer
# ============================================================
class ColumnBufferParquetWriter:
    def __init__(self, path: str, schema: pa.Schema, flush_rows: int = 100000):
        self.path = path
        self.schema = schema
        self.flush_rows = int(flush_rows)
        self.writer: Optional[pq.ParquetWriter] = None
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.buffers: Dict[str, List[Any]] = {field.name: [] for field in schema}
        self.row_count = 0

    def append_columns(self, cols: Dict[str, List[Any]]) -> None:
        if not cols:
            return
        n = len(next(iter(cols.values()))) if cols else 0
        if n == 0:
            return

        for field in self.schema:
            self.buffers[field.name].extend(cols[field.name])

        self.row_count += n
        if self.row_count >= self.flush_rows:
            self.flush()

    def append_one(self, row: Dict[str, Any]) -> None:
        for field in self.schema:
            self.buffers[field.name].append(row.get(field.name))
        self.row_count += 1
        if self.row_count >= self.flush_rows:
            self.flush()

    def flush(self) -> None:
        if self.row_count == 0:
            return

        arrays = {}
        for field in self.schema:
            arrays[field.name] = pa.array(self.buffers[field.name], type=field.type)

        table = pa.Table.from_pydict(arrays, schema=self.schema)

        if self.writer is None:
            self.writer = pq.ParquetWriter(
                self.path,
                self.schema,
                compression="zstd",
            )
        self.writer.write_table(table)

        for k in self.buffers:
            self.buffers[k].clear()
        self.row_count = 0

        del arrays
        del table

    def close(self) -> None:
        self.flush()
        if self.writer is not None:
            self.writer.close()
            self.writer = None


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

    return df[required]


def clean_chunk_to_numpy(
    df: pd.DataFrame,
    allowed_rating_min: float,
    allowed_rating_max: float,
    drop_invalid_rows: bool,
) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
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

    if df.empty:
        return {
            "user_id": np.empty(0, dtype=np.int64),
            "movie_id": np.empty(0, dtype=np.int64),
            "rating": np.empty(0, dtype=np.float64),
            "timestamp": np.empty(0, dtype=np.int64),
        }, stats

    user_id = df["user_id"].to_numpy(dtype=np.int64, copy=True)
    movie_id = df["movie_id"].to_numpy(dtype=np.int64, copy=True)
    rating = df["rating"].to_numpy(dtype=np.float64, copy=True)
    timestamp = df["timestamp"].to_numpy(dtype=np.int64, copy=True)

    if drop_invalid_rows:
        mask = (
            (timestamp > 0) &
            (rating >= allowed_rating_min) &
            (rating <= allowed_rating_max)
        )
        stats["rows_dropped_invalid"] = int(len(timestamp) - int(mask.sum()))

        user_id = user_id[mask]
        movie_id = movie_id[mask]
        rating = rating[mask]
        timestamp = timestamp[mask]

    stats["rows_out"] = len(user_id)
    return {
        "user_id": user_id,
        "movie_id": movie_id,
        "rating": rating,
        "timestamp": timestamp,
    }, stats


# ============================================================
# Movie embedding loading
# ============================================================
@dataclass
class MovieEmbeddingStore:
    movie_to_index: Dict[int, int]
    embeddings: np.ndarray
    embedding_dim: int

    def get(self, movie_id: int) -> Optional[np.ndarray]:
        idx = self.movie_to_index.get(int(movie_id))
        if idx is None:
            return None
        return self.embeddings[idx]


def load_movie_embeddings(parquet_path: str, batch_size: int = 50000) -> MovieEmbeddingStore:
    movie_ids: List[int] = []
    embeddings_list: List[np.ndarray] = []
    embedding_dim: Optional[int] = None

    with open_input_binary(parquet_path) as source:
        parquet_file = pq.ParquetFile(source)

        for batch in parquet_file.iter_batches(batch_size=batch_size, columns=["movieId", "embedding"]):
            table = pa.Table.from_batches([batch])
            movie_col = table.column("movieId").to_pylist()
            emb_col = table.column("embedding").to_pylist()

            for movie_id_raw, emb_raw in zip(movie_col, emb_col):
                if movie_id_raw is None or emb_raw is None:
                    continue
                try:
                    movie_id = int(movie_id_raw)
                    emb = np.asarray(emb_raw, dtype=np.float32)
                except Exception:
                    continue

                if emb.ndim != 1:
                    continue

                if embedding_dim is None:
                    embedding_dim = int(emb.shape[0])

                if emb.shape[0] != embedding_dim:
                    continue

                movie_ids.append(movie_id)
                embeddings_list.append(emb)

            del table
            del batch

    if embedding_dim is None or not embeddings_list:
        raise ValueError("No valid embeddings found in movie embedding parquet")

    embeddings = np.vstack(embeddings_list).astype(np.float32, copy=False)
    movie_to_index = {mid: idx for idx, mid in enumerate(movie_ids)}

    del embeddings_list
    gc.collect()

    return MovieEmbeddingStore(
        movie_to_index=movie_to_index,
        embeddings=embeddings,
        embedding_dim=int(embedding_dim),
    )


# ============================================================
# User embedding aggregation
# ============================================================
def weighted_average_embedding(
    movie_ids: np.ndarray,
    ratings: np.ndarray,
    time_weights: np.ndarray,
    movie_store: MovieEmbeddingStore,
    preference_anchor_rating: float,
    min_positive_weight_sum: float,
) -> Tuple[Optional[np.ndarray], int, int]:
    embedding_dim = movie_store.embedding_dim
    vec_sum = np.zeros(embedding_dim, dtype=np.float32)
    weight_sum = 0.0
    used_count = 0
    missing_count = 0

    for i in range(len(movie_ids)):
        idx = movie_store.movie_to_index.get(int(movie_ids[i]))
        if idx is None:
            missing_count += 1
            continue

        pref_weight = max(float(ratings[i]) - preference_anchor_rating, 0.0)
        final_weight = pref_weight * float(time_weights[i])

        if final_weight <= 0:
            continue

        vec_sum += final_weight * movie_store.embeddings[idx]
        weight_sum += final_weight
        used_count += 1

    if weight_sum <= min_positive_weight_sum:
        return None, used_count, missing_count

    return (vec_sum / weight_sum).astype(np.float32, copy=False), used_count, missing_count


# ============================================================
# Per-user processing
# ============================================================
@dataclass
class UserBuildStats:
    users_seen: int = 1
    users_kept: int = 0
    users_skipped_too_few: int = 0
    bootstrap_events: int = 0
    remaining_events: int = 0
    users_with_long_embedding: int = 0
    users_with_short_embedding: int = 0


def deduplicate_keep_last_sorted(
    user_id: np.ndarray,
    movie_id: np.ndarray,
    rating: np.ndarray,
    timestamp: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # 输入已经按 timestamp 排序时，反向 unique 保留最后一次
    seen = set()
    keep_rev_indices = []
    for i in range(len(movie_id) - 1, -1, -1):
        mid = int(movie_id[i])
        if mid not in seen:
            seen.add(mid)
            keep_rev_indices.append(i)

    keep_indices = np.array(keep_rev_indices[::-1], dtype=np.int64)
    return (
        user_id[keep_indices],
        movie_id[keep_indices],
        rating[keep_indices],
        timestamp[keep_indices],
    )


def utc_datetimes_from_seconds(ts: np.ndarray) -> List[datetime]:
    return [datetime.fromtimestamp(int(x), tz=timezone.utc) for x in ts]


def process_single_user_arrays(
    user_id_arr: np.ndarray,
    movie_id_arr: np.ndarray,
    rating_arr: np.ndarray,
    timestamp_arr: np.ndarray,
    min_interactions_per_user: int,
    bootstrap_ratio: float,
    min_bootstrap_interactions: int,
    min_remaining_interactions: int,
    deduplicate_user_movie: bool,
    positive_rating_threshold: float,
    recent_k: int,
    half_life_days: int,
    profile_version: str,
    movie_store: MovieEmbeddingStore,
    preference_anchor_rating: float,
    min_positive_weight_sum: float,
) -> Tuple[
    Optional[Dict[str, List[Any]]],
    Optional[Dict[str, List[Any]]],
    Optional[Dict[str, Any]],
    Optional[Dict[str, Any]],
    Dict[str, int]
]:
    stats = UserBuildStats().__dict__

    n = len(user_id_arr)
    if n == 0:
        stats["users_skipped_too_few"] = 1
        return None, None, None, None, stats

    # The original dataset has sorted user id
    # resort the timestamp per user
    order = np.argsort(timestamp_arr, kind="stable")
    user_id_arr = user_id_arr[order]
    movie_id_arr = movie_id_arr[order]
    rating_arr = rating_arr[order]
    timestamp_arr = timestamp_arr[order]

    if deduplicate_user_movie:
        user_id_arr, movie_id_arr, rating_arr, timestamp_arr = deduplicate_keep_last_sorted(
            user_id_arr, movie_id_arr, rating_arr, timestamp_arr
        )
        n = len(user_id_arr)

    if n < min_interactions_per_user:
        stats["users_skipped_too_few"] = 1
        return None, None, None, None, stats

    bootstrap_n = max(min_bootstrap_interactions, int(math.floor(n * bootstrap_ratio)))
    max_bootstrap_allowed = n - min_remaining_interactions
    bootstrap_n = min(bootstrap_n, max_bootstrap_allowed)

    if bootstrap_n < min_bootstrap_interactions or (n - bootstrap_n) < min_remaining_interactions:
        stats["users_skipped_too_few"] = 1
        return None, None, None, None, stats

    boot_slice = slice(0, bootstrap_n)
    remain_slice = slice(bootstrap_n, n)

    boot_user_id = user_id_arr[boot_slice]
    boot_movie_id = movie_id_arr[boot_slice]
    boot_rating = rating_arr[boot_slice]
    boot_timestamp = timestamp_arr[boot_slice]

    remain_user_id = user_id_arr[remain_slice]
    remain_movie_id = movie_id_arr[remain_slice]
    remain_rating = rating_arr[remain_slice]
    remain_timestamp = timestamp_arr[remain_slice]

    if len(boot_user_id) == 0 or len(remain_user_id) == 0:
        stats["users_skipped_too_few"] = 1
        return None, None, None, None, stats

    user_id = int(boot_user_id[0])

    event_order = np.arange(1, len(boot_user_id) + 1, dtype=np.int32)
    boot_event_time = utc_datetimes_from_seconds(boot_timestamp)

    boot_avg_rating = float(boot_rating.mean())
    is_positive = boot_rating >= positive_rating_threshold
    rating_centered = boot_rating - boot_avg_rating

    user_last_ts = int(boot_timestamp.max())
    diff_days = (user_last_ts - boot_timestamp).astype(np.float64) / 86400.0
    decay_lambda = np.log(2.0) / max(half_life_days, 1)
    time_weight = np.exp(-decay_lambda * diff_days).astype(np.float64, copy=False)

    future_event_order = np.arange(1, len(remain_user_id) + 1, dtype=np.int32)
    remain_event_time = utc_datetimes_from_seconds(remain_timestamp)

    long_emb, long_used, long_missing = weighted_average_embedding(
        movie_ids=boot_movie_id,
        ratings=boot_rating,
        time_weights=time_weight,
        movie_store=movie_store,
        preference_anchor_rating=preference_anchor_rating,
        min_positive_weight_sum=min_positive_weight_sum,
    )

    recent_start = max(0, len(boot_movie_id) - recent_k)
    short_emb, short_used, short_missing = weighted_average_embedding(
        movie_ids=boot_movie_id[recent_start:],
        ratings=boot_rating[recent_start:],
        time_weights=time_weight[recent_start:],
        movie_store=movie_store,
        preference_anchor_rating=preference_anchor_rating,
        min_positive_weight_sum=min_positive_weight_sum,
    )

    built_at = datetime.now(timezone.utc)

    base_user_events_cols = {
        "user_id": boot_user_id.tolist(),
        "movie_id": boot_movie_id.tolist(),
        "rating": boot_rating.astype(np.float64, copy=False).tolist(),
        "timestamp": boot_timestamp.tolist(),
        "event_time": boot_event_time,
        "event_order": event_order.tolist(),
        "is_positive": is_positive.tolist(),
        "rating_centered": rating_centered.astype(np.float64, copy=False).tolist(),
        "time_weight": time_weight.astype(np.float64, copy=False).tolist(),
    }

    remaining_user_events_cols = {
        "user_id": remain_user_id.tolist(),
        "movie_id": remain_movie_id.tolist(),
        "rating": remain_rating.astype(np.float64, copy=False).tolist(),
        "timestamp": remain_timestamp.tolist(),
        "event_time": remain_event_time,
        "future_event_order": future_event_order.tolist(),
    }

    std_rating = float(np.std(boot_rating, ddof=1)) if len(boot_rating) > 1 else 0.0

    base_user_row = {
        "user_id": user_id,
        "num_total_interactions": int(n),
        "num_bootstrap_interactions": int(len(boot_user_id)),
        "num_remaining_interactions": int(len(remain_user_id)),
        "first_timestamp": int(timestamp_arr.min()),
        "last_bootstrap_timestamp": int(boot_timestamp.max()),
        "last_total_timestamp": int(timestamp_arr.max()),
        "avg_rating_bootstrap": float(boot_avg_rating),
        "std_rating_bootstrap": std_rating,
        "activity_span_days": float((int(timestamp_arr.max()) - int(timestamp_arr.min())) / 86400.0),
        "profile_confidence": float(np.log1p(len(boot_user_id))),
        "built_at": built_at,
    }

    base_user_profile_row = {
        "user_id": user_id,
        "long_term_embedding": long_emb.tolist() if long_emb is not None else None,
        "short_term_embedding": short_emb.tolist() if short_emb is not None else None,
        "embedding_dim": int(movie_store.embedding_dim),
        "num_embedded_bootstrap_interactions": int(long_used),
        "num_missing_embedding_movies": int(long_missing),
        "rating_bias": float(boot_avg_rating),
        "activity_level": float(len(boot_user_id)),
        "profile_version": profile_version,
        "built_at": built_at,
    }

    stats["users_kept"] = 1
    stats["bootstrap_events"] = len(boot_user_id)
    stats["remaining_events"] = len(remain_user_id)
    if long_emb is not None:
        stats["users_with_long_embedding"] = 1
    if short_emb is not None:
        stats["users_with_short_embedding"] = 1

    return (
        base_user_events_cols,
        remaining_user_events_cols,
        base_user_row,
        base_user_profile_row,
        stats,
    )


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
# Helpers for chunk/user boundary streaming
# ============================================================
def concat_pending_with_chunk(
    pending: Optional[Dict[str, np.ndarray]],
    current: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    if pending is None or len(pending["user_id"]) == 0:
        return current
    if len(current["user_id"]) == 0:
        return pending

    return {
        "user_id": np.concatenate([pending["user_id"], current["user_id"]]),
        "movie_id": np.concatenate([pending["movie_id"], current["movie_id"]]),
        "rating": np.concatenate([pending["rating"], current["rating"]]),
        "timestamp": np.concatenate([pending["timestamp"], current["timestamp"]]),
    }


def split_finalized_and_pending_by_last_user(
    arrs: Dict[str, np.ndarray],
) -> Tuple[Optional[Dict[str, np.ndarray]], Optional[Dict[str, np.ndarray]]]:
    user_id = arrs["user_id"]
    if len(user_id) == 0:
        return None, None

    last_user = user_id[-1]
    cut = len(user_id)
    while cut > 0 and user_id[cut - 1] == last_user:
        cut -= 1

    finalized = None
    if cut > 0:
        finalized = {
            "user_id": arrs["user_id"][:cut],
            "movie_id": arrs["movie_id"][:cut],
            "rating": arrs["rating"][:cut],
            "timestamp": arrs["timestamp"][:cut],
        }

    pending = {
        "user_id": arrs["user_id"][cut:],
        "movie_id": arrs["movie_id"][cut:],
        "rating": arrs["rating"][cut:],
        "timestamp": arrs["timestamp"][cut:],
    }

    return finalized, pending


def iter_users_from_sorted_arrays(arrs: Dict[str, np.ndarray]):
    user_id = arrs["user_id"]
    if len(user_id) == 0:
        return

    # 由于数据已按 user_id 排序，用边界扫描替代 groupby
    boundaries = np.flatnonzero(user_id[1:] != user_id[:-1]) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(user_id)]))

    movie_id = arrs["movie_id"]
    rating = arrs["rating"]
    timestamp = arrs["timestamp"]

    for s, e in zip(starts, ends):
        yield (
            user_id[s:e],
            movie_id[s:e],
            rating[s:e],
            timestamp[s:e],
        )


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
    chunksize = int(deep_get(cfg, ["input", "chunksize"], 100000))

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

    base_dir = deep_get(cfg, ["output", "base_dir"], "s3://artifacts/bootstraper_v0_embedding")
    write_parquet = bool(deep_get(cfg, ["output", "write_parquet"], True))
    writer_flush_rows = int(deep_get(cfg, ["output", "writer_flush_rows"], 100000))

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

    local_output_dir = tempfile.mkdtemp(prefix="bootstrap_outputs_")

    base_user_events_target = os.path.join(base_dir, "base_user_events.parquet")
    base_users_target = os.path.join(base_dir, "base_users.parquet")
    base_user_profiles_target = os.path.join(base_dir, "base_user_profiles.parquet")
    remaining_user_events_target = os.path.join(base_dir, "remaining_user_events.parquet")

    base_user_events_path = os.path.join(local_output_dir, "base_user_events.parquet")
    base_users_path = os.path.join(local_output_dir, "base_users.parquet")
    base_user_profiles_path = os.path.join(local_output_dir, "base_user_profiles.parquet")
    remaining_user_events_path = os.path.join(local_output_dir, "remaining_user_events.parquet")

    print(f"[{job_name}] loading movie embeddings from {movie_embedding_path}")
    movie_store = load_movie_embeddings(movie_embedding_path)
    print(f"[{job_name}] loaded movie embeddings: {len(movie_store.movie_to_index)}, dim={movie_store.embedding_dim}")

    writers = {}
    if write_parquet:
        writers["base_user_events"] = ColumnBufferParquetWriter(
            base_user_events_path, BASE_USER_EVENTS_SCHEMA, flush_rows=writer_flush_rows
        )
        writers["base_users"] = ColumnBufferParquetWriter(
            base_users_path, BASE_USERS_SCHEMA, flush_rows=max(10000, writer_flush_rows // 10)
        )
        writers["base_user_profiles"] = ColumnBufferParquetWriter(
            base_user_profiles_path, BASE_USER_PROFILES_SCHEMA, flush_rows=max(10000, writer_flush_rows // 10)
        )
        writers["remaining_user_events"] = ColumnBufferParquetWriter(
            remaining_user_events_path, REMAINING_USER_EVENTS_SCHEMA, flush_rows=writer_flush_rows
        )

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

    pending_user_arrays: Optional[Dict[str, np.ndarray]] = None

    print(f"[{job_name}] start reading ratings: {ratings_path}")
    for chunk_idx, raw_chunk in enumerate(csv_chunks_from_input(ratings_path, chunksize), start=1):
        global_stats["chunks_seen"] += 1
        global_stats["rows_in_raw"] += len(raw_chunk)

        clean_arrs, clean_stats = clean_chunk_to_numpy(
            raw_chunk,
            allowed_rating_min=allowed_rating_min,
            allowed_rating_max=allowed_rating_max,
            drop_invalid_rows=drop_invalid_rows,
        )

        global_stats["rows_after_cleaning"] += clean_stats["rows_out"]
        global_stats["rows_dropped_null"] += clean_stats["rows_dropped_null"]
        global_stats["rows_dropped_invalid"] += clean_stats["rows_dropped_invalid"]

        del raw_chunk

        merged_arrs = concat_pending_with_chunk(pending_user_arrays, clean_arrs)
        pending_user_arrays = None
        del clean_arrs

        if len(merged_arrs["user_id"]) == 0:
            continue

        finalized_arrs, pending_user_arrays = split_finalized_and_pending_by_last_user(merged_arrs)
        del merged_arrs

        if finalized_arrs is not None:
            for user_id_arr, movie_id_arr, rating_arr, timestamp_arr in iter_users_from_sorted_arrays(finalized_arrs):
                (
                    base_user_events_cols,
                    remaining_user_events_cols,
                    base_user_row,
                    base_user_profile_row,
                    stats,
                ) = process_single_user_arrays(
                    user_id_arr=user_id_arr,
                    movie_id_arr=movie_id_arr,
                    rating_arr=rating_arr,
                    timestamp_arr=timestamp_arr,
                    min_interactions_per_user=min_interactions_per_user,
                    bootstrap_ratio=bootstrap_ratio,
                    min_bootstrap_interactions=min_bootstrap_interactions,
                    min_remaining_interactions=min_remaining_interactions,
                    deduplicate_user_movie=deduplicate_user_movie,
                    positive_rating_threshold=positive_rating_threshold,
                    recent_k=recent_k,
                    half_life_days=half_life_days,
                    profile_version=profile_version,
                    movie_store=movie_store,
                    preference_anchor_rating=preference_anchor_rating,
                    min_positive_weight_sum=min_positive_weight_sum,
                )

                for k, v in stats.items():
                    global_stats[k] += v

                if write_parquet:
                    if base_user_events_cols is not None:
                        writers["base_user_events"].append_columns(base_user_events_cols)
                    if remaining_user_events_cols is not None:
                        writers["remaining_user_events"].append_columns(remaining_user_events_cols)
                    if base_user_row is not None:
                        writers["base_users"].append_one(base_user_row)
                    if base_user_profile_row is not None:
                        writers["base_user_profiles"].append_one(base_user_profile_row)

                if global_stats["users_seen"] > 0 and global_stats["users_seen"] % log_every_users == 0:
                    print(
                        f"[{job_name}] users_seen={global_stats['users_seen']}, "
                        f"users_kept={global_stats['users_kept']}, "
                        f"users_with_long_embedding={global_stats['users_with_long_embedding']}, "
                        f"bootstrap_events={global_stats['bootstrap_events']}, "
                        f"remaining_events={global_stats['remaining_events']}"
                    )

            del finalized_arrs

        if chunk_idx % 10 == 0:
            print(
                f"[{job_name}] chunk={chunk_idx}, "
                f"rows_in_raw={global_stats['rows_in_raw']}, "
                f"rows_after_cleaning={global_stats['rows_after_cleaning']}"
            )
            gc.collect()
            print(f"[{job_name}] Chunk {chunk_idx} done, memory garbage collected.")

    if pending_user_arrays is not None and len(pending_user_arrays["user_id"]) > 0:
        (
            base_user_events_cols,
            remaining_user_events_cols,
            base_user_row,
            base_user_profile_row,
            stats,
        ) = process_single_user_arrays(
            user_id_arr=pending_user_arrays["user_id"],
            movie_id_arr=pending_user_arrays["movie_id"],
            rating_arr=pending_user_arrays["rating"],
            timestamp_arr=pending_user_arrays["timestamp"],
            min_interactions_per_user=min_interactions_per_user,
            bootstrap_ratio=bootstrap_ratio,
            min_bootstrap_interactions=min_bootstrap_interactions,
            min_remaining_interactions=min_remaining_interactions,
            deduplicate_user_movie=deduplicate_user_movie,
            positive_rating_threshold=positive_rating_threshold,
            recent_k=recent_k,
            half_life_days=half_life_days,
            profile_version=profile_version,
            movie_store=movie_store,
            preference_anchor_rating=preference_anchor_rating,
            min_positive_weight_sum=min_positive_weight_sum,
        )

        for k, v in stats.items():
            global_stats[k] += v

        if write_parquet:
            if base_user_events_cols is not None:
                writers["base_user_events"].append_columns(base_user_events_cols)
            if remaining_user_events_cols is not None:
                writers["remaining_user_events"].append_columns(remaining_user_events_cols)
            if base_user_row is not None:
                writers["base_users"].append_one(base_user_row)
            if base_user_profile_row is not None:
                writers["base_user_profiles"].append_one(base_user_profile_row)

    for writer in writers.values():
        writer.close()

    if write_parquet:
        uploaded_paths = {
            "base_user_events": upload_file_to_path(base_user_events_path, base_user_events_target),
            "base_users": upload_file_to_path(base_users_path, base_users_target),
            "base_user_profiles": upload_file_to_path(base_user_profiles_path, base_user_profiles_target),
            "remaining_user_events": upload_file_to_path(remaining_user_events_path, remaining_user_events_target),
        }
    else:
        uploaded_paths = {}

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
            "embedding_dim": movie_store.embedding_dim,
            "num_movie_embeddings": len(movie_store.movie_to_index),
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
            "base_dir": to_s3_uri(base_dir),
            "parquet_files": {
                "base_user_events": uploaded_paths.get("base_user_events", to_s3_uri(base_user_events_target)),
                "base_users": uploaded_paths.get("base_users", to_s3_uri(base_users_target)),
                "base_user_profiles": uploaded_paths.get("base_user_profiles", to_s3_uri(base_user_profiles_target)),
                "remaining_user_events": uploaded_paths.get("remaining_user_events", to_s3_uri(remaining_user_events_target)),
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

    manifest_payload = json.dumps(manifest, indent=2, ensure_ascii=False).encode("utf-8")
    manifest_uri = upload_bytes_to_path(manifest_payload, manifest_path)

    print(f"[{job_name}] done")
    print(f"[{job_name}] manifest uploaded: {manifest_uri}")
    print(json.dumps(global_stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()