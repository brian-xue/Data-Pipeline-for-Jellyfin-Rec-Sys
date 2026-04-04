import os
import json
import shutil
import argparse
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Optional

import re

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import s3fs
import yaml

from pyiceberg.catalog import load_catalog
from pyiceberg.schema import Schema
from pyiceberg.types import (
    NestedField,
    LongType,
    DoubleType,
    TimestampType,
    TimestamptzType,
    IntegerType,
    ListType,
    FloatType,
    StringType,
)


# ============================================================
# Utils
# ============================================================
DEFAULT_MINIO_ENDPOINT = "http://minio:9000"


def resolve_config_path(path: str) -> str:
    if os.path.isabs(path) and os.path.exists(path):
        return path

    candidates = [
        path,
        os.path.join(os.getcwd(), path),
        os.path.join(os.getcwd(), "scripts", path),
        os.path.join(os.path.dirname(__file__), path),
    ]

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(
        f"Config file not found: {path}. Tried: {candidates}"
    )


def load_yaml_config(path: str) -> Dict[str, Any]:
    config_path = resolve_config_path(path)
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def deep_get(d: Dict[str, Any], keys: List[str], default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def reset_dir(path: str) -> None:
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def bucket_mapping() -> Dict[str, str]:
    return {
        "raw": os.getenv("MINIO_RAW_BUCKET", "raw"),
        "cleaned": os.getenv("MINIO_CLEANED_BUCKET", "cleaned"),
        "embedding": os.getenv("MINIO_EMBEDDING_BUCKET", "embedding"),
        "artifacts": os.getenv("MINIO_ARTIFACTS_BUCKET", "artifacts"),
        "warehouse": os.getenv("MINIO_WAREHOUSE_BUCKET", "warehouse"),
    }


def s3_storage_options() -> Dict[str, Any]:
    endpoint = os.getenv("MINIO_ENDPOINT", DEFAULT_MINIO_ENDPOINT)
    key = (
        os.getenv("MINIO_ACCESS_KEY")
        or os.getenv("AWS_ACCESS_KEY_ID")
        or os.getenv("MINIO_ROOT_USER")
    )
    secret = (
        os.getenv("MINIO_SECRET_KEY")
        or os.getenv("AWS_SECRET_ACCESS_KEY")
        or os.getenv("MINIO_ROOT_PASSWORD")
    )

    options: Dict[str, Any] = {
        "client_kwargs": {"endpoint_url": endpoint},
    }
    if key:
        options["key"] = key
    if secret:
        options["secret"] = secret
    return options


def s3_filesystem() -> s3fs.S3FileSystem:
    opts = s3_storage_options()
    return s3fs.S3FileSystem(
        key=opts.get("key"),
        secret=opts.get("secret"),
        client_kwargs=opts.get("client_kwargs"),
    )


def local_data_path_to_s3_uri(path_like: str) -> Optional[str]:
    if path_like.startswith("s3://"):
        return path_like

    try:
        path_obj = Path(path_like).resolve()
    except Exception:
        path_obj = Path(path_like)

    parts = path_obj.parts
    if len(parts) < 3 or parts[0] != "/" or parts[1] != "data":
        return None

    top_level = parts[2]
    mapped_bucket = bucket_mapping().get(top_level)
    if mapped_bucket is None:
        return None

    key = PurePosixPath(*parts[3:]).as_posix()
    if key in {"", "."}:
        return f"s3://{mapped_bucket}"
    return f"s3://{mapped_bucket}/{key}"


def to_s3_uri(path_like: str) -> str:
    if path_like.startswith("s3://"):
        return path_like
    mapped = local_data_path_to_s3_uri(path_like)
    if mapped is None:
        raise ValueError(f"Path cannot be mapped to MinIO bucket: {path_like}")
    return mapped


def resolve_input_path(path_like: str) -> str:
    if path_like.startswith("s3://"):
        return path_like
    if os.path.exists(path_like):
        return path_like
    mapped = local_data_path_to_s3_uri(path_like)
    return mapped if mapped is not None else path_like


def materialize_input_to_local(path_like: str, local_path: str) -> str:
    resolved = resolve_input_path(path_like)
    if not resolved.startswith("s3://"):
        if not os.path.exists(resolved):
            raise FileNotFoundError(f"Input file not found: {resolved}")
        if os.path.isdir(resolved):
            return materialize_parquet_dataset_to_local_file(
                source=resolved,
                local_path=local_path,
            )
        return resolved

    fs = s3_filesystem()
    src = resolved.replace("s3://", "")
    if not fs.exists(src):
        if src.endswith("movie_embedding.parquet"):
            alt_src = src.replace("movie_embedding.parquet", "movie_embeddings.parquet")
            if fs.exists(alt_src):
                src = alt_src
            else:
                raise FileNotFoundError(
                    f"S3 input file not found: {src}. Also tried: {alt_src}"
                )
        else:
            raise FileNotFoundError(f"S3 input file not found: {src}")

    if is_s3_directory_like(fs, src):
        return materialize_parquet_dataset_to_local_file(
            source=resolved,
            local_path=local_path,
            filesystem=fs,
        )

    ensure_dir(os.path.dirname(local_path))
    with fs.open(src, "rb") as src_handle, open(local_path, "wb") as dst_handle:
        shutil.copyfileobj(src_handle, dst_handle)
    return local_path


def is_s3_directory_like(fs: s3fs.S3FileSystem, src: str) -> bool:
    try:
        if fs.isdir(src):
            return True
    except Exception:
        pass

    prefix = src.rstrip("/")
    try:
        children = fs.ls(prefix, detail=False)
        return len(children) > 0
    except Exception:
        return False


def materialize_parquet_dataset_to_local_file(
    source: str,
    local_path: str,
    filesystem: Optional[s3fs.S3FileSystem] = None,
) -> str:
    ensure_dir(os.path.dirname(local_path))

    if filesystem is None:
        dataset = ds.dataset(source, format="parquet")
    else:
        dataset = ds.dataset(source, format="parquet", filesystem=filesystem)

    scanner = dataset.scanner(batch_size=50000)
    writer = pq.ParquetWriter(local_path, dataset.schema, compression="zstd")
    try:
        for record_batch in scanner.to_batches():
            writer.write_table(pa.Table.from_batches([record_batch], schema=dataset.schema))
    finally:
        writer.close()

    return local_path


def upload_file_to_s3(local_file: str, target_path: str) -> str:
    target_uri = to_s3_uri(target_path)
    fs = s3_filesystem()
    fs_path = target_uri.replace("s3://", "")
    parent = str(PurePosixPath(fs_path).parent)
    if parent and parent != ".":
        fs.makedirs(parent, exist_ok=True)

    with open(local_file, "rb") as src_handle, fs.open(fs_path, "wb") as dst_handle:
        shutil.copyfileobj(src_handle, dst_handle)

    return target_uri


def upload_bytes_to_s3(payload: bytes, target_path: str) -> str:
    target_uri = to_s3_uri(target_path)
    fs = s3_filesystem()
    fs_path = target_uri.replace("s3://", "")
    parent = str(PurePosixPath(fs_path).parent)
    if parent and parent != ".":
        fs.makedirs(parent, exist_ok=True)

    with fs.open(fs_path, "wb") as handle:
        handle.write(payload)

    return target_uri


def ensure_output_dir(path_like: str) -> None:
    if path_like.startswith("s3://"):
        fs = s3_filesystem()
        fs.makedirs(path_like.replace("s3://", ""), exist_ok=True)
        return

    try:
        mapped_s3 = to_s3_uri(path_like)
        fs = s3_filesystem()
        fs.makedirs(mapped_s3.replace("s3://", ""), exist_ok=True)
        return
    except Exception:
        pass

    ensure_dir(path_like)


# ============================================================
# Preflight checks
# ============================================================
def require_value(name: str, value: Any) -> None:
    if value is None:
        raise ValueError(f"Missing required config: {name}")
    if isinstance(value, str) and not value.strip():
        raise ValueError(f"Empty required config: {name}")


def validate_local_dir_writable(path: str) -> None:
    ensure_dir(path)
    test_path = os.path.join(path, ".write_test")
    try:
        with open(test_path, "wb") as f:
            f.write(b"ok")
    finally:
        try:
            if os.path.exists(test_path):
                os.remove(test_path)
        except Exception:
            pass


def validate_local_dir_has_space(path: str, min_free_gb: float = 1.0) -> None:
    usage = shutil.disk_usage(path)
    free_gb = usage.free / (1024 ** 3)
    if free_gb < min_free_gb:
        raise RuntimeError(
            f"Insufficient free space under {path}: {free_gb:.2f} GiB available, "
            f"require at least {min_free_gb:.2f} GiB"
        )


def validate_s3_or_local_path_exists(path_like: str) -> None:
    resolved = resolve_input_path(path_like)

    if not resolved.startswith("s3://"):
        if not os.path.exists(resolved):
            raise FileNotFoundError(f"Input path not found: {resolved}")
        return

    fs = s3_filesystem()
    fs_path = resolved.replace("s3://", "")
    if fs.exists(fs_path):
        return

    if fs_path.endswith("movie_embedding.parquet"):
        alt_path = fs_path.replace("movie_embedding.parquet", "movie_embeddings.parquet")
        if fs.exists(alt_path):
            return

    raise FileNotFoundError(f"S3 input path not found: {resolved}")


def get_parquet_schema_names(path_like: str) -> List[str]:
    resolved = resolve_input_path(path_like)

    if resolved.startswith("s3://"):
        fs = s3_filesystem()
        src = resolved.replace("s3://", "")

        if not fs.exists(src) and src.endswith("movie_embedding.parquet"):
            alt_src = src.replace("movie_embedding.parquet", "movie_embeddings.parquet")
            if fs.exists(alt_src):
                src = alt_src
                resolved = f"s3://{src}"

        if is_s3_directory_like(fs, src):
            dataset = ds.dataset(resolved, format="parquet", filesystem=fs)
            return dataset.schema.names

        with fs.open(src, "rb") as f:
            parquet_file = pq.ParquetFile(f)
            return parquet_file.schema_arrow.names

    if os.path.isdir(resolved):
        dataset = ds.dataset(resolved, format="parquet")
        return dataset.schema.names

    parquet_file = pq.ParquetFile(resolved)
    return parquet_file.schema_arrow.names


def validate_required_columns(path_like: str, required_columns: List[str], label: str) -> None:
    names = set(get_parquet_schema_names(path_like))
    missing = [col for col in required_columns if col not in names]
    if missing:
        raise ValueError(
            f"{label} missing required columns: {missing}. "
            f"Existing columns: {sorted(names)}"
        )


def run_preflight_checks(cfg: Dict[str, Any]) -> None:
    # Required config
    require_value("input.base_user_profiles_path", deep_get(cfg, ["input", "base_user_profiles_path"]))
    require_value("input.remaining_user_events_path", deep_get(cfg, ["input", "remaining_user_events_path"]))
    require_value("input.movie_embedding_path", deep_get(cfg, ["input", "movie_embedding_path"]))

    require_value("iceberg.catalog_name", deep_get(cfg, ["iceberg", "catalog_name"]))
    require_value("iceberg.namespace", deep_get(cfg, ["iceberg", "namespace"]))
    require_value("iceberg.table_name", deep_get(cfg, ["iceberg", "table_name"]))
    require_value("iceberg.uri", deep_get(cfg, ["iceberg", "uri"]))
    require_value("iceberg.warehouse", deep_get(cfg, ["iceberg", "warehouse"]))

    # Weight validation
    long_weight = float(deep_get(cfg, ["user_embedding", "long_weight"], 0.4))
    short_weight = float(deep_get(cfg, ["user_embedding", "short_weight"], 0.6))
    if long_weight < 0 or short_weight < 0:
        raise ValueError("user_embedding.long_weight and short_weight must be non-negative")
    if long_weight == 0 and short_weight == 0:
        raise ValueError("user_embedding.long_weight and short_weight cannot both be 0")

    # Runtime tmp dir validation
    local_tmp_dir = deep_get(cfg, ["runtime", "local_tmp_dir"], "/mnt/block/tmp")
    validate_local_dir_writable(local_tmp_dir)
    validate_local_dir_has_space(local_tmp_dir, min_free_gb=1.0)

    # Input path validation
    base_user_profiles_path = deep_get(cfg, ["input", "base_user_profiles_path"])
    remaining_user_events_path = deep_get(cfg, ["input", "remaining_user_events_path"])
    movie_embedding_path = deep_get(cfg, ["input", "movie_embedding_path"])

    validate_s3_or_local_path_exists(base_user_profiles_path)
    validate_s3_or_local_path_exists(remaining_user_events_path)
    validate_s3_or_local_path_exists(movie_embedding_path)

    # Schema validation
    validate_required_columns(
        base_user_profiles_path,
        ["user_id", "long_term_embedding", "short_term_embedding", "profile_version"],
        "base_user_profiles",
    )
    validate_required_columns(
        remaining_user_events_path,
        ["user_id", "movie_id", "timestamp", "event_time", "future_event_order", "rating"],
        "remaining_user_events",
    )
    validate_required_columns(
        movie_embedding_path,
        ["movieId", "embedding"],
        "movie_embedding",
    )


# ===============================
# Versioning helpers
# ===============================
def get_registry_json(registry_path: str) -> dict:
    fs = s3_filesystem()
    fs_path = registry_path.replace("s3://", "")
    if not fs.exists(fs_path):
        return {"versions": [], "latest": None}
    with fs.open(fs_path, "rb") as f:
        try:
            return json.load(f)
        except Exception:
            return {"versions": [], "latest": None}


def save_registry_json(registry_path: str, registry: dict) -> str:
    payload = json.dumps(registry, indent=2, ensure_ascii=False).encode("utf-8")
    return upload_bytes_to_s3(payload, registry_path)


def parse_version_number(version: str) -> int | None:
    if not isinstance(version, str):
        return None
    m = re.match(r"^v(\d+)$", version.strip())
    if not m:
        return None
    return int(m.group(1))


def next_version_name(existing_versions: list, latest_version: str | None = None) -> str:
    max_num = 0

    latest_num = parse_version_number(latest_version or "")
    if latest_num is not None:
        max_num = max(max_num, latest_num)

    for v in existing_versions:
        current_num = parse_version_number(v.get("version", ""))
        if current_num is not None:
            max_num = max(max_num, current_num)

    return f"v{max_num + 1:04d}"


# ============================================================
# Fused user embedding schema
# ============================================================
FUSED_USER_SCHEMA = pa.schema([
    pa.field("user_id", pa.int64()),
    pa.field("user_embedding", pa.list_(pa.float32())),
    pa.field("profile_version", pa.string()),
])


# ============================================================
# Iceberg schema for final dataset
# ============================================================
def offline_positive_samples_iceberg_schema() -> Schema:
    return Schema(
        NestedField(1, "user_id", LongType(), required=False),
        NestedField(2, "movie_id", LongType(), required=False),
        NestedField(3, "timestamp", LongType(), required=False),
        NestedField(4, "event_time", TimestamptzType(), required=False),
        NestedField(5, "future_event_order", IntegerType(), required=False),
        NestedField(6, "label", DoubleType(), required=False),
        NestedField(
            7,
            "user_embedding",
            ListType(
                element_id=8,
                element_type=FloatType(),
                element_required=False,
            ),
            required=False,
        ),
        NestedField(
            9,
            "movie_embedding",
            ListType(
                element_id=10,
                element_type=DoubleType(),
                element_required=False,
            ),
            required=False,
        ),
        NestedField(11, "profile_version", StringType(), required=False),
    )


# ============================================================
# Incremental parquet writer
# ============================================================
class IncrementalParquetWriter:
    def __init__(self, path: str, schema: pa.Schema):
        self.path = path
        self.schema = schema
        self.writer: Optional[pq.ParquetWriter] = None
        ensure_dir(os.path.dirname(path))

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
# Embedding fusion
# ============================================================
def to_numpy_or_none(x) -> Optional[np.ndarray]:
    if x is None:
        return None
    try:
        arr = np.asarray(x, dtype=np.float32)
    except Exception:
        return None
    if arr.ndim != 1:
        return None
    return arr


def fuse_user_embedding(
    long_emb,
    short_emb,
    long_weight: float,
    short_weight: float,
) -> Optional[List[float]]:
    long_vec = to_numpy_or_none(long_emb)
    short_vec = to_numpy_or_none(short_emb)

    if long_vec is not None and short_vec is not None:
        if long_vec.shape != short_vec.shape:
            return None
        fused = long_weight * long_vec + short_weight * short_vec
        return fused.astype(np.float32).tolist()

    if short_vec is not None:
        return short_vec.astype(np.float32).tolist()

    if long_vec is not None:
        return long_vec.astype(np.float32).tolist()

    return None


# ============================================================
# Step 1: Stream base_user_profiles -> fused_user_embeddings.parquet
# ============================================================
def build_fused_user_embeddings(
    base_user_profiles_path: str,
    fused_user_embeddings_path: str,
    batch_size: int,
    long_weight: float,
    short_weight: float,
) -> Dict[str, int]:
    writer = IncrementalParquetWriter(fused_user_embeddings_path, FUSED_USER_SCHEMA)

    stats = {
        "profile_rows_seen": 0,
        "fused_user_rows_written": 0,
        "dropped_missing_embedding_rows": 0,
    }

    source_path = base_user_profiles_path
    if os.path.isdir(base_user_profiles_path):
        source_path = materialize_parquet_dataset_to_local_file(
            source=base_user_profiles_path,
            local_path=f"{fused_user_embeddings_path}.base_profiles.tmp.parquet",
        )

    parquet_file = pq.ParquetFile(source_path)

    try:
        for record_batch in parquet_file.iter_batches(
            batch_size=batch_size,
            columns=["user_id", "long_term_embedding", "short_term_embedding", "profile_version"],
        ):
            table = pa.Table.from_batches([record_batch])
            rows = table.to_pylist()

            out_rows: List[Dict[str, Any]] = []

            for row in rows:
                stats["profile_rows_seen"] += 1

                fused = fuse_user_embedding(
                    row.get("long_term_embedding"),
                    row.get("short_term_embedding"),
                    long_weight=long_weight,
                    short_weight=short_weight,
                )

                if fused is None:
                    stats["dropped_missing_embedding_rows"] += 1
                    continue

                out_rows.append({
                    "user_id": int(row["user_id"]),
                    "user_embedding": fused,
                    "profile_version": row.get("profile_version"),
                })

            writer.write_rows(out_rows)
            stats["fused_user_rows_written"] += len(out_rows)
    finally:
        writer.close()

    return stats


# ============================================================
# Step 2: Use DuckDB to build offline positive samples parquet
# ============================================================
def build_offline_positive_samples_parts_streaming(
    remaining_user_events_path: str,
    fused_user_embeddings_path: str,
    movie_embedding_path: str,
    local_tmp_dir: str,
    output_parts_prefix: str,
    scan_batch_rows: int = 2000,
) -> Dict[str, Any]:
    """
    Stream remaining_user_events in small Arrow batches.
    For each batch:
      - register the batch as a DuckDB temp view
      - join with local fused_user_embeddings + movie_embeddings parquet
      - write one small parquet part
      - upload to S3/MinIO
      - delete local part immediately
    """
    ensure_dir(local_tmp_dir)

    event_dataset = ds.dataset(remaining_user_events_path, format="parquet")
    scanner = event_dataset.scanner(
        batch_size=scan_batch_rows,
        columns=[
            "user_id",
            "movie_id",
            "timestamp",
            "event_time",
            "future_event_order",
            "rating",
        ],
    )

    total_rows = 0
    num_parts = 0
    part_uris: List[str] = []

    con = duckdb.connect(database=":memory:")
    try:
        con.execute("SET memory_limit='3GB'")

        for batch_idx, event_batch in enumerate(scanner.to_batches()):
            if event_batch.num_rows == 0:
                continue

            event_table = pa.Table.from_batches([event_batch])
            con.register("event_batch", event_table)

            joined = con.execute(
                f"""
                SELECT
                    r.user_id AS user_id,
                    r.movie_id AS movie_id,
                    r.timestamp AS timestamp,
                    r.event_time AS event_time,
                    CAST(r.future_event_order AS INTEGER) AS future_event_order,
                    CAST(r.rating AS DOUBLE) AS label,
                    u.user_embedding AS user_embedding,
                    m.embedding AS movie_embedding,
                    u.profile_version AS profile_version
                FROM event_batch r
                INNER JOIN read_parquet('{fused_user_embeddings_path}') u
                    ON r.user_id = u.user_id
                INNER JOIN read_parquet('{movie_embedding_path}') m
                    ON r.movie_id = m.movieId
                """
            ).fetch_arrow_table()

            con.unregister("event_batch")

            if joined.num_rows == 0:
                continue

            local_part_path = os.path.join(
                local_tmp_dir,
                f"offline_positive_samples_part_{batch_idx:05d}.parquet",
            )

            pq.write_table(
                joined,
                local_part_path,
                compression="zstd",
            )

            target_path = f"{output_parts_prefix}/part_{batch_idx:05d}.parquet"
            part_uri = upload_file_to_s3(local_part_path, target_path)

            os.remove(local_part_path)

            total_rows += joined.num_rows
            num_parts += 1
            part_uris.append(part_uri)

            print(
                f"[stream] uploaded part {batch_idx:05d}, "
                f"input_rows={event_table.num_rows}, output_rows={joined.num_rows}"
            )

        return {
            "offline_positive_samples_rows": total_rows,
            "num_parts": num_parts,
            "part_uris": part_uris,
        }
    finally:
        con.close()

# ============================================================
# Step 3: Append parquet to Iceberg
# ============================================================
def ensure_namespace(catalog, namespace: str) -> None:
    existing = catalog.list_namespaces()
    normalized = set()
    for ns in existing:
        if isinstance(ns, tuple):
            normalized.add(".".join(ns))
            if len(ns) == 1:
                normalized.add(ns[0])
        else:
            normalized.add(ns)

    if namespace not in normalized:
        catalog.create_namespace(namespace)


def ensure_table(catalog, identifier: str, schema: Schema) -> None:
    namespace, table_name = identifier.split(".", 1)
    ensure_namespace(catalog, namespace)

    existing_tables = catalog.list_tables(namespace)
    normalized = set()
    for t in existing_tables:
        if isinstance(t, tuple):
            normalized.add(".".join(t))
            if len(t) == 2:
                normalized.add(f"{t[0]}.{t[1]}")
                normalized.add(t[1])
        else:
            normalized.add(t)

    if identifier not in normalized and table_name not in normalized:
        catalog.create_table(identifier, schema=schema)


def append_parquet_to_iceberg(
    catalog_name: str,
    table_identifier: str,
    parquet_path: str,
    batch_rows: int,
    cfg: Dict[str, Any],
) -> None:
    iceberg_uri = deep_get(cfg, ["iceberg", "uri"])
    iceberg_warehouse = deep_get(cfg, ["iceberg", "warehouse"])

    if not iceberg_uri:
        raise ValueError("Missing config: iceberg.uri")
    if not iceberg_warehouse:
        raise ValueError("Missing config: iceberg.warehouse")

    access_key = (
    os.getenv("MINIO_ACCESS_KEY")
    or os.getenv("AWS_ACCESS_KEY_ID")
    or os.getenv("MINIO_ROOT_USER")
)
    secret_key = (
        os.getenv("MINIO_SECRET_KEY")
        or os.getenv("AWS_SECRET_ACCESS_KEY")
        or os.getenv("MINIO_ROOT_PASSWORD")
    )

    catalog = load_catalog(
        catalog_name,
        uri=iceberg_uri,
        warehouse=iceberg_warehouse,
        **{
        "s3.endpoint": "http://minio:9000",
        "s3.path-style-access": "true",
        "s3.access-key-id": access_key,
        "s3.secret-access-key": secret_key,
        "s3.region": "eu-central-1",
    },
    )

    ensure_table(
        catalog=catalog,
        identifier=table_identifier,
        schema=offline_positive_samples_iceberg_schema(),
    )

    table = catalog.load_table(table_identifier)
    parquet_file = pq.ParquetFile(parquet_path)

    for batch in parquet_file.iter_batches(batch_size=batch_rows):
        arrow_table = pa.Table.from_batches([batch])
        table.append(arrow_table)


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)

    print("[preflight] validating config, storage, and parquet schema")
    run_preflight_checks(cfg)
    print("[preflight] ok")

    job_name = deep_get(cfg, ["job", "name"], "build_offline_positive_samples")
    created_at = datetime.now(timezone.utc).isoformat()

    base_user_profiles_path = deep_get(cfg, ["input", "base_user_profiles_path"])
    remaining_user_events_path = deep_get(cfg, ["input", "remaining_user_events_path"])
    movie_embedding_path = deep_get(cfg, ["input", "movie_embedding_path"])

    long_weight = float(deep_get(cfg, ["user_embedding", "long_weight"], 0.4))
    short_weight = float(deep_get(cfg, ["user_embedding", "short_weight"], 0.6))

    profile_batch_size = int(deep_get(cfg, ["runtime", "profile_batch_size"], 5000))
    iceberg_append_batch_rows = int(deep_get(cfg, ["runtime", "iceberg_append_batch_rows"], 50000))
    local_tmp_dir = deep_get(cfg, ["runtime", "local_tmp_dir"], "/mnt/block/tmp")
    iceberg_enabled = bool(deep_get(cfg, ["iceberg", "enabled"], False))

    dataset_root = deep_get(
        cfg,
        ["output", "warehouse_dir"],
        "s3://warehouse/datasets/offline-positive-samples",
    )
    write_parquet = bool(deep_get(cfg, ["output", "write_parquet"], True))

    latest_dir = f"{dataset_root}/latest"
    versions_dir = f"{dataset_root}/versions"
    registry_path = f"{dataset_root}/registry/version.json"

    registry = get_registry_json(registry_path)
    version_name = next_version_name(
        registry.get("versions", []),
        registry.get("latest"),
    )
    version_dir = f"{versions_dir}/{version_name}"
    parts_prefix = f"{version_dir}/parts"

    latest_data_path = f"{latest_dir}/data.parquet"
    latest_manifest_path = f"{latest_dir}/manifest.json"
    version_manifest_path = f"{version_dir}/manifest.json"

    ensure_output_dir(version_dir)
    ensure_output_dir(parts_prefix)
    ensure_output_dir(latest_dir)

    print(
        f"[{job_name}] resolved version: latest={registry.get('latest')} -> next={version_name}, "
        f"output_dir={version_dir}"
    )

    catalog_name = deep_get(cfg, ["iceberg", "catalog_name"], "default")
    namespace = deep_get(cfg, ["iceberg", "namespace"], "recsys")
    table_name = deep_get(cfg, ["iceberg", "table_name"], "offline_positive_samples")
    table_identifier = f"{namespace}.{table_name}"

    os.makedirs(local_tmp_dir, exist_ok=True)
    runtime_dir = tempfile.mkdtemp(prefix="offline_samples_", dir=local_tmp_dir)

    try:
        local_inputs_dir = os.path.join(runtime_dir, "inputs")
        local_staging_dir = os.path.join(runtime_dir, "staging")
        reset_dir(local_staging_dir)

        local_base_user_profiles_path = materialize_input_to_local(
            base_user_profiles_path,
            os.path.join(local_inputs_dir, "base_user_profiles.parquet"),
        )
        local_remaining_user_events_path = materialize_input_to_local(
            remaining_user_events_path,
            os.path.join(local_inputs_dir, "remaining_user_events.parquet"),
        )
        local_movie_embedding_path = materialize_input_to_local(
            movie_embedding_path,
            os.path.join(local_inputs_dir, "movie_embeddings.parquet"),
        )

        fused_user_embeddings_path = os.path.join(local_staging_dir, "fused_user_embeddings.parquet")
        # offline_positive_samples_path = os.path.join(local_staging_dir, "offline_positive_samples.parquet")

        print(f"[{job_name}] step1: build fused user embeddings")
        fuse_stats = build_fused_user_embeddings(
            base_user_profiles_path=local_base_user_profiles_path,
            fused_user_embeddings_path=fused_user_embeddings_path,
            batch_size=profile_batch_size,
            long_weight=long_weight,
            short_weight=short_weight,
        )

        stream_batch_rows = int(deep_get(cfg, ["runtime", "profile_batch_size"], 2000))

        print(f"[{job_name}] step2: build offline positive samples parquet parts")
        sample_stats = build_offline_positive_samples_parts_streaming(
            remaining_user_events_path=local_remaining_user_events_path,
            fused_user_embeddings_path=fused_user_embeddings_path,
            movie_embedding_path=local_movie_embedding_path,
            local_tmp_dir=local_staging_dir,
            output_parts_prefix=parts_prefix,
            scan_batch_rows=stream_batch_rows,
        )

        if iceberg_enabled:
            print(f"[{job_name}] step3: append to Iceberg {table_identifier}")
            try:
                append_parquet_to_iceberg(
                    catalog_name=catalog_name,
                    table_identifier=table_identifier,
                    parquet_path=offline_positive_samples_path,
                    batch_rows=iceberg_append_batch_rows,
                    cfg=cfg,
                )
            except Exception as e:
                iceberg_error = str(e)
                print(f"[{job_name}] error appending to Iceberg: {iceberg_error}")
                iceberg_enabled = False
        else:
                print(f"[{job_name}] iceberg append skipped (disabled in config)")

        manifest = {
            "job_name": job_name,
            "created_at": created_at,
            "version": version_name,
            "input": {
                "base_user_profiles_path": resolve_input_path(base_user_profiles_path),
                "remaining_user_events_path": resolve_input_path(remaining_user_events_path),
                "movie_embedding_path": resolve_input_path(movie_embedding_path),
            },
            "user_embedding": {
                "long_weight": long_weight,
                "short_weight": short_weight,
                "fallback_rule": "0.4*long+0.6*short; short_only_if_no_long; long_only_if_no_short; drop_if_both_missing",
            },
            "output": {
                "data_parts_prefix": parts_prefix,
                "num_parts": sample_stats["num_parts"],
                "version_dir": version_dir,
                "latest_dir": latest_dir,
            },
            "iceberg": {
                "enabled" : iceberg_enabled,
                "catalog_name": catalog_name,
                "namespace": namespace,
                "table_name": table_name,
                "table_identifier": table_identifier,
            },
            "stats": {
                **fuse_stats,
                **sample_stats,
            },
        }

        manifest_payload = json.dumps(manifest, indent=2, ensure_ascii=False).encode("utf-8")
        version_manifest_uri = upload_bytes_to_s3(manifest_payload, version_manifest_path)
        latest_manifest_uri = upload_bytes_to_s3(manifest_payload, latest_manifest_path)

        row_count = int(sample_stats.get("offline_positive_samples_rows", 0))
        registry_entry = {
            "version": version_name,
            "created_at": created_at,
            "data_parts_prefix": parts_prefix,
            "manifest": version_manifest_uri,
            "row_count": row_count,
        }
        registry["versions"] = registry.get("versions", [])
        registry["versions"].append(registry_entry)
        registry["latest"] = version_name
        save_registry_json(registry_path, registry)

        print(f"[{job_name}] done")
        print(f"[{job_name}] uploaded parts prefix: {parts_prefix}")
        print(f"[{job_name}] uploaded manifest: {version_manifest_uri}")
        print(f"[{job_name}] uploaded latest/manifest: {latest_manifest_uri}")
        print(f"[{job_name}] updated registry: {registry_path}")
        print(json.dumps(manifest["stats"], indent=2, ensure_ascii=False))
    finally:
        shutil.rmtree(runtime_dir, ignore_errors=True)


if __name__ == "__main__":
    main()