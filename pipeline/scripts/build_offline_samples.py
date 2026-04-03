import os
import json
import argparse
import shutil
from datetime import datetime, timezone
from typing import Any, Dict, List

import duckdb
import yaml


UINT64_MAX = 18446744073709551615.0
DEFAULT_SIGMOID_K = 5.0
DEFAULT_SIGMOID_C = 0.5
DEFAULT_JITTER_AMPLITUDE = 0.1


# ============================================================
# Utils
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


def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def require_value(name: str, value: Any) -> None:
    if value is None:
        raise ValueError(f"Missing required config: {name}")
    if isinstance(value, str) and not value.strip():
        raise ValueError(f"Empty required config: {name}")


def is_s3_path(path: str) -> bool:
    return isinstance(path, str) and path.startswith("s3://")


def get_registry_json(registry_path: str) -> dict:
    if is_s3_path(registry_path):
        raise ValueError("S3 registry JSON is not supported by the pure-Python registry helper. Use a local metadata root.")
    if not os.path.exists(registry_path):
        return {"versions": [], "latest": None}
    with open(registry_path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except Exception:
            return {"versions": [], "latest": None}


def save_registry_json(registry_path: str, registry: dict) -> None:
    if is_s3_path(registry_path):
        raise ValueError("S3 registry JSON is not supported by the pure-Python registry helper. Use a local metadata root.")
    ensure_dir(os.path.dirname(registry_path))
    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)


def next_version_name(existing_versions: List[dict]) -> str:
    max_num = 0
    for item in existing_versions:
        version = item.get("version", "")
        if version.startswith("v") and version[1:].isdigit():
            max_num = max(max_num, int(version[1:]))
    return f"v{max_num + 1:04d}"


# ============================================================
# DuckDB / S3
# ============================================================
def sql_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def configure_duckdb_runtime(con: duckdb.DuckDBPyConnection, cfg: Dict[str, Any]) -> None:
    runtime_cfg = deep_get(cfg, ["runtime"], {}) or {}

    con.execute("INSTALL httpfs;")
    con.execute("LOAD httpfs;")

    try:
        con.execute("INSTALL aws;")
        con.execute("LOAD aws;")
    except Exception:
        pass

    threads = runtime_cfg.get("threads")
    if threads:
        con.execute(f"SET threads = {int(threads)};")

    temp_directory = runtime_cfg.get("temp_directory")
    if temp_directory:
        ensure_dir(temp_directory)
        con.execute(f"SET temp_directory = {sql_quote(temp_directory)};")

    memory_limit = runtime_cfg.get("memory_limit")
    if memory_limit:
        con.execute(f"SET memory_limit = {sql_quote(str(memory_limit))};")

    preserve_insertion_order = runtime_cfg.get("preserve_insertion_order")
    if preserve_insertion_order is not None:
        flag = "true" if bool(preserve_insertion_order) else "false"
        con.execute(f"SET preserve_insertion_order = {flag};")

    s3_cfg = deep_get(cfg, ["storage", "s3"], {}) or {}
    if not s3_cfg:
        return

    use_secret = bool(s3_cfg.get("use_secret", False))
    secret_name = s3_cfg.get("secret_name", "recsys_s3")

    endpoint = s3_cfg.get("endpoint")
    region = s3_cfg.get("region")
    url_style = s3_cfg.get("url_style")
    use_ssl = s3_cfg.get("use_ssl")
    key_id = s3_cfg.get("access_key_id") or os.getenv("AWS_ACCESS_KEY_ID")
    secret = s3_cfg.get("secret_access_key") or os.getenv("AWS_SECRET_ACCESS_KEY")
    session_token = s3_cfg.get("session_token") or os.getenv("AWS_SESSION_TOKEN")

    if use_secret:
        parts = [
            f"CREATE OR REPLACE SECRET {secret_name} (",
            "TYPE s3",
        ]
        if key_id:
            parts.append(f", KEY_ID {sql_quote(key_id)}")
        if secret:
            parts.append(f", SECRET {sql_quote(secret)}")
        if session_token:
            parts.append(f", SESSION_TOKEN {sql_quote(session_token)}")
        if region:
            parts.append(f", REGION {sql_quote(region)}")
        if endpoint:
            parts.append(f", ENDPOINT {sql_quote(endpoint)}")
        if url_style:
            parts.append(f", URL_STYLE {sql_quote(url_style)}")
        if use_ssl is not None:
            parts.append(f", USE_SSL {'true' if bool(use_ssl) else 'false'}")
        parts.append(")")
        con.execute("".join(parts))
    else:
        if region:
            con.execute(f"SET s3_region = {sql_quote(region)};")
        if endpoint:
            con.execute(f"SET s3_endpoint = {sql_quote(endpoint)};")
        if url_style:
            con.execute(f"SET s3_url_style = {sql_quote(url_style)};")
        if key_id:
            con.execute(f"SET s3_access_key_id = {sql_quote(key_id)};")
        if secret:
            con.execute(f"SET s3_secret_access_key = {sql_quote(secret)};")
        if session_token:
            con.execute(f"SET s3_session_token = {sql_quote(session_token)};")
        if use_ssl is not None:
            con.execute(f"SET s3_use_ssl = {'true' if bool(use_ssl) else 'false'};")


# ============================================================
# Resolve latest input from source registry
# ============================================================
def resolve_latest_input_source(source_root_dir: str) -> Dict[str, str]:
    require_value("source_root_dir", source_root_dir)

    registry_path = os.path.join(source_root_dir, "registry", "versions.json")
    registry = get_registry_json(registry_path)

    latest_version = registry.get("latest")
    if not latest_version:
        raise ValueError(f"No latest version found in source registry: {registry_path}")

    versions = registry.get("versions", [])
    latest_entry = None
    for item in versions:
        if item.get("version") == latest_version:
            latest_entry = item
            break

    if latest_entry is None:
        raise ValueError(
            f"Latest version '{latest_version}' not found in registry entries: {registry_path}"
        )

    data_parts_prefix = latest_entry.get("data_parts_prefix")
    if data_parts_prefix:
        return {
            "source_registry": registry_path,
            "source_version": latest_version,
            "input_source": data_parts_prefix,
            "input_kind": "parts_prefix",
        }

    data_parquet = latest_entry.get("data_parquet")
    if data_parquet:
        return {
            "source_registry": registry_path,
            "source_version": latest_version,
            "input_source": data_parquet,
            "input_kind": "data_parquet",
        }

    version_dir = latest_entry.get("version_dir")
    if version_dir:
        parts_dir = os.path.join(version_dir, "parts")
        if not is_s3_path(parts_dir) and os.path.isdir(parts_dir):
            return {
                "source_registry": registry_path,
                "source_version": latest_version,
                "input_source": parts_dir,
                "input_kind": "version_dir_fallback",
            }

        data_file = os.path.join(version_dir, "data.parquet")
        if not is_s3_path(data_file) and os.path.exists(data_file):
            return {
                "source_registry": registry_path,
                "source_version": latest_version,
                "input_source": data_file,
                "input_kind": "version_dir_fallback",
            }

    raise ValueError(
        f"Cannot resolve latest input dataset from source registry: {registry_path}, latest entry: {latest_entry}"
    )


# ============================================================
# Parquet read helpers
# ============================================================
def parquet_glob_from_input(input_source: str, input_kind: str) -> str:
    if input_kind == "parts_prefix":
        return f"{input_source.rstrip('/')}/*.parquet"

    if input_kind in {"data_parquet", "version_dir_fallback"}:
        if is_s3_path(input_source):
            if input_source.endswith(".parquet"):
                return input_source
            return f"{input_source.rstrip('/')}/*.parquet"
        if os.path.isdir(input_source):
            return os.path.join(input_source, "*.parquet")
        return input_source

    raise ValueError(f"Unsupported input_kind: {input_kind}")


def parquet_read_expr(input_source: str, input_kind: str, union_by_name: bool = True) -> str:
    glob_path = parquet_glob_from_input(input_source, input_kind)
    return f"read_parquet({sql_quote(glob_path)}, union_by_name={'true' if union_by_name else 'false'})"


# ============================================================
# SQL builders
# ============================================================
def build_split_index_query(
    source_expr: str,
    user_id_col: str,
    movie_id_col: str,
    timestamp_col: str,
    train_ratio: float,
    val_ratio: float,
) -> str:
    train_cutoff = train_ratio
    val_cutoff = train_ratio + val_ratio

    return f"""
    WITH light_source AS (
        SELECT
            {user_id_col} AS user_id,
            {movie_id_col} AS movie_id,
            {timestamp_col} AS event_timestamp
        FROM {source_expr}
    ),
    dedup_candidates AS (
        SELECT
            user_id,
            event_timestamp,
            movie_id,
            COUNT(*) OVER (
                PARTITION BY user_id, event_timestamp
            ) AS key_dup_cnt
        FROM light_source
    ),
    unique_keys AS (
        SELECT
            user_id,
            event_timestamp,
            movie_id
        FROM dedup_candidates
        WHERE key_dup_cnt = 1
    ),
    ranked AS (
        SELECT
            user_id,
            event_timestamp,
            ROW_NUMBER() OVER (
                PARTITION BY user_id
                ORDER BY event_timestamp ASC, movie_id ASC
            ) AS seq_order,
            COUNT(*) OVER (
                PARTITION BY user_id
            ) AS total_user_rows
        FROM unique_keys
    )
    SELECT
        user_id,
        event_timestamp,
        CASE
            WHEN total_user_rows = 1 THEN 'train'
            WHEN ((CAST(seq_order AS DOUBLE) - 0.5) / total_user_rows) < {train_cutoff} THEN 'train'
            WHEN ((CAST(seq_order AS DOUBLE) - 0.5) / total_user_rows) < {val_cutoff} THEN 'val'
            ELSE 'test'
        END AS dataset_split
    FROM ranked
    """


def build_full_rows_query(
    source_expr: str,
    rating_col: str,
    user_id_col: str,
    movie_id_col: str,
    timestamp_col: str,
    user_embedding_col: str,
    movie_embedding_col: str,
) -> str:
    return f"""
    WITH full_source AS (
        SELECT
            {user_id_col} AS user_id,
            {movie_id_col} AS movie_id,
            {timestamp_col} AS event_timestamp,
            {user_embedding_col} AS user_embedding,
            {movie_embedding_col} AS movie_embedding,
            CAST({rating_col} AS DOUBLE) AS rating_raw
        FROM {source_expr}
    ),
    full_dedup AS (
        SELECT
            user_id,
            movie_id,
            event_timestamp,
            user_embedding,
            movie_embedding,
            rating_raw,
            COUNT(*) OVER (
                PARTITION BY user_id, event_timestamp
            ) AS key_dup_cnt
        FROM full_source
    )
    SELECT
        user_id,
        movie_id,
        event_timestamp,
        user_embedding,
        movie_embedding,
        rating_raw
    FROM full_dedup
    WHERE key_dup_cnt = 1
    """


def build_labeled_joined_query(
    full_rows_query: str,
    sigmoid_k: float,
    sigmoid_c: float,
    jitter_amplitude: float,
) -> str:
    if jitter_amplitude <= 0 or jitter_amplitude >= 0.5:
        raise ValueError("jitter_amplitude must be in (0, 0.5). Recommended value is 0.1.")

    return f"""
    WITH joined_rows AS (
        SELECT
            f.user_id,
            f.movie_id,
            f.event_timestamp,
            f.user_embedding,
            f.movie_embedding,
            f.rating_raw,
            s.dataset_split
        FROM ({full_rows_query}) AS f
        INNER JOIN split_index AS s
          ON f.user_id = s.user_id
         AND f.event_timestamp = s.event_timestamp
    ),
    with_jitter AS (
        SELECT
            user_id,
            movie_id,
            event_timestamp,
            user_embedding,
            movie_embedding,
            rating_raw,
            dataset_split,
            LEAST(
                5.0,
                GREATEST(
                    0.5,
                    rating_raw + (
                        (
                            CAST(
                                md5_number_lower(
                                    CAST(user_id AS VARCHAR) || ':' || CAST(movie_id AS VARCHAR)
                                ) AS DOUBLE
                            ) / {UINT64_MAX}
                        ) * {2.0 * jitter_amplitude} - {jitter_amplitude}
                    )
                )
            ) AS rating_jittered
        FROM joined_rows
    )
    SELECT
        user_id,
        movie_id,
        event_timestamp,
        user_embedding,
        movie_embedding,
        rating_raw,
        rating_jittered,
        1.0 / (
            1.0 + EXP(
                -{sigmoid_k} * (((rating_jittered - 0.5) / 4.5) - {sigmoid_c})
            )
        ) AS label,
        dataset_split
    FROM with_jitter
    """


# ============================================================
# Split + write
# ============================================================
def build_split_dataset(
    con: duckdb.DuckDBPyConnection,
    input_source: str,
    input_kind: str,
    output_dir: str,
    split_strategy: str = "per_user_ratio",
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    keep_rating_columns: bool = True,
    sigmoid_k: float = DEFAULT_SIGMOID_K,
    sigmoid_c: float = DEFAULT_SIGMOID_C,
    jitter_amplitude: float = DEFAULT_JITTER_AMPLITUDE,
    rating_col: str = "rating",
    user_id_col: str = "user_id",
    movie_id_col: str = "movie_id",
    timestamp_col: str = "timestamp",
    user_embedding_col: str = "user_embedding",
    movie_embedding_col: str = "movie_embedding",
    union_by_name: bool = True,
    per_thread_output: bool = False,
    write_partitioned_output: bool = False,
) -> Dict[str, int]:
    if not is_s3_path(output_dir):
        ensure_dir(output_dir)

    train_path = os.path.join(output_dir, "train.parquet")
    val_path = os.path.join(output_dir, "val.parquet")
    test_path = os.path.join(output_dir, "test.parquet")

    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-8:
        raise ValueError(
            f"train_ratio + val_ratio + test_ratio must equal 1.0, got {total_ratio}"
        )
    if split_strategy != "per_user_ratio":
        raise ValueError(f"Unsupported split strategy: {split_strategy}")

    source_expr = parquet_read_expr(input_source, input_kind, union_by_name=union_by_name)

    split_index_query = build_split_index_query(
        source_expr=source_expr,
        user_id_col=user_id_col,
        movie_id_col=movie_id_col,
        timestamp_col=timestamp_col,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )
    full_rows_query = build_full_rows_query(
        source_expr=source_expr,
        rating_col=rating_col,
        user_id_col=user_id_col,
        movie_id_col=movie_id_col,
        timestamp_col=timestamp_col,
        user_embedding_col=user_embedding_col,
        movie_embedding_col=movie_embedding_col,
    )
    labeled_joined_query = build_labeled_joined_query(
        full_rows_query=full_rows_query,
        sigmoid_k=sigmoid_k,
        sigmoid_c=sigmoid_c,
        jitter_amplitude=jitter_amplitude,
    )

    # Lightweight split index first: only user_id / timestamp after dropping ambiguous keys.
    con.execute(f"CREATE OR REPLACE TEMP TABLE split_index AS {split_index_query}")

    # Join back to full rows on (user_id, timestamp), then compute label.
    con.execute(f"CREATE OR REPLACE TEMP TABLE split_data AS {labeled_joined_query}")

    extra_cols = ", rating_raw, rating_jittered" if keep_rating_columns else ""
    per_thread_clause = ", PER_THREAD_OUTPUT TRUE" if per_thread_output else ""

    def copy_one(split_name: str, path: str) -> None:
        con.execute(f"""
            COPY (
                SELECT
                    user_id,
                    movie_id,
                    user_embedding,
                    movie_embedding
                    {extra_cols},
                    label
                FROM split_data
                WHERE dataset_split = {sql_quote(split_name)}
                ORDER BY user_id, event_timestamp, movie_id
            )
            TO {sql_quote(path)}
            (FORMAT PARQUET, COMPRESSION ZSTD{per_thread_clause})
        """)

    if write_partitioned_output:
        partitioned_root = output_dir.rstrip("/")
        con.execute(f"""
            COPY (
                SELECT
                    user_id,
                    movie_id,
                    user_embedding,
                    movie_embedding
                    {extra_cols},
                    label,
                    dataset_split
                FROM split_data
                ORDER BY dataset_split, user_id, event_timestamp, movie_id
            )
            TO {sql_quote(partitioned_root)}
            (FORMAT PARQUET, COMPRESSION ZSTD, PARTITION_BY (dataset_split){per_thread_clause})
        """)
    else:
        copy_one("train", train_path)
        copy_one("val", val_path)
        copy_one("test", test_path)

    stats_row = con.execute(
        """
        WITH raw_stats AS (
            SELECT
                COUNT(*) AS raw_rows,
                SUM(CASE WHEN key_dup_cnt > 1 THEN 1 ELSE 0 END) AS ambiguous_rows
            FROM (
                SELECT
                    COUNT(*) OVER (PARTITION BY user_id, event_timestamp) AS key_dup_cnt
                FROM (
                    SELECT
                        CAST(user_id AS VARCHAR) AS user_id,
                        CAST(event_timestamp AS VARCHAR) AS event_timestamp
                    FROM (
                        SELECT
                            {user_id_col} AS user_id,
                            {timestamp_col} AS event_timestamp
                        FROM {source_expr}
                    )
                )
            )
        )
        SELECT
            (SELECT raw_rows FROM raw_stats) AS raw_rows,
            (SELECT ambiguous_rows FROM raw_stats) AS ambiguous_rows,
            (SELECT COUNT(*) FROM split_index) AS split_index_rows,
            SUM(CASE WHEN dataset_split = 'train' THEN 1 ELSE 0 END) AS train_rows,
            SUM(CASE WHEN dataset_split = 'val' THEN 1 ELSE 0 END) AS val_rows,
            SUM(CASE WHEN dataset_split = 'test' THEN 1 ELSE 0 END) AS test_rows,
            COUNT(*) AS total_rows
        FROM split_data
        """.format(
            user_id_col=user_id_col,
            timestamp_col=timestamp_col,
            source_expr=source_expr,
        )
    ).fetchone()

    raw_rows = int(stats_row[0])
    ambiguous_rows = int(stats_row[1])
    split_index_rows = int(stats_row[2])
    train_rows = int(stats_row[3])
    val_rows = int(stats_row[4])
    test_rows = int(stats_row[5])
    total_rows = int(stats_row[6])

    return {
        "raw_rows": raw_rows,
        "ambiguous_rows_dropped": ambiguous_rows,
        "split_index_rows": split_index_rows,
        "train_rows": train_rows,
        "val_rows": val_rows,
        "test_rows": test_rows,
        "total_rows": total_rows,
    }


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)

    job_name = deep_get(cfg, ["job", "name"], "split_versioned_dataset_with_label_transform")
    created_at = datetime.now(timezone.utc).isoformat()

    dataset_type = deep_get(cfg, ["dataset", "type"], "offline")
    if dataset_type not in {"offline", "online"}:
        raise ValueError("dataset.type must be either 'offline' or 'online'")

    input_override_source = deep_get(cfg, ["input", "source"])
    input_override_kind = deep_get(cfg, ["input", "source_kind"])

    if input_override_source:
        require_value("input.source_kind", input_override_kind)
        input_info = {
            "source_registry": None,
            "source_version": None,
            "input_source": input_override_source,
            "input_kind": input_override_kind,
        }
        source_root_dir = None
    else:
        if dataset_type == "offline":
            source_root_dir = deep_get(cfg, ["input", "offline_root_dir"])
        else:
            source_root_dir = deep_get(cfg, ["input", "online_root_dir"])
        require_value(f"input.{dataset_type}_root_dir", source_root_dir)
        input_info = resolve_latest_input_source(source_root_dir)

    input_source = input_info["input_source"]
    input_kind = input_info["input_kind"]
    source_registry = input_info.get("source_registry")
    source_version = input_info.get("source_version")

    output_root = deep_get(cfg, ["output", "root_dir"], "/warehouse/dataset/versioned_dataset")
    local_metadata_root = deep_get(cfg, ["output", "local_metadata_root"])

    split_strategy = deep_get(cfg, ["split", "strategy"], "per_user_ratio")
    train_ratio = float(deep_get(cfg, ["split", "train_ratio"], 0.7))
    val_ratio = float(deep_get(cfg, ["split", "val_ratio"], 0.1))
    test_ratio = float(deep_get(cfg, ["split", "test_ratio"], 0.2))

    rating_col = deep_get(cfg, ["columns", "rating"], "rating")
    user_id_col = deep_get(cfg, ["columns", "user_id"], "user_id")
    movie_id_col = deep_get(cfg, ["columns", "movie_id"], "movie_id")
    timestamp_col = deep_get(cfg, ["columns", "timestamp"], "timestamp")
    user_embedding_col = deep_get(cfg, ["columns", "user_embedding"], "user_embedding")
    movie_embedding_col = deep_get(cfg, ["columns", "movie_embedding"], "movie_embedding")

    sigmoid_k = float(deep_get(cfg, ["label_transform", "sigmoid_k"], DEFAULT_SIGMOID_K))
    sigmoid_c = float(deep_get(cfg, ["label_transform", "sigmoid_c"], DEFAULT_SIGMOID_C))
    jitter_amplitude = float(deep_get(cfg, ["label_transform", "jitter_amplitude"], DEFAULT_JITTER_AMPLITUDE))
    keep_rating_columns = bool(deep_get(cfg, ["label_transform", "keep_rating_columns"], True))

    union_by_name = bool(deep_get(cfg, ["read", "union_by_name"], True))
    per_thread_output = bool(deep_get(cfg, ["write", "per_thread_output"], False))
    write_partitioned_output = bool(deep_get(cfg, ["write", "write_partitioned_output"], False))

    if is_s3_path(output_root) and not local_metadata_root:
        raise ValueError(
            "When output.root_dir is on S3, output.local_metadata_root must be a local path for registry/manifest bookkeeping."
        )

    dataset_root = os.path.join(output_root, dataset_type)
    versions_dir = os.path.join(dataset_root, "versions")
    latest_dir = os.path.join(dataset_root, "latest")

    metadata_dataset_root = dataset_root if not local_metadata_root else os.path.join(local_metadata_root, dataset_type)
    registry_path = os.path.join(metadata_dataset_root, "registry", "versions.json")

    if not is_s3_path(versions_dir):
        ensure_dir(versions_dir)
        ensure_dir(latest_dir)
    ensure_dir(os.path.dirname(registry_path))

    registry = get_registry_json(registry_path)
    version_name = next_version_name(registry.get("versions", []))
    version_dir = os.path.join(versions_dir, version_name)

    duckdb_path = deep_get(cfg, ["runtime", "duckdb_path"], ":memory:")
    if duckdb_path != ":memory:":
        ensure_dir(os.path.dirname(duckdb_path))

    con = duckdb.connect(database=duckdb_path)
    try:
        configure_duckdb_runtime(con, cfg)

        stats = build_split_dataset(
            con=con,
            input_source=input_source,
            input_kind=input_kind,
            output_dir=version_dir,
            split_strategy=split_strategy,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            keep_rating_columns=keep_rating_columns,
            sigmoid_k=sigmoid_k,
            sigmoid_c=sigmoid_c,
            jitter_amplitude=jitter_amplitude,
            rating_col=rating_col,
            user_id_col=user_id_col,
            movie_id_col=movie_id_col,
            timestamp_col=timestamp_col,
            user_embedding_col=user_embedding_col,
            movie_embedding_col=movie_embedding_col,
            union_by_name=union_by_name,
            per_thread_output=per_thread_output,
            write_partitioned_output=write_partitioned_output,
        )
    finally:
        con.close()

    if not write_partitioned_output:
        latest_train = os.path.join(latest_dir, "train.parquet")
        latest_val = os.path.join(latest_dir, "val.parquet")
        latest_test = os.path.join(latest_dir, "test.parquet")

        if not any(is_s3_path(p) for p in [version_dir, latest_dir]):
            ensure_dir(latest_dir)
            for src, dst in [
                (os.path.join(version_dir, "train.parquet"), latest_train),
                (os.path.join(version_dir, "val.parquet"), latest_val),
                (os.path.join(version_dir, "test.parquet"), latest_test),
            ]:
                if os.path.exists(dst):
                    os.remove(dst)
                shutil.copyfile(src, dst)

    manifest = {
        "job_name": job_name,
        "created_at": created_at,
        "dataset_type": dataset_type,
        "version": version_name,
        "input": {
            "source_root_dir": source_root_dir,
            "source_registry": source_registry,
            "source_version": source_version,
            "input_source": input_source,
            "input_kind": input_kind,
        },
        "split": {
            "strategy": split_strategy,
            "lightweight_split_key": [user_id_col, timestamp_col],
            "ambiguous_key_policy": "drop rows where (user_id, timestamp) is not unique before split and before join-back",
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
        },
        "label_transform": {
            "deterministic_jitter_key": [user_id_col, movie_id_col],
            "jitter_method": "md5_number_lower(user_id || ':' || movie_id)",
            "jitter_amplitude": jitter_amplitude,
            "rating_clip_range": [0.5, 5.0],
            "normalize_formula": "(rating_jittered - 0.5) / 4.5",
            "sigmoid_k": sigmoid_k,
            "sigmoid_c": sigmoid_c,
            "transform_stage": "after split-index join-back",
        },
        "output": {
            "root_dir": output_root,
            "dataset_root": dataset_root,
            "version_dir": version_dir,
            "latest_dir": latest_dir,
            "partitioned_output": write_partitioned_output,
            "train_parquet": None if write_partitioned_output else os.path.join(version_dir, "train.parquet"),
            "val_parquet": None if write_partitioned_output else os.path.join(version_dir, "val.parquet"),
            "test_parquet": None if write_partitioned_output else os.path.join(version_dir, "test.parquet"),
            "local_metadata_root": local_metadata_root,
        },
        "schema_note": {
            "index_fields": ["user_id", "movie_id"],
            "auxiliary_fields": ["rating_raw", "rating_jittered"] if keep_rating_columns else [],
            "training_core_fields": ["user_embedding", "movie_embedding", "label"],
        },
        "stats": stats,
    }

    version_metadata_dir = os.path.join(metadata_dataset_root, "versions", version_name)
    latest_metadata_dir = os.path.join(metadata_dataset_root, "latest")
    ensure_dir(version_metadata_dir)
    ensure_dir(latest_metadata_dir)

    manifest_path = os.path.join(version_metadata_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    latest_manifest_path = os.path.join(latest_metadata_dir, "manifest.json")
    with open(latest_manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    registry_entry = {
        "version": version_name,
        "created_at": created_at,
        "dataset_type": dataset_type,
        "source_version": source_version,
        "data_parts_prefix": None if write_partitioned_output else None,  # Not using parts prefix in this implementation
        "manifest": manifest_path,
        "row_count": stats["total_rows"],
    }

    registry["versions"] = registry.get("versions", [])
    registry["versions"].append(registry_entry)
    registry["latest"] = version_name
    save_registry_json(registry_path, registry)

    print(f"[{job_name}] done")
    print(json.dumps(manifest["stats"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
