import os
import json
import argparse
import shutil
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import tempfile

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



def parent_dir(path: str) -> str:
    if is_s3_path(path):
        return path.rsplit("/", 1)[0]
    return os.path.dirname(path)

def load_json_from_s3(path: str, con: duckdb.DuckDBPyConnection) -> dict:
    rows = con.execute(
        f"SELECT content FROM read_text({sql_quote(path)});"
    ).fetchall()

    if not rows:
        raise ValueError(f"No content read from {path}")

    text = rows[0][0]

    if text is None or not text.strip():
        raise ValueError(f"Empty JSON content from {path}")

    return json.loads(text)

def load_json_any(path: str, con: duckdb.DuckDBPyConnection | None = None) -> dict:
    """
    Support both:
      - local file path
      - s3://... path (requires DuckDB connection with S3 configured)

    For S3 JSON, read raw text via DuckDB read_text(), then parse with json.loads().
    This is more robust than read_json_auto() for registry/manifest files that are
    stored as a single JSON object.
    """
    print("[DEBUG] using connection id:", id(con))
    if not is_s3_path(path):
        if not os.path.exists(path):
            print(f"[WARN] Local JSON file not found: {path}")
            return {"versions": [], "latest": None}
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[ERROR] Failed to read local JSON from {path}: {e}")
            return {"versions": [], "latest": None}

    if con is None:
        raise ValueError(f"DuckDB connection is required for S3 JSON read: {path}")

    try:
        obj = load_json_from_s3(path, con)

        return obj

    except Exception as e:
        print(f"[ERROR] Failed to read JSON from {path}: {e}")
        return {"versions": [], "latest": None}


def save_json_any(path: str, obj: dict, con: duckdb.DuckDBPyConnection | None = None) -> None:
    """
    Support both:
      - local file path
      - s3://... path  (requires DuckDB connection with S3 configured)
    """
    if not is_s3_path(path):
        ensure_dir(os.path.dirname(path))
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)
        return

    if con is None:
        raise ValueError(f"DuckDB connection is required for S3 JSON write: {path}")

    # simplest robust way: stage to local temp file, then COPY to S3
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as tmp:
        json.dump(obj, tmp, indent=2, ensure_ascii=False)
        tmp_path = tmp.name

    try:
        con.execute(f"""
        COPY (SELECT read_blob({sql_quote(tmp_path)}) AS data)
        TO {sql_quote(path)}
        (FORMAT 'binary');
    """)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def get_registry_json(registry_path: str, con: duckdb.DuckDBPyConnection|None = None) -> dict:
    return load_json_any(registry_path, con=con)



def save_registry_json(registry_path: str, registry: dict, con: duckdb.DuckDBPyConnection|None = None) -> None:
    save_json_any(registry_path, registry, con=con)



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

    # ---- load extensions ----
    try:
        con.execute("LOAD httpfs;")
    except Exception:
        con.execute("INSTALL httpfs;")
        con.execute("LOAD httpfs;")

    # ---- basic runtime config ----
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

    # ---- SIMPLE S3 CONFIG ----
    s3_cfg = deep_get(cfg, ["storage", "s3"], {}) or {}
    if not s3_cfg:
        print("[WARN] No storage.s3 config found, S3 may not work.")
        return

    endpoint = s3_cfg.get("endpoint")
    region = s3_cfg.get("region", "us-east-1")
    access_key = s3_cfg.get("access_key_id") or os.getenv("AWS_ACCESS_KEY_ID")
    secret_key = s3_cfg.get("secret_access_key") or os.getenv("AWS_SECRET_ACCESS_KEY")
    use_ssl = bool(s3_cfg.get("use_ssl", False))
    url_style = s3_cfg.get("url_style", "path")

    if not endpoint:
        raise ValueError("storage.s3.endpoint is required")

    # remove http:// or https://
    endpoint_no_scheme = endpoint.replace("http://", "").replace("https://", "")

    print(f"[INFO] Configuring DuckDB S3: endpoint={endpoint_no_scheme}")

    con.execute(f"SET s3_region = {sql_quote(region)};")
    con.execute(f"SET s3_endpoint = {sql_quote(endpoint_no_scheme)};")
    con.execute(f"SET s3_use_ssl = {'true' if use_ssl else 'false'};")
    con.execute(f"SET s3_url_style = {sql_quote(url_style)};")

    if access_key:
        con.execute(f"SET s3_access_key_id = {sql_quote(access_key)};")
    if secret_key:
        con.execute(f"SET s3_secret_access_key = {sql_quote(secret_key)};")
    print("[DEBUG] setting s3_access_key_id =", access_key)

# ============================================================
# Resolve latest input from source registry
# ============================================================
def resolve_latest_input_source(source_root_dir: str, con: duckdb.DuckDBPyConnection|None = None) -> Dict[str, str]:
    require_value("source_root_dir", source_root_dir)

    registry_path = os.path.join(source_root_dir, "registry", "version.json")
    registry = get_registry_json(registry_path, con)
    # print("found the registry:", registry)
    latest_version = registry.get("latest_version") or registry.get("latest")
    if not latest_version:
        raise ValueError(f"No latest version found in source registry: {registry_path}")

    versions = registry.get("versions")

    if isinstance(versions, list) and versions:
        latest_entry = None
        for item in versions:
            if item.get("version") == latest_version:
                latest_entry = item
                break
    else:
        # fallback: treat the registry object itself as the latest entry
        if registry.get("version") == latest_version:
            latest_entry = registry
        else:
            latest_entry = None
    if latest_entry is None:
        raise ValueError(
            f"Cannot find latest version entry: {latest_version} in registry {registry_path}"
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

    version_dir = latest_entry.get("version_dir") or latest_entry.get("data_parts_prefix") or latest_entry.get("data_parquet")
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
    print(f"[DEBUG] parquet_read_expr: glob_path={glob_path}, union_by_name={union_by_name}")
    return (
        f"read_parquet({sql_quote(glob_path)})" #, union_by_name={'true' if union_by_name else 'false'})"
    )


# ============================================================
# Label transform
# ============================================================
def build_labeled_source_query(
    source_expr: str,
    rating_col: str,
    user_id_col: str,
    movie_id_col: str,
    timestamp_col: str,
    user_embedding_col: str,
    movie_embedding_col: str,
    sigmoid_k: float,
    sigmoid_c: float,
    jitter_amplitude: float,
) -> str:
    if jitter_amplitude <= 0 or jitter_amplitude >= 0.5:
        raise ValueError("jitter_amplitude must be in (0, 0.5). Recommended value is 0.1.")

    return f"""
    WITH base AS (
        SELECT
            {user_id_col} AS user_id,
            {movie_id_col} AS movie_id,
            {user_embedding_col} AS user_embedding,
            {movie_embedding_col} AS movie_embedding,
            CAST({rating_col} AS DOUBLE) AS rating_raw,
            {timestamp_col} AS event_timestamp,
            (
                (
                    CAST(
                        md5_number_lower(
                            CAST({user_id_col} AS VARCHAR) || ':' || CAST({movie_id_col} AS VARCHAR)
                        ) AS DOUBLE
                    ) / {UINT64_MAX}
                ) * {2.0 * jitter_amplitude}
            ) - {jitter_amplitude} AS rating_jitter
        FROM {source_expr}
    ),
    transformed AS (
        SELECT
            user_id,
            movie_id,
            user_embedding,
            movie_embedding,
            rating_raw,
            LEAST(5.0, GREATEST(0.5, rating_raw + rating_jitter)) AS rating_jittered,
            event_timestamp
        FROM base
    )
    SELECT
        user_id,
        movie_id,
        user_embedding,
        movie_embedding,
        rating_raw,
        rating_jittered,
        1.0 / (
            1.0 + EXP(
                -{sigmoid_k} * (((rating_jittered - 0.5) / 4.5) - {sigmoid_c})
            )
        ) AS label,
        event_timestamp
    FROM transformed
    """


# ============================================================
# Split + write
# ============================================================
# def build_split_dataset(
#     con: duckdb.DuckDBPyConnection,
#     input_source: str,
#     input_kind: str,
#     output_dir: str,
#     split_strategy: str = "per_user_ratio",
#     train_ratio: float = 0.7,
#     val_ratio: float = 0.1,
#     test_ratio: float = 0.2,
#     keep_rating_columns: bool = True,
#     sigmoid_k: float = DEFAULT_SIGMOID_K,
#     sigmoid_c: float = DEFAULT_SIGMOID_C,
#     jitter_amplitude: float = DEFAULT_JITTER_AMPLITUDE,
#     rating_col: str = "rating",
#     user_id_col: str = "user_id",
#     movie_id_col: str = "movie_id",
#     timestamp_col: str = "timestamp",
#     user_embedding_col: str = "user_embedding",
#     movie_embedding_col: str = "movie_embedding",
#     union_by_name: bool = False,
#     per_thread_output: bool = False,
#     write_partitioned_output: bool = False,
# ) -> Dict[str, int]:
#     """
#     Stable version:
#     1) build a lightweight labeled query without embeddings
#     2) do split on the lightweight query
#     3) join back to source to fetch embeddings
#     4) write train/val/test parquet
#     """

#     if not is_s3_path(output_dir):
#         ensure_dir(output_dir)

#     train_path = os.path.join(output_dir, "train.parquet")
#     val_path = os.path.join(output_dir, "val.parquet")
#     test_path = os.path.join(output_dir, "test.parquet")

#     total_ratio = train_ratio + val_ratio + test_ratio
#     if abs(total_ratio - 1.0) > 1e-8:
#         raise ValueError(
#             f"train_ratio + val_ratio + test_ratio must equal 1.0, got {total_ratio}"
#         )

#     if split_strategy != "per_user_ratio":
#         raise ValueError(f"Unsupported split strategy: {split_strategy}")

#     source_expr = parquet_read_expr(
#         input_source=input_source,
#         input_kind=input_kind,
#         union_by_name=union_by_name,
#     )

#     train_cutoff = train_ratio
#     val_cutoff = train_ratio + val_ratio
#     per_thread_clause = ", PER_THREAD_OUTPUT TRUE" if per_thread_output else ""
#     extra_cols = ", rating_raw, rating_jittered" if keep_rating_columns else ""

#     # ----------------------------------------------------------
#     # Detect input schema
#     # ----------------------------------------------------------
#     schema_rows = con.execute(f"DESCRIBE SELECT * FROM {source_expr}").fetchall()
#     input_columns = {row[0] for row in schema_rows}

#     if user_id_col not in input_columns:
#         raise ValueError(f"Missing required user id column: {user_id_col}")
#     if movie_id_col not in input_columns:
#         raise ValueError(f"Missing required movie id column: {movie_id_col}")
#     if timestamp_col not in input_columns:
#         raise ValueError(f"Missing required timestamp column: {timestamp_col}")
#     if user_embedding_col not in input_columns:
#         raise ValueError(f"Missing required user embedding column: {user_embedding_col}")
#     if movie_embedding_col not in input_columns:
#         raise ValueError(f"Missing required movie embedding column: {movie_embedding_col}")

#     # rating fallback:
#     # if rating not found, reuse existing label as rating_raw
#     if rating_col in input_columns:
#         rating_expr_base = f"CAST({rating_col} AS DOUBLE)"
#         source_rating_select = f"{rating_col} AS source_rating"
#     elif "label" in input_columns:
#         print(f"[WARN] Column '{rating_col}' not found, fallback to existing 'label' as rating_raw")
#         rating_expr_base = "CAST(label AS DOUBLE)"
#         source_rating_select = "label AS source_rating"
#     else:
#         raise ValueError(
#             f"Neither rating column '{rating_col}' nor fallback column 'label' exists in input schema: {sorted(input_columns)}"
#         )

#     # ----------------------------------------------------------
#     # Step 1: build split index from lightweight labeled query
#     # no embedding columns here
#     # ----------------------------------------------------------
#     con.execute(f"""
#         CREATE OR REPLACE TEMP TABLE split_index AS
#         WITH labeled_light AS (
#             WITH base AS (
#                 SELECT
#                     {user_id_col} AS user_id,
#                     {movie_id_col} AS movie_id,
#                     {timestamp_col} AS event_timestamp,
#                     {rating_expr_base} AS rating_raw,
#                     (
#                         (
#                             CAST(
#                                 md5_number_lower(
#                                     CAST({user_id_col} AS VARCHAR) || ':' || CAST({movie_id_col} AS VARCHAR)
#                                 ) AS DOUBLE
#                             ) / {UINT64_MAX}
#                         ) * {2.0 * jitter_amplitude}
#                     ) - {jitter_amplitude} AS rating_jitter
#                 FROM {source_expr}
#             ),
#             transformed AS (
#                 SELECT
#                     user_id,
#                     movie_id,
#                     event_timestamp,
#                     rating_raw,
#                     LEAST(5.0, GREATEST(0.5, rating_raw + rating_jitter)) AS rating_jittered
#                 FROM base
#             )
#             SELECT
#                 user_id,
#                 movie_id,
#                 event_timestamp,
#                 rating_raw,
#                 rating_jittered,
#                 1.0 / (
#                     1.0 + EXP(
#                         -{sigmoid_k} * (((rating_jittered - 0.5) / 4.5) - {sigmoid_c})
#                     )
#                 ) AS label
#             FROM transformed
#         ),
#         ranked AS (
#             SELECT
#                 user_id,
#                 movie_id,
#                 event_timestamp,
#                 rating_raw,
#                 rating_jittered,
#                 label,
#                 ROW_NUMBER() OVER (
#                     PARTITION BY user_id
#                     ORDER BY event_timestamp ASC, movie_id ASC
#                 ) AS seq_order,
#                 COUNT(*) OVER (
#                     PARTITION BY user_id
#                 ) AS total_user_rows
#             FROM labeled_light
#         )
#         SELECT
#             user_id,
#             movie_id,
#             event_timestamp,
#             rating_raw,
#             rating_jittered,
#             label,
#             seq_order,
#             total_user_rows,
#             CASE
#                 WHEN total_user_rows = 1 THEN 'train'
#                 WHEN ((CAST(seq_order AS DOUBLE) - 0.5) / total_user_rows) < {train_cutoff} THEN 'train'
#                 WHEN ((CAST(seq_order AS DOUBLE) - 0.5) / total_user_rows) < {val_cutoff} THEN 'val'
#                 ELSE 'test'
#             END AS dataset_split
#         FROM ranked
#     """)

#     # ----------------------------------------------------------
#     # Step 2: de-duplicate source rows for join
#     # ----------------------------------------------------------
#     con.execute(f"""
#         CREATE OR REPLACE TEMP TABLE source_features AS
#         SELECT
#             user_id,
#             movie_id,
#             event_timestamp,
#             user_embedding,
#             movie_embedding
#         FROM (
#             SELECT
#                 {user_id_col} AS user_id,
#                 {movie_id_col} AS movie_id,
#                 {timestamp_col} AS event_timestamp,
#                 {user_embedding_col} AS user_embedding,
#                 {movie_embedding_col} AS movie_embedding,
#                 ROW_NUMBER() OVER (
#                     PARTITION BY {user_id_col}, {movie_id_col}, {timestamp_col}
#                     ORDER BY {movie_id_col}
#                 ) AS rn
#             FROM {source_expr}
#         )
#         WHERE rn = 1
#     """)

#     # ----------------------------------------------------------
#     # Step 3: join split result back to embeddings
#     # ----------------------------------------------------------
#     con.execute("""
#         CREATE OR REPLACE TEMP TABLE split_data AS
#         SELECT
#             s.user_id,
#             s.movie_id,
#             f.user_embedding,
#             f.movie_embedding,
#             s.rating_raw,
#             s.rating_jittered,
#             s.label,
#             s.seq_order,
#             s.total_user_rows,
#             s.dataset_split
#         FROM split_index s
#         INNER JOIN source_features f
#             ON s.user_id = f.user_id
#            AND s.movie_id = f.movie_id
#            AND s.event_timestamp = f.event_timestamp
#     """)

#     # ----------------------------------------------------------
#     # Step 4: write output
#     # ----------------------------------------------------------
#     def copy_one(split_name: str, path: str) -> None:
#         con.execute(f"""
#             COPY (
#                 SELECT
#                     user_id,
#                     movie_id,
#                     user_embedding,
#                     movie_embedding
#                     {extra_cols},
#                     label
#                 FROM split_data
#                 WHERE dataset_split = {sql_quote(split_name)}
#                 ORDER BY user_id, seq_order
#             )
#             TO {sql_quote(path)}
#             (FORMAT PARQUET, COMPRESSION ZSTD{per_thread_clause})
#         """)

#     if write_partitioned_output:
#         partitioned_root = output_dir.rstrip("/")
#         con.execute(f"""
#             COPY (
#                 SELECT
#                     user_id,
#                     movie_id,
#                     user_embedding,
#                     movie_embedding
#                     {extra_cols},
#                     label,
#                     dataset_split
#                 FROM split_data
#                 ORDER BY dataset_split, user_id, seq_order
#             )
#             TO {sql_quote(partitioned_root)}
#             (FORMAT PARQUET, COMPRESSION ZSTD, PARTITION_BY (dataset_split){per_thread_clause})
#         """)
#     else:
#         copy_one("train", train_path)
#         copy_one("val", val_path)
#         copy_one("test", test_path)

#     # ----------------------------------------------------------
#     # Step 5: stats
#     # ----------------------------------------------------------
#     stats_row = con.execute("""
#         SELECT
#             SUM(CASE WHEN dataset_split = 'train' THEN 1 ELSE 0 END) AS train_rows,
#             SUM(CASE WHEN dataset_split = 'val' THEN 1 ELSE 0 END) AS val_rows,
#             SUM(CASE WHEN dataset_split = 'test' THEN 1 ELSE 0 END) AS test_rows,
#             COUNT(*) AS total_rows
#         FROM split_data
#     """).fetchone()

#     dropped_row = con.execute("""
#         SELECT
#             (SELECT COUNT(*) FROM split_index) - (SELECT COUNT(*) FROM split_data)
#     """).fetchone()

#     return {
#         "train_rows": int(stats_row[0]),
#         "val_rows": int(stats_row[1]),
#         "test_rows": int(stats_row[2]),
#         "total_rows": int(stats_row[3]),
#         "rows_dropped_on_join": int(dropped_row[0]),
#     }

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
    union_by_name: bool = False,
    per_thread_output: bool = False,
    write_partitioned_output: bool = False,
    num_buckets: int = 128,
) -> Dict[str, int]:
    if not is_s3_path(output_dir):
        ensure_dir(output_dir)

    if split_strategy != "per_user_ratio":
        raise ValueError(f"Unsupported split strategy: {split_strategy}")

    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-8:
        raise ValueError(
            f"train_ratio + val_ratio + test_ratio must equal 1.0, got {total_ratio}"
        )

    source_expr = parquet_read_expr(
        input_source=input_source,
        input_kind=input_kind,
        union_by_name=union_by_name,
    )

    schema_rows = con.execute(f"DESCRIBE SELECT * FROM {source_expr}").fetchall()
    input_columns = {row[0] for row in schema_rows}

    if rating_col in input_columns:
        rating_expr_base = f"CAST({rating_col} AS DOUBLE)"
    elif "label" in input_columns:
        print(f"[WARN] Column '{rating_col}' not found, fallback to existing 'label' as rating_raw")
        rating_expr_base = "CAST(label AS DOUBLE)"
    else:
        raise ValueError(
            f"Neither rating column '{rating_col}' nor fallback column 'label' exists in input schema: {sorted(input_columns)}"
        )

    train_cutoff = train_ratio
    val_cutoff = train_ratio + val_ratio
    per_thread_clause = ", PER_THREAD_OUTPUT TRUE" if per_thread_output else ""
    extra_cols = ", rating_raw, rating_jittered" if keep_rating_columns else ""

    train_root = os.path.join(output_dir, "train_parts")
    val_root = os.path.join(output_dir, "val_parts")
    test_root = os.path.join(output_dir, "test_parts")

    if not is_s3_path(train_root):
        ensure_dir(train_root)
        ensure_dir(val_root)
        ensure_dir(test_root)

    total_train = 0
    total_val = 0
    total_test = 0
    total_dropped = 0

    for bucket_id in range(num_buckets):
        print(f"[INFO] processing bucket {bucket_id}/{num_buckets - 1}")

        # 1) build lightweight split index for this bucket only
        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE split_index_bucket AS
            WITH labeled_light AS (
                WITH base AS (
                    SELECT
                        {user_id_col} AS user_id,
                        {movie_id_col} AS movie_id,
                        {timestamp_col} AS event_timestamp,
                        {rating_expr_base} AS rating_raw,
                        (
                            (
                                CAST(
                                    md5_number_lower(
                                        CAST({user_id_col} AS VARCHAR) || ':' || CAST({movie_id_col} AS VARCHAR)
                                    ) AS DOUBLE
                                ) / {UINT64_MAX}
                            ) * {2.0 * jitter_amplitude}
                        ) - {jitter_amplitude} AS rating_jitter
                    FROM {source_expr}
                    WHERE ABS(hash(CAST({user_id_col} AS VARCHAR))) % {num_buckets} = {bucket_id}
                ),
                transformed AS (
                    SELECT
                        user_id,
                        movie_id,
                        event_timestamp,
                        rating_raw,
                        LEAST(5.0, GREATEST(0.5, rating_raw + rating_jitter)) AS rating_jittered
                    FROM base
                )
                SELECT
                    user_id,
                    movie_id,
                    event_timestamp,
                    rating_raw,
                    rating_jittered,
                    1.0 / (
                        1.0 + EXP(
                            -{sigmoid_k} * (((rating_jittered - 0.5) / 4.5) - {sigmoid_c})
                        )
                    ) AS label
                FROM transformed
            ),
            ranked AS (
                SELECT
                    user_id,
                    movie_id,
                    event_timestamp,
                    rating_raw,
                    rating_jittered,
                    label,
                    ROW_NUMBER() OVER (
                        PARTITION BY user_id
                        ORDER BY event_timestamp ASC, movie_id ASC
                    ) AS seq_order,
                    COUNT(*) OVER (
                        PARTITION BY user_id
                    ) AS total_user_rows
                FROM labeled_light
            )
            SELECT
                user_id,
                movie_id,
                event_timestamp,
                rating_raw,
                rating_jittered,
                label,
                seq_order,
                total_user_rows,
                CASE
                    WHEN total_user_rows = 1 THEN 'train'
                    WHEN ((CAST(seq_order AS DOUBLE) - 0.5) / total_user_rows) < {train_cutoff} THEN 'train'
                    WHEN ((CAST(seq_order AS DOUBLE) - 0.5) / total_user_rows) < {val_cutoff} THEN 'val'
                    ELSE 'test'
                END AS dataset_split
            FROM ranked
        """)
        print (f"[DEBUG] split_index_bucket for bucket {bucket_id}:")
        # 2) current bucket source features only
        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE source_features_bucket AS
            SELECT
                user_id,
                movie_id,
                event_timestamp,
                user_embedding,
                movie_embedding
            FROM (
                SELECT
                    {user_id_col} AS user_id,
                    {movie_id_col} AS movie_id,
                    {timestamp_col} AS event_timestamp,
                    {user_embedding_col} AS user_embedding,
                    {movie_embedding_col} AS movie_embedding,
                    ROW_NUMBER() OVER (
                        PARTITION BY {user_id_col}, {movie_id_col}, {timestamp_col}
                        ORDER BY {movie_id_col}
                    ) AS rn
                FROM {source_expr}
                WHERE ABS(hash(CAST({user_id_col} AS VARCHAR))) % {num_buckets} = {bucket_id}
            )
            WHERE rn = 1
        """)
        print (f"[DEBUG] source_features_bucket for bucket {bucket_id}:")

        # 3) stats before writing
        bucket_stats = con.execute("""
            SELECT
                SUM(CASE WHEN dataset_split = 'train' THEN 1 ELSE 0 END) AS train_rows,
                SUM(CASE WHEN dataset_split = 'val' THEN 1 ELSE 0 END) AS val_rows,
                SUM(CASE WHEN dataset_split = 'test' THEN 1 ELSE 0 END) AS test_rows,
                COUNT(*) AS total_rows
            FROM split_index_bucket
        """).fetchone()

        joined_count = con.execute("""
            SELECT COUNT(*)
            FROM split_index_bucket s
            INNER JOIN source_features_bucket f
                ON s.user_id = f.user_id
               AND s.movie_id = f.movie_id
               AND s.event_timestamp = f.event_timestamp
        """).fetchone()[0]

        total_train += int(bucket_stats[0] or 0)
        total_val += int(bucket_stats[1] or 0)
        total_test += int(bucket_stats[2] or 0)
        total_dropped += int(bucket_stats[3] or 0) - int(joined_count or 0)
        print (f"[DEBUG] bucket {bucket_id} stats: train={bucket_stats[0]}, val={bucket_stats[1]}, test={bucket_stats[2]}, total={bucket_stats[3]}, dropped_on_join={int(bucket_stats[3] or 0) - int(joined_count or 0)}")

        # 4) directly COPY each split; no global split_data temp table
        def copy_bucket_split(split_name: str, out_path: str):
            con.execute(f"""
                COPY (
                    SELECT
                        s.user_id,
                        s.movie_id,
                        f.user_embedding,
                        f.movie_embedding
                        {extra_cols},
                        s.label
                    FROM split_index_bucket s
                    INNER JOIN source_features_bucket f
                        ON s.user_id = f.user_id
                       AND s.movie_id = f.movie_id
                       AND s.event_timestamp = f.event_timestamp
                    WHERE s.dataset_split = {sql_quote(split_name)}
                    ORDER BY s.user_id, s.seq_order
                )
                TO {sql_quote(out_path)}
                (FORMAT PARQUET, COMPRESSION ZSTD{per_thread_clause})
            """)

        copy_bucket_split("train", os.path.join(train_root, f"bucket_{bucket_id:04d}.parquet"))
        copy_bucket_split("val", os.path.join(val_root, f"bucket_{bucket_id:04d}.parquet"))
        copy_bucket_split("test", os.path.join(test_root, f"bucket_{bucket_id:04d}.parquet"))
        print (f"[INFO] finished bucket {bucket_id}/{num_buckets - 1}")
    return {
        "train_rows": total_train,
        "val_rows": total_val,
        "test_rows": total_test,
        "total_rows": total_train + total_val + total_test,
        "rows_dropped_on_join": total_dropped,
        "num_buckets": num_buckets,
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

    duckdb_path = deep_get(cfg, ["runtime", "duckdb_path"], ":memory:")
    if duckdb_path != ":memory:":
        ensure_dir(os.path.dirname(duckdb_path))

    con = duckdb.connect(database=duckdb_path)
    try:
        configure_duckdb_runtime(con, cfg)
        print("[DEBUG] using connection id:", id(con))

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
            input_info = resolve_latest_input_source(source_root_dir, con=con)

        
        input_source = input_info["input_source"]
        input_kind = input_info["input_kind"]
        source_registry = input_info.get("source_registry")
        source_version = input_info.get("source_version")


        dataset_root = output_root
        versions_dir = os.path.join(dataset_root, "versions")
        latest_dir = os.path.join(dataset_root, "latest")

        metadata_dataset_root = dataset_root if not local_metadata_root else os.path.join(local_metadata_root, dataset_type)
        registry_path = os.path.join(metadata_dataset_root, "registry", "version.json")

        if not is_s3_path(versions_dir):
            ensure_dir(versions_dir)
            ensure_dir(latest_dir)
            ensure_dir(os.path.dirname(registry_path))

        registry = get_registry_json(registry_path, con=con)

        version_name = next_version_name(registry.get("versions", []))
        version_dir = os.path.join(versions_dir, version_name)
        duckdb_path = deep_get(cfg, ["runtime", "duckdb_path"], ":memory:")
        if duckdb_path != ":memory:":
            ensure_dir(os.path.dirname(duckdb_path))

        print("[INFO] : Version name:", version_name)
        print("[INFO] : Input source:", input_source)
        print("[INFO] : Version dir:", version_dir)
        print("[INFO] : Versions dir:", versions_dir)


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

        print(f"[{job_name}] split dataset created with stats: {stats}")

        # con = duckdb.connect(database=duckdb_path)
        # try:
        #     configure_duckdb_runtime(con, cfg)

        #     stats = build_split_dataset(
        #         con=con,
        #         input_source=input_source,
        #         input_kind=input_kind,
        #         output_dir=version_dir,
        #         split_strategy=split_strategy,
        #         train_ratio=train_ratio,
        #         val_ratio=val_ratio,
        #         test_ratio=test_ratio,
        #         keep_rating_columns=keep_rating_columns,
        #         sigmoid_k=sigmoid_k,
        #         sigmoid_c=sigmoid_c,
        #         jitter_amplitude=jitter_amplitude,
        #         rating_col=rating_col,
        #         user_id_col=user_id_col,
        #         movie_id_col=movie_id_col,
        #         timestamp_col=timestamp_col,
        #         user_embedding_col=user_embedding_col,
        #         movie_embedding_col=movie_embedding_col,
        #         union_by_name=union_by_name,
        #         per_thread_output=per_thread_output,
        #         write_partitioned_output=write_partitioned_output,
        #     )
        # finally:
        #     con.close()
        


        if not write_partitioned_output:
            latest_train = os.path.join(latest_dir, "train.parquet")
            latest_val = os.path.join(latest_dir, "val.parquet")
            latest_test = os.path.join(latest_dir, "test.parquet")

            if any(is_s3_path(p) for p in [version_dir, latest_dir]):
                # Keep registry/manifest locally, but avoid trying to copy remote objects in Python.
                copy_targets = []
            else:
                ensure_dir(latest_dir)
                copy_targets = [
                    (os.path.join(version_dir, "train.parquet"), latest_train),
                    (os.path.join(version_dir, "val.parquet"), latest_val),
                    (os.path.join(version_dir, "test.parquet"), latest_test),
                ]
                for src, dst in copy_targets:
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
            "label_transform": {
                "deterministic_jitter_key": [user_id_col, movie_id_col],
                "jitter_method": "md5_number_lower(user_id || ':' || movie_id)",
                "jitter_amplitude": jitter_amplitude,
                "rating_clip_range": [0.5, 5.0],
                "normalize_formula": "(rating_jittered - 0.5) / 4.5",
                "sigmoid_k": sigmoid_k,
                "sigmoid_c": sigmoid_c,
            },
            "split": {
                "strategy": split_strategy,
                "train_ratio": train_ratio,
                "val_ratio": val_ratio,
                "test_ratio": test_ratio,
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
        if not is_s3_path(version_metadata_dir):
            ensure_dir(version_metadata_dir)
        if not is_s3_path(latest_metadata_dir):
            ensure_dir(latest_metadata_dir)

        manifest_path = os.path.join(version_metadata_dir, "manifest.json")
        save_json_any(manifest_path, manifest, con=con)

        latest_manifest_path = os.path.join(latest_metadata_dir, "manifest.json")
        save_json_any(latest_manifest_path, manifest, con=con)

        registry_entry = {
            "version": version_name,
            "created_at": created_at,
            "dataset_type": dataset_type,
            "source_version": source_version,
            "version_dir": version_dir,
            "manifest": manifest_path,
            "train_parquet": None if write_partitioned_output else os.path.join(version_dir, "train.parquet"),
            "val_parquet": None if write_partitioned_output else os.path.join(version_dir, "val.parquet"),
            "test_parquet": None if write_partitioned_output else os.path.join(version_dir, "test.parquet"),
            "row_count": stats["total_rows"],
        }

        registry["versions"] = registry.get("versions", [])
        registry["versions"].append(registry_entry)
        registry["latest"] = version_name
        save_registry_json(registry_path, registry, con=con)

    finally:
        print(f"[Error] Closing DuckDB connection")
        con.close()

    print(f"[{job_name}] done")
    print(json.dumps(manifest["stats"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
