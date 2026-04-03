import os
import json
import argparse
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import duckdb
import yaml


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
    os.makedirs(path, exist_ok=True)


def require_value(name: str, value: Any) -> None:
    if value is None:
        raise ValueError(f"Missing required config: {name}")
    if isinstance(value, str) and not value.strip():
        raise ValueError(f"Empty required config: {name}")


def get_registry_json(registry_path: str) -> dict:
    if not os.path.exists(registry_path):
        return {"versions": [], "latest": None}
    with open(registry_path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except Exception:
            return {"versions": [], "latest": None}


# ============================================================
# Resolve latest source
# ============================================================
def resolve_latest_input_source(source_root_dir: str) -> Dict[str, str]:
    """
    Resolve latest upstream dataset from a local registry file.

    Supported upstream registry fields:
      - data_parts_prefix
      - data_parquet
      - version_dir

    Returns:
      {
        "source_registry": ...,
        "source_version": ...,
        "input_source": ...,
        "input_kind": "parts_prefix" | "data_parquet" | "version_dir_fallback"
      }
    """
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
        data_file = os.path.join(version_dir, "data.parquet")

        if parts_dir:
            return {
                "source_registry": registry_path,
                "source_version": latest_version,
                "input_source": parts_dir,
                "input_kind": "version_dir_fallback",
            }

        if data_file:
            return {
                "source_registry": registry_path,
                "source_version": latest_version,
                "input_source": data_file,
                "input_kind": "version_dir_fallback",
            }

    raise ValueError(
        f"Cannot resolve latest input dataset from source registry: {registry_path}, "
        f"latest entry: {latest_entry}"
    )


def parquet_read_expr(input_source: str, input_kind: str) -> str:
    """
    Build DuckDB read_parquet expression for either:
      - single parquet file
      - parquet dataset directory with many small files
    """
    if input_kind == "parts_prefix":
        return f"read_parquet('{input_source}/*.parquet')"

    if input_kind in {"data_parquet", "version_dir_fallback"}:
        # If it looks like a parquet directory/prefix, read all files inside.
        if input_source.endswith("/") or not input_source.endswith(".parquet"):
            return f"read_parquet('{input_source.rstrip('/')}/*.parquet')"
        return f"read_parquet('{input_source}')"

    raise ValueError(f"Unsupported input_kind: {input_kind}")


def s3_prefix_parquet_glob(prefix: str) -> str:
    return prefix.rstrip("/") + "/*.parquet"


# ============================================================
# DuckDB helpers
# ============================================================
def configure_duckdb_for_s3(
    con: duckdb.DuckDBPyConnection,
    endpoint: str,
    region: str,
    access_key: Optional[str],
    secret_key: Optional[str],
    use_ssl: bool,
    url_style: str,
) -> None:
    try:
        con.execute("LOAD httpfs;")
    except Exception:
        con.execute("INSTALL httpfs;")
        con.execute("LOAD httpfs;")

    # DuckDB expects endpoint without scheme in many setups.
    endpoint_no_scheme = endpoint.replace("http://", "").replace("https://", "")

    con.execute(f"SET s3_region='{region}'")
    con.execute(f"SET s3_endpoint='{endpoint_no_scheme}'")
    con.execute(f"SET s3_use_ssl={'true' if use_ssl else 'false'}")
    con.execute(f"SET s3_url_style='{url_style}'")

    if access_key:
        con.execute(f"SET s3_access_key_id='{access_key}'")
    if secret_key:
        con.execute(f"SET s3_secret_access_key='{secret_key}'")


def scalar_row_to_dict(description: List[Any], row: tuple) -> Dict[str, Any]:
    out = {}
    for idx, col in enumerate(description):
        key = col[0]
        value = row[idx]
        if isinstance(value, datetime):
            value = value.isoformat()
        out[key] = value
    return out


def fetch_one_dict(con: duckdb.DuckDBPyConnection, query: str) -> Dict[str, Any]:
    print(f"[fetch_one_dict] Executing query:\n{query}\n")
    cur = con.execute(query)
    row = cur.fetchone()
    if row is None:
        return {}
    return scalar_row_to_dict(cur.description, row)


def fetch_all_dicts(con: duckdb.DuckDBPyConnection, query: str) -> List[Dict[str, Any]]:
    print(f"[fetch_all_dicts] Executing query:\n{query}\n")
    cur = con.execute(query)
    rows = cur.fetchall()
    cols = [c[0] for c in cur.description]
    result = []
    for row in rows:
        item = {}
        for i, val in enumerate(row):
            if isinstance(val, datetime):
                val = val.isoformat()
            item[cols[i]] = val
        result.append(item)
    return result


# ============================================================
# Profiling
# ============================================================
def profile_dataset(
    input_source: str,
    input_kind: str,
    output_json: str,
    time_column: str = "timestamp",
    user_column: str = "user_id",
    movie_column: str = "movie_id",
    label_column: str = "label",
    user_embedding_column: str = "user_embedding",
    movie_embedding_column: str = "movie_embedding",
    duckdb_memory_limit: str = "2GB",
    duckdb_threads: int = 4,
    temp_directory: Optional[str] = None,
    embedding_sample_rows: int = 100000,
    enable_user_stats: bool = True,
    enable_movie_stats: bool = True,
    enable_label_histogram: bool = True,
    enable_embedding_similarity_sample: bool = True,
    s3_endpoint: str = "http://minio:9000",
    s3_region: str = "eu-central-1",
    s3_use_ssl: bool = False,
    s3_url_style: str = "path",
    s3_access_key_id: Optional[str] = None,
    s3_secret_access_key: Optional[str] = None,
) -> Dict[str, Any]:
    ensure_dir(os.path.dirname(output_json))

    read_expr = parquet_read_expr(input_source, input_kind)

    con = duckdb.connect(database=":memory:")
    print(f"[profile_dataset] Starting profiling for input_source: {input_source}, input_kind: {input_kind}")
    try:
        con.execute(f"SET memory_limit='{duckdb_memory_limit}'")
        con.execute(f"SET threads={duckdb_threads}")
        if temp_directory:
            ensure_dir(temp_directory)
            con.execute(f"SET temp_directory='{temp_directory}'")

        if input_source.startswith("s3://"):
            configure_duckdb_for_s3(
                con=con,
                endpoint=s3_endpoint,
                region=s3_region,
                access_key=s3_access_key_id,
                secret_key=s3_secret_access_key,
                use_ssl=s3_use_ssl,
                url_style=s3_url_style,
            )

        profile: Dict[str, Any] = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "input": {
                "input_source": input_source,
                "input_kind": input_kind,
                "time_column": time_column,
                "user_column": user_column,
                "movie_column": movie_column,
                "label_column": label_column,
                "user_embedding_column": user_embedding_column,
                "movie_embedding_column": movie_embedding_column,
            },
            "runtime": {
                "duckdb_memory_limit": duckdb_memory_limit,
                "duckdb_threads": duckdb_threads,
                "temp_directory": temp_directory,
                "embedding_sample_rows": embedding_sample_rows,
            },
            "s3": {
                "endpoint": s3_endpoint,
                "region": s3_region,
                "use_ssl": s3_use_ssl,
                "url_style": s3_url_style,
            },
        }

        # 1) Dataset overview
        profile["dataset_overview"] = fetch_one_dict(
            con,
            f"""
            SELECT
                COUNT(*) AS total_rows,
                MIN({time_column}) AS min_time,
                MAX({time_column}) AS max_time,
                APPROX_COUNT_DISTINCT({user_column}) AS approx_distinct_users,
                APPROX_COUNT_DISTINCT({movie_column}) AS approx_distinct_movies
            FROM {read_expr}
            """
        )

        # 2) Label stats
        # profile["label_stats"] = fetch_one_dict(
        #     con,
        #     f"""
        #     SELECT
        #         MIN({label_column}) AS label_min,
        #         MAX({label_column}) AS label_max,
        #         AVG({label_column}) AS label_avg,
        #         STDDEV_SAMP({label_column}) AS label_std,
        #         APPROX_QUANTILE({label_column}, 0.01) AS label_p01,
        #         APPROX_QUANTILE({label_column}, 0.10) AS label_p10,
        #         APPROX_QUANTILE({label_column}, 0.50) AS label_p50,
        #         APPROX_QUANTILE({label_column}, 0.90) AS label_p90,
        #         APPROX_QUANTILE({label_column}, 0.99) AS label_p99
        #     FROM {read_expr}
        #     """
        # )

        # if enable_label_histogram:
        #     profile["label_histogram"] = fetch_all_dicts(
        #         con,
        #         f"""
        #         SELECT
        #             {label_column} AS label_value,
        #             COUNT(*) AS row_count
        #         FROM {read_expr}
        #         GROUP BY {label_column}
        #         ORDER BY {label_column}
        #         """
        #     )

        # 3) User activity stats
        if enable_user_stats:
            con.execute(
                f"""
                CREATE TEMP TABLE user_counts AS
                SELECT
                    {user_column} AS user_id,
                    COUNT(*) AS cnt
                FROM {read_expr}
                GROUP BY {user_column}
                """
            )

            profile["user_activity_stats"] = fetch_one_dict(
                con,
                """
                SELECT
                    COUNT(*) AS num_users,
                    MIN(cnt) AS min_events_per_user,
                    MAX(cnt) AS max_events_per_user,
                    AVG(cnt) AS avg_events_per_user,
                    STDDEV_SAMP(cnt) AS std_events_per_user,
                    APPROX_QUANTILE(cnt, 0.50) AS p50_events_per_user,
                    APPROX_QUANTILE(cnt, 0.90) AS p90_events_per_user,
                    APPROX_QUANTILE(cnt, 0.99) AS p99_events_per_user
                FROM user_counts
                """
            )

        # 4) Movie activity stats
        if enable_movie_stats:
            con.execute(
                f"""
                CREATE TEMP TABLE movie_counts AS
                SELECT
                    {movie_column} AS movie_id,
                    COUNT(*) AS cnt
                FROM {read_expr}
                GROUP BY {movie_column}
                """
            )

            profile["movie_activity_stats"] = fetch_one_dict(
                con,
                """
                SELECT
                    COUNT(*) AS num_movies,
                    MIN(cnt) AS min_events_per_movie,
                    MAX(cnt) AS max_events_per_movie,
                    AVG(cnt) AS avg_events_per_movie,
                    STDDEV_SAMP(cnt) AS std_events_per_movie,
                    APPROX_QUANTILE(cnt, 0.50) AS p50_events_per_movie,
                    APPROX_QUANTILE(cnt, 0.90) AS p90_events_per_movie,
                    APPROX_QUANTILE(cnt, 0.99) AS p99_events_per_movie
                FROM movie_counts
                """
            )

        # 5) Embedding shape stats on sample
        # profile["embedding_shape_sample_stats"] = fetch_one_dict(
        #     con,
        #     f"""
        #     SELECT
        #         COUNT(*) AS sampled_rows,
        #         MIN(LENGTH({user_embedding_column})) AS user_emb_dim_min,
        #         MAX(LENGTH({user_embedding_column})) AS user_emb_dim_max,
        #         APPROX_QUANTILE(LENGTH({user_embedding_column}), 0.50) AS user_emb_dim_p50,
        #         MIN(LENGTH({movie_embedding_column})) AS movie_emb_dim_min,
        #         MAX(LENGTH({movie_embedding_column})) AS movie_emb_dim_max,
        #         APPROX_QUANTILE(LENGTH({movie_embedding_column}), 0.50) AS movie_emb_dim_p50
        #     FROM (
        #         SELECT
        #             {user_embedding_column},
        #             {movie_embedding_column}
        #         FROM {read_expr}
        #         USING SAMPLE reservoir({embedding_sample_rows} ROWS)
        #     ) t
        #     """
        # )

        # 6) Small sample of cosine similarity between embeddings
        if enable_embedding_similarity_sample:
            # profile["embedding_similarity_sample_stats"] = fetch_one_dict(
            #     con,
            #     f"""
            #     WITH sampled AS (
            #         SELECT
            #             {user_embedding_column} AS uemb,
            #             {movie_embedding_column} AS memb
            #         FROM {read_expr}
            #         USING SAMPLE reservoir({embedding_sample_rows} ROWS)
            #     ),
            #     sim AS (
            #         SELECT
            #             list_dot_product(uemb, memb) /
            #             NULLIF(
            #                 SQRT(list_dot_product(uemb, uemb)) *
            #                 SQRT(list_dot_product(memb, memb)),
            #                 0
            #             ) AS cosine_sim
            #         FROM sampled
            #     )
            #     SELECT
            #         COUNT(*) AS sampled_rows,
            #         MIN(cosine_sim) AS sim_min,
            #         MAX(cosine_sim) AS sim_max,
            #         AVG(cosine_sim) AS sim_avg,
            #         STDDEV_SAMP(cosine_sim) AS sim_std,
            #         APPROX_QUANTILE(cosine_sim, 0.01) AS sim_p01,
            #         APPROX_QUANTILE(cosine_sim, 0.10) AS sim_p10,
            #         APPROX_QUANTILE(cosine_sim, 0.50) AS sim_p50,
            #         APPROX_QUANTILE(cosine_sim, 0.90) AS sim_p90,
            #         APPROX_QUANTILE(cosine_sim, 0.99) AS sim_p99
            #     FROM sim
            #     """
            # )
            profile["embedding_similarity_sample_histogram"] = fetch_all_dicts(
                con,
                f"""
                WITH sampled AS (
                    SELECT
                        {user_embedding_column} AS uemb,
                        {movie_embedding_column} AS memb,
                        {label_column} AS label_value
                    FROM {read_expr}
                    where random() <0.02
                ),
                sim AS (
                    SELECT
                        label_value,
                        list_dot_product(uemb, memb) /
                        NULLIF(
                            SQRT(list_dot_product(uemb, uemb)) *
                            SQRT(list_dot_product(memb, memb)),
                            0
                        ) AS cosine_sim
                    FROM sampled
                ),
                binned AS (
                    SELECT
                        label_value,
                        cosine_sim,
                        FLOOR(cosine_sim * 20) / 20 AS sim_bin
                    FROM sim
                    WHERE cosine_sim IS NOT NULL
                )
                SELECT
                    sim_bin,
                    COUNT(*) AS sample_count,
                    AVG(label_value) AS avg_label,
                    MIN(label_value) AS min_label,
                    MAX(label_value) AS max_label,
                    AVG(cosine_sim) AS avg_cosine_sim
                FROM binned
                GROUP BY sim_bin
                ORDER BY sim_bin
                """
            )

        # 7) Earliest and latest timestamps sample
        # profile["time_sample_extremes"] = fetch_all_dicts(
        #     con,
        #     f"""
        #     (
        #         SELECT {time_column} AS time_value
        #         FROM {read_expr}
        #         ORDER BY {time_column} ASC
        #         LIMIT 5
        #     )
        #     UNION ALL
        #     (
        #         SELECT {time_column} AS time_value
        #         FROM {read_expr}
        #         ORDER BY {time_column} DESC
        #         LIMIT 5
        #     )
        #     """
        # )

        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(profile, f, indent=2, ensure_ascii=False)

        return profile
    finally:
        con.close()


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)

    # You can either provide source_root_dir to resolve latest,
    # or directly provide input_source + input_kind.
    source_root_dir = deep_get(cfg, ["input", "source_root_dir"])
    direct_input_source = deep_get(cfg, ["input", "input_source"])
    direct_input_kind = deep_get(cfg, ["input", "input_kind"])

    if source_root_dir:
        input_info = resolve_latest_input_source(source_root_dir)
    else:
        require_value("input.input_source", direct_input_source)
        require_value("input.input_kind", direct_input_kind)
        input_info = {
            "source_registry": None,
            "source_version": None,
            "input_source": direct_input_source,
            "input_kind": direct_input_kind,
        }

    output_json = deep_get(
        cfg,
        ["output", "profile_json"],
        "profile_latest.json",
    )

    time_column = deep_get(cfg, ["profile", "time_column"], "timestamp")
    user_column = deep_get(cfg, ["profile", "user_column"], "user_id")
    movie_column = deep_get(cfg, ["profile", "movie_column"], "movie_id")
    label_column = deep_get(cfg, ["profile", "label_column"], "label")
    user_embedding_column = deep_get(cfg, ["profile", "user_embedding_column"], "user_embedding")
    movie_embedding_column = deep_get(cfg, ["profile", "movie_embedding_column"], "movie_embedding")

    duckdb_memory_limit = deep_get(cfg, ["runtime", "duckdb_memory_limit"], "2GB")
    duckdb_threads = int(deep_get(cfg, ["runtime", "duckdb_threads"], 4))
    temp_directory = deep_get(cfg, ["runtime", "temp_directory"], None)
    embedding_sample_rows = int(deep_get(cfg, ["runtime", "embedding_sample_rows"], 100000))

    enable_user_stats = bool(deep_get(cfg, ["profile", "enable_user_stats"], True))
    enable_movie_stats = bool(deep_get(cfg, ["profile", "enable_movie_stats"], True))
    enable_label_histogram = bool(deep_get(cfg, ["profile", "enable_label_histogram"], True))
    enable_embedding_similarity_sample = bool(
        deep_get(cfg, ["profile", "enable_embedding_similarity_sample"], True)
    )

    s3_endpoint = deep_get(cfg, ["s3", "endpoint"], "http://minio:9000")
    s3_region = deep_get(cfg, ["s3", "region"], "eu-central-1")
    s3_use_ssl = bool(deep_get(cfg, ["s3", "use_ssl"], False))
    s3_url_style = deep_get(cfg, ["s3", "url_style"], "path")

    s3_access_key_id = (
        deep_get(cfg, ["s3", "access_key_id"])
        or os.getenv("MINIO_ACCESS_KEY")
        or os.getenv("AWS_ACCESS_KEY_ID")
        or os.getenv("MINIO_ROOT_USER")
    )
    s3_secret_access_key = (
        deep_get(cfg, ["s3", "secret_access_key"])
        or os.getenv("MINIO_SECRET_KEY")
        or os.getenv("AWS_SECRET_ACCESS_KEY")
        or os.getenv("MINIO_ROOT_PASSWORD")
    )

    profile = profile_dataset(
        input_source=input_info["input_source"],
        input_kind=input_info["input_kind"],
        output_json=output_json,
        time_column=time_column,
        user_column=user_column,
        movie_column=movie_column,
        label_column=label_column,
        user_embedding_column=user_embedding_column,
        movie_embedding_column=movie_embedding_column,
        duckdb_memory_limit=duckdb_memory_limit,
        duckdb_threads=duckdb_threads,
        temp_directory=temp_directory,
        embedding_sample_rows=embedding_sample_rows,
        enable_user_stats=enable_user_stats,
        enable_movie_stats=enable_movie_stats,
        enable_label_histogram=enable_label_histogram,
        enable_embedding_similarity_sample=enable_embedding_similarity_sample,
        s3_endpoint=s3_endpoint,
        s3_region=s3_region,
        s3_use_ssl=s3_use_ssl,
        s3_url_style=s3_url_style,
        s3_access_key_id=s3_access_key_id,
        s3_secret_access_key=s3_secret_access_key,
    )

    profile["resolved_latest"] = {
        "source_registry": input_info["source_registry"],
        "source_version": input_info["source_version"],
        "input_source": input_info["input_source"],
        "input_kind": input_info["input_kind"],
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2, ensure_ascii=False)

    print("[profile_dataset] done")
    print(json.dumps({
        "output_json": output_json,
        "total_rows": profile["dataset_overview"]["total_rows"],
        "approx_distinct_users": profile["dataset_overview"]["approx_distinct_users"],
        "approx_distinct_movies": profile["dataset_overview"]["approx_distinct_movies"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()