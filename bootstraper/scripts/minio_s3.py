from __future__ import annotations

import io
import os
from pathlib import Path, PurePosixPath
from typing import Any

import pandas as pd
import s3fs


DEFAULT_ENDPOINT = "http://minio:9000"


def _bucket_env() -> dict[str, str]:
    return {
        "raw": os.getenv("MINIO_RAW_BUCKET", "raw"),
        "cleaned": os.getenv("MINIO_CLEANED_BUCKET", "cleaned"),
        "embedding": os.getenv("MINIO_EMBEDDING_BUCKET", "embedding"),
        "artifacts": os.getenv("MINIO_ARTIFACTS_BUCKET", "artifacts"),
        "warehouse": os.getenv("MINIO_WAREHOUSE_BUCKET", "warehouse"),
    }


def s3_storage_options() -> dict[str, Any]:
    endpoint = os.getenv("MINIO_ENDPOINT", DEFAULT_ENDPOINT)
    key = os.getenv("MINIO_ACCESS_KEY") or os.getenv("AWS_ACCESS_KEY_ID") or os.getenv("MINIO_ROOT_USER")
    secret = os.getenv("MINIO_SECRET_KEY") or os.getenv("AWS_SECRET_ACCESS_KEY") or os.getenv("MINIO_ROOT_PASSWORD")

    options: dict[str, Any] = {
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


def local_data_path_to_s3_uri(path_like: str | Path) -> str | None:
    path_str = str(path_like)
    if path_str.startswith("s3://"):
        return path_str

    try:
        p = Path(path_str).resolve()
    except Exception:
        p = Path(path_str)

    parts = p.parts
    if len(parts) < 3 or parts[0] != "/" or parts[1] != "data":
        return None

    top = parts[2]
    mapped_bucket = _bucket_env().get(top)
    if mapped_bucket is None:
        return None

    key = PurePosixPath(*parts[3:]).as_posix()
    if key in {"", "."}:
        return f"s3://{mapped_bucket}"
    return f"s3://{mapped_bucket}/{key}"


def to_s3_uri(path_like: str | Path) -> str:
    value = str(path_like)
    if value.startswith("s3://"):
        return value
    mapped = local_data_path_to_s3_uri(path_like)
    if mapped is None:
        raise ValueError(f"Path cannot be mapped to MinIO bucket: {value}")
    return mapped


def resolve_input_path(path_like: str | Path) -> str:
    value = str(path_like)
    if value.startswith("s3://"):
        return value
    if Path(value).exists():
        return value
    mapped = local_data_path_to_s3_uri(value)
    return mapped if mapped is not None else value


def path_exists(path_like: str | Path) -> bool:
    resolved = resolve_input_path(path_like)
    if resolved.startswith("s3://"):
        fs = s3_filesystem()
        return fs.exists(resolved.replace("s3://", ""))
    return Path(resolved).exists()


def list_csv_files(path_like: str | Path) -> list[str]:
    resolved = resolve_input_path(path_like)
    if resolved.startswith("s3://"):
        fs = s3_filesystem()
        target = resolved.replace("s3://", "").rstrip("/")
        if not fs.exists(target):
            return []
        files = fs.ls(target, detail=False)
        return [f"s3://{item}" for item in files if item.lower().endswith(".csv")]

    path = Path(resolved)
    return [str(p) for p in sorted(path.glob("*.csv"))]


def read_csv_auto(path_like: str | Path, **kwargs: Any) -> pd.DataFrame:
    resolved = resolve_input_path(path_like)
    if resolved.startswith("s3://"):
        return pd.read_csv(resolved, storage_options=s3_storage_options(), **kwargs)
    return pd.read_csv(resolved, **kwargs)


def read_parquet_auto(path_like: str | Path, **kwargs: Any) -> pd.DataFrame:
    resolved = resolve_input_path(path_like)
    if resolved.startswith("s3://"):
        return pd.read_parquet(resolved, storage_options=s3_storage_options(), **kwargs)
    return pd.read_parquet(resolved, **kwargs)


def upload_bytes_to_path(
    data: bytes,
    target_path_like: str | Path,
    max_retries: int = 6,
    base_delay: float = 1.0,
) -> str:
    import time

    target_uri = to_s3_uri(target_path_like)
    fs = s3_filesystem()
    fs_path = target_uri.replace("s3://", "")
    parent = str(PurePosixPath(fs_path).parent)
    if parent and parent != ".":
        fs.makedirs(parent, exist_ok=True)

    delay = base_delay
    for attempt in range(max_retries):
        try:
            with fs.open(fs_path, "wb") as handle:
                handle.write(data)
            return target_uri
        except OSError as e:
            msg = str(e)
            is_slowdown = "SlowDownWrite" in msg or "reduce your request rate" in msg
            if attempt == max_retries - 1 or not is_slowdown:
                raise
            print(
                f"[upload bytes retry] attempt={attempt + 1}/{max_retries}, "
                f"target={target_uri}, sleep={delay}s, error={msg}"
            )
            time.sleep(delay)
            delay *= 2

    raise RuntimeError(f"Failed to upload bytes after retries: {target_uri}")


def upload_file_to_path(
    local_file: str | Path,
    target_path_like: str | Path,
    chunk_size: int = 8 * 1024 * 1024,
    max_retries: int = 6,
    base_delay: float = 1.0,
) -> str:
    target_uri = to_s3_uri(target_path_like)
    fs = s3_filesystem()
    fs_path = target_uri.replace("s3://", "")
    parent = str(PurePosixPath(fs_path).parent)
    if parent and parent != ".":
        fs.makedirs(parent, exist_ok=True)

    local_file = str(local_file)

    delay = base_delay
    for attempt in range(max_retries):
        try:
            with open(local_file, "rb") as src, fs.open(fs_path, "wb") as dst:
                while True:
                    chunk = src.read(chunk_size)
                    if not chunk:
                        break
                    dst.write(chunk)
            return target_uri

        except OSError as e:
            msg = str(e)
            is_slowdown = "SlowDownWrite" in msg or "reduce your request rate" in msg

            if attempt == max_retries - 1 or not is_slowdown:
                raise

            print(
                f"[upload retry] attempt={attempt + 1}/{max_retries}, "
                f"target={target_uri}, sleep={delay}s, error={msg}"
            )
            time.sleep(delay)
            delay *= 2

    raise RuntimeError(f"Failed to upload after retries: {target_uri}")


def write_dataframe_parquet_to_path(df: pd.DataFrame, target_path_like: str | Path, index: bool = False) -> str:
    buffer = io.BytesIO()
    df.to_parquet(buffer, index=index)
    return upload_bytes_to_path(buffer.getvalue(), target_path_like)


def write_dataframe_jsonl_to_path(df: pd.DataFrame, target_path_like: str | Path) -> str:
    text = df.to_json(orient="records", lines=True, force_ascii=False)
    return upload_bytes_to_path(text.encode("utf-8"), target_path_like)
