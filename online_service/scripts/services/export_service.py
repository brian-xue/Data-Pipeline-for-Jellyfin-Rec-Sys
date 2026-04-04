from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import PurePosixPath
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import s3fs

from scripts.config.config import ObjectStorageConfig


def _s3_filesystem(storage: ObjectStorageConfig) -> s3fs.S3FileSystem:
    return s3fs.S3FileSystem(
        key=storage.access_key,
        secret=storage.secret_key,
        client_kwargs={"endpoint_url": storage.endpoint},
    )


def _is_s3(path: str) -> bool:
    return path.startswith("s3://")


def _strip_s3(path: str) -> str:
    return path.replace("s3://", "", 1)


def _ensure_parent(path: str, fs: s3fs.S3FileSystem | None = None) -> None:
    parent = str(PurePosixPath(path).parent)
    if parent in {"", "."}:
        return
    if fs is None:
        os.makedirs(parent, exist_ok=True)
    else:
        fs.makedirs(parent, exist_ok=True)


def _normalize_value_for_parquet(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def _normalize_rows_for_parquet(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        normalized.append({key: _normalize_value_for_parquet(val) for key, val in row.items()})
    return normalized


def export_rows_to_parquet(rows: list[dict[str, Any]], output_path: str, storage: ObjectStorageConfig) -> str:
    table = pa.Table.from_pylist(_normalize_rows_for_parquet(rows))

    if _is_s3(output_path):
        fs = _s3_filesystem(storage)
        fs_path = _strip_s3(output_path)
        _ensure_parent(fs_path, fs)
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
            tmp_name = tmp.name
        try:
            pq.write_table(table, tmp_name, compression="zstd")
            with open(tmp_name, "rb") as src, fs.open(fs_path, "wb") as dst:
                dst.write(src.read())
        finally:
            if os.path.exists(tmp_name):
                os.remove(tmp_name)
        return output_path

    _ensure_parent(output_path)
    pq.write_table(table, output_path, compression="zstd")
    return output_path


def write_json(payload: dict[str, Any], output_path: str, storage: ObjectStorageConfig) -> str:
    data = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")

    if _is_s3(output_path):
        fs = _s3_filesystem(storage)
        fs_path = _strip_s3(output_path)
        _ensure_parent(fs_path, fs)
        with fs.open(fs_path, "wb") as handle:
            handle.write(data)
        return output_path

    _ensure_parent(output_path)
    with open(output_path, "wb") as handle:
        handle.write(data)
    return output_path


def read_json_or_default(path: str, default_payload: dict[str, Any], storage: ObjectStorageConfig) -> dict[str, Any]:
    if _is_s3(path):
        fs = _s3_filesystem(storage)
        fs_path = _strip_s3(path)
        if not fs.exists(fs_path):
            return default_payload
        with fs.open(fs_path, "rb") as handle:
            try:
                return json.load(handle)
            except Exception:
                return default_payload

    if not os.path.exists(path):
        return default_payload
    with open(path, "r", encoding="utf-8") as handle:
        try:
            return json.load(handle)
        except Exception:
            return default_payload


def build_export_manifest(version: str, exported_paths: dict[str, str], stats: dict[str, Any]) -> dict[str, Any]:
    return {
        "version": version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "exports": exported_paths,
        "stats": stats,
    }
