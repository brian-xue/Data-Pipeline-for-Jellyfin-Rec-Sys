import os
import json
import argparse
import tempfile
from datetime import datetime, timezone
from typing import Any, Dict, Optional, List

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from minio_s3 import to_s3_uri, upload_bytes_to_path, upload_file_to_path, s3_filesystem


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
    return fs.open(
        to_s3_uri(path_str),
        "rb",
        cache_type="readahead",
        block_size=8 * 1024 * 1024,
    )


# ============================================================
# Output schema
# ============================================================
SIMULATOR_BASE_PROFILE_SCHEMA = pa.schema([
    pa.field("user_id", pa.int64()),
    pa.field("embedding", pa.list_(pa.float32())),
])


# ============================================================
# Simple parquet writer
# ============================================================
class ColumnBufferParquetWriter:
    def __init__(self, path: str, schema: pa.Schema, flush_rows: int = 50000):
        self.path = path
        self.schema = schema
        self.flush_rows = int(flush_rows)
        self.writer: Optional[pq.ParquetWriter] = None
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.buffers = {field.name: [] for field in schema}
        self.row_count = 0

    def append_one(self, row: Dict[str, Any]) -> None:
        for field in self.schema:
            self.buffers[field.name].append(row.get(field.name))
        self.row_count += 1
        if self.row_count >= self.flush_rows:
            self.flush()

    def flush(self) -> None:
        if self.row_count == 0:
            return

        arrays = {
            field.name: pa.array(self.buffers[field.name], type=field.type)
            for field in self.schema
        }
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

    def close(self) -> None:
        self.flush()
        if self.writer is not None:
            self.writer.close()
            self.writer = None


# ============================================================
# Helpers
# ============================================================
def choose_embedding(long_emb, short_emb):
    if long_emb is not None and len(long_emb) > 0:
        return long_emb, "long"
    if short_emb is not None and len(short_emb) > 0:
        return short_emb, "short"
    return None, None


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)

    job_name = deep_get(cfg, ["job", "name"], "build_simulator_base_profile")
    input_path = deep_get(cfg, ["input", "base_user_profiles_path"])
    output_dir = deep_get(cfg, ["output", "base_dir"], "s3://artifacts/simulator_base_profile")
    flush_rows = int(deep_get(cfg, ["output", "writer_flush_rows"], 50000))
    manifest_path = deep_get(cfg, ["artifact", "manifest_path"], os.path.join(output_dir, "manifest.json"))

    if not input_path:
        raise ValueError("input.base_user_profiles_path is required")

    local_output_dir = tempfile.mkdtemp(prefix="simulator_base_profile_")
    local_output_file = os.path.join(local_output_dir, "simulator_base_profile.parquet")
    output_target = os.path.join(output_dir, "simulator_base_profile.parquet")

    writer = ColumnBufferParquetWriter(
        path=local_output_file,
        schema=SIMULATOR_BASE_PROFILE_SCHEMA,
        flush_rows=flush_rows,
    )

    stats = {
        "rows_seen": 0,
        "rows_written": 0,
        "rows_skipped_no_embedding": 0,
        "rows_used_long_embedding": 0,
        "rows_used_short_embedding": 0,
    }

    print(f"[{job_name}] reading: {input_path}")

    with open_input_binary(input_path) as source:
        parquet_file = pq.ParquetFile(source)

        for batch in parquet_file.iter_batches(
            batch_size=flush_rows,
            columns=["user_id", "long_term_embedding", "short_term_embedding"],
        ):
            table = pa.Table.from_batches([batch])

            user_ids = table.column("user_id").to_pylist()
            long_embs = table.column("long_term_embedding").to_pylist()
            short_embs = table.column("short_term_embedding").to_pylist()

            for user_id, long_emb, short_emb in zip(user_ids, long_embs, short_embs):
                stats["rows_seen"] += 1

                emb, emb_source = choose_embedding(long_emb, short_emb)
                if emb is None:
                    stats["rows_skipped_no_embedding"] += 1
                    continue

                writer.append_one({
                    "user_id": int(user_id),
                    "embedding": [float(x) for x in emb],
                })

                stats["rows_written"] += 1
                if emb_source == "long":
                    stats["rows_used_long_embedding"] += 1
                else:
                    stats["rows_used_short_embedding"] += 1

    writer.close()

    uploaded_profile_path = upload_file_to_path(local_output_file, output_target)

    manifest = {
        "job_name": job_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": {
            "base_user_profiles_path": to_s3_uri(input_path),
        },
        "output": {
            "base_dir": to_s3_uri(output_dir),
            "simulator_base_profile": uploaded_profile_path,
        },
        "selection_rule": {
            "preferred_embedding": "long_term_embedding",
            "fallback_embedding": "short_term_embedding",
            "skip_if_both_missing": True,
        },
        "stats": stats,
    }

    manifest_payload = json.dumps(manifest, indent=2, ensure_ascii=False).encode("utf-8")
    manifest_uri = upload_bytes_to_path(manifest_payload, manifest_path)

    print(f"[{job_name}] done")
    print(f"[{job_name}] profile uploaded: {uploaded_profile_path}")
    print(f"[{job_name}] manifest uploaded: {manifest_uri}")
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()