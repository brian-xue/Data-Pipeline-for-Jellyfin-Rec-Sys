import os
import json
import math
import hashlib
import argparse
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
import io
import time
import yaml
import boto3
from botocore.client import Config
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pyarrow.fs as pafs


DEFAULT_BATCH_SIZE = 100_000
DEFAULT_OFFLINE_SIGMOID_K = 5.0
DEFAULT_OFFLINE_SIGMOID_C = 0.5
DEFAULT_OFFLINE_JITTER_AMPLITUDE = 0.1
DEFAULT_ONLINE_PLAY_TIME_CAP = 600.0


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


def split_s3_uri(uri: str) -> Tuple[str, str]:
    if not is_s3_path(uri):
        raise ValueError(f"Not an S3 URI: {uri}")
    no_scheme = uri[len("s3://"):]
    parts = no_scheme.split("/", 1)
    bucket = parts[0]
    key = "" if len(parts) == 1 else parts[1]
    return bucket, key


def s3_join(base: str, *parts: str) -> str:
    if not is_s3_path(base):
        return os.path.join(base, *parts)
    out = base.rstrip("/")
    for p in parts:
        out += "/" + p.strip("/")
    return out


def safe_json_default(obj: Any):
    if isinstance(obj, datetime):
        return obj.isoformat()
    return str(obj)

def upload_bytes_with_retry(
    s3_client,
    bucket: str,
    key: str,
    payload: bytes,
    content_type: str = "application/octet-stream",
    max_retries: int = 5,
    base_sleep_seconds: float = 1.0,
) -> None:
    last_error = None
    max_retries+= 3

    for attempt in range(1, max_retries + 1):
        try:
            s3_client.put_object(
                Bucket=bucket,
                Key=key,
                Body=payload,
                ContentType=content_type,
            )
            return
        except Exception as e:
            last_error = e
            print(
                f"[WARN] upload failed: s3://{bucket}/{key} "
                f"(attempt {attempt}/{max_retries}) -> {e}"
            )
            if attempt < max_retries:
                sleep_s = base_sleep_seconds * (2 ** (attempt - 1))
                time.sleep(sleep_s)

    raise RuntimeError(
        f"Failed to upload s3://{bucket}/{key} after {max_retries} attempts: {last_error}"
    )

def build_checkpoint_payload(
    *,
    job_name: str,
    input_source: str,
    input_kind: str,
    version_dir: str,
    dataset_type: str,
    resume_row: int,
    batch_idx: int,
    train_rows: int,
    val_rows: int,
    test_rows: int,
    train_parts: int,
    val_parts: int,
    test_parts: int,
    train_buffer_start_row: Optional[int],
    val_buffer_start_row: Optional[int],
    test_buffer_start_row: Optional[int],
    total_rows: int,
    train_cutoff: int,
    val_cutoff: int,
    last_write_split: Optional[str] = None,
    last_write_part_id: Optional[int] = None,
    last_write_file: Optional[str] = None,
    last_write_status: Optional[str] = None,
    last_write_error: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "job_name": job_name,
        "input": {
            "input_source": input_source,
            "input_kind": input_kind,
        },
        "output": {
            "version_dir": version_dir,
            "dataset_type": dataset_type,
        },
        "progress": {
            "resume_row": resume_row,
            "batch_idx": batch_idx,
            "train_rows": train_rows,
            "val_rows": val_rows,
            "test_rows": test_rows,
            "train_parts": train_parts,
            "val_parts": val_parts,
            "test_parts": test_parts,
        },
        "buffer_state": {
            "train_buffer_start_row": train_buffer_start_row,
            "val_buffer_start_row": val_buffer_start_row,
            "test_buffer_start_row": test_buffer_start_row,
        },
        "last_write": {
            "split": last_write_split,
            "part_id": last_write_part_id,
            "file": last_write_file,
            "status": last_write_status,
            "error": last_write_error,
        },
        "meta": {
            "total_rows": total_rows,
            "train_cutoff": train_cutoff,
            "val_cutoff": val_cutoff,
        },
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


# ============================================================
# S3 Clients / Filesystems
# ============================================================
def build_boto3_s3_client(cfg: Dict[str, Any]):
    s3_cfg = deep_get(cfg, ["storage", "s3"], {}) or {}

    endpoint = s3_cfg.get("endpoint")
    region = s3_cfg.get("region", "us-east-1")
    access_key = s3_cfg.get("access_key_id") or os.getenv("AWS_ACCESS_KEY_ID")
    secret_key = s3_cfg.get("secret_access_key") or os.getenv("AWS_SECRET_ACCESS_KEY")

    require_value("storage.s3.endpoint", endpoint)
    require_value("storage.s3.access_key_id", access_key)
    require_value("storage.s3.secret_access_key", secret_key)

    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region,
        config=Config(signature_version="s3v4"),
    )


def build_arrow_s3_filesystem(cfg: Dict[str, Any]) -> pafs.S3FileSystem:
    s3_cfg = deep_get(cfg, ["storage", "s3"], {}) or {}

    endpoint = s3_cfg.get("endpoint")
    region = s3_cfg.get("region", "us-east-1")
    access_key = s3_cfg.get("access_key_id") or os.getenv("AWS_ACCESS_KEY_ID")
    secret_key = s3_cfg.get("secret_access_key") or os.getenv("AWS_SECRET_ACCESS_KEY")
    use_ssl = bool(s3_cfg.get("use_ssl", False))
    scheme = "https" if use_ssl else "http"

    require_value("storage.s3.endpoint", endpoint)
    require_value("storage.s3.access_key_id", access_key)
    require_value("storage.s3.secret_access_key", secret_key)

    endpoint_no_scheme = endpoint.replace("http://", "").replace("https://", "")

    return pafs.S3FileSystem(
        access_key=access_key,
        secret_key=secret_key,
        endpoint_override=endpoint_no_scheme,
        scheme=scheme,
        region=region,
    )


# ============================================================
# JSON IO
# ============================================================
def load_json_any(
    path: str,
    filesystem: Optional[pafs.FileSystem] = None,
    s3_client=None,
) -> dict:
    if is_s3_path(path):
        if s3_client is None:
            raise ValueError(f"s3_client is required for S3 JSON read: {path}")
        bucket, key = split_s3_uri(path)
        print(f"Loading JSON from S3: bucket={bucket}, key={key}")
        obj = s3_client.get_object(Bucket=bucket, Key=key)
        data = obj["Body"].read().decode("utf-8")
        return json.loads(data)

    if not os.path.exists(path):
        return {"versions": [], "latest": None}

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json_any(
    path: str,
    obj: dict,
    filesystem: Optional[pafs.FileSystem] = None,
    s3_client=None,
) -> None:
    payload = json.dumps(
        obj,
        indent=2,
        ensure_ascii=False,
        default=safe_json_default,
    ).encode("utf-8")

    if is_s3_path(path):
        if s3_client is None:
            raise ValueError(f"s3_client is required for S3 JSON write: {path}")
        bucket, key = split_s3_uri(path)
        s3_client.put_object(
            Bucket=bucket,
            Key=key,
            Body=payload,
            ContentType="application/json",
        )
        return

    ensure_dir(os.path.dirname(path))
    with open(path, "wb") as f:
        f.write(payload)

# ============================================================
# Local checkpoint for resume
# ============================================================
def load_local_checkpoint(path: str) -> Optional[Dict[str, Any]]:
    if not path or not os.path.exists(path):
        return None

    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    # basic validation
    if not isinstance(obj, dict):
        raise ValueError(f"Invalid checkpoint format: {path}")

    progress = obj.get("progress", {})
    output = obj.get("output", {})
    meta = obj.get("meta", {})

    required_progress_keys = [
        "resume_row",
        "batch_idx",
        "train_parts",
        "val_parts",
        "test_parts",
        "train_rows",
        "val_rows",
        "test_rows",
    ]
    for k in required_progress_keys:
        if k not in progress:
            raise ValueError(f"Checkpoint missing progress.{k}: {path}")

    if "version_dir" not in output:
        raise ValueError(f"Checkpoint missing output.version_dir: {path}")

    # basic validation: meta is not always required, but recommended
    if not isinstance(meta, dict):
        raise ValueError(f"Checkpoint meta must be dict: {path}")

    return obj


def save_local_checkpoint(path: str, obj: Dict[str, Any]) -> None:
    ensure_dir(os.path.dirname(path))
    tmp_path = f"{path}.tmp"

    payload = json.dumps(
        obj,
        indent=2,
        ensure_ascii=False,
        default=safe_json_default,
    )

    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())

    os.replace(tmp_path, path)


def delete_local_checkpoint(path: str) -> None:
    if path and os.path.exists(path):
        os.remove(path)


# ============================================================
# Registry
# ============================================================
def next_version_name(existing_versions: List[dict]) -> str:
    max_num = 0
    for item in existing_versions:
        version = item.get("version", "")
        if isinstance(version, str) and version.startswith("v") and version[1:].isdigit():
            max_num = max(max_num, int(version[1:]))
    return f"v{max_num + 1:04d}"

def strip_s3_scheme(path: str) -> str:
    if is_s3_path(path):
        return path[len("s3://"):]
    return path



def resolve_latest_input_source(
    source_root_dir: str,
    filesystem: Optional[pafs.FileSystem] = None,
    s3_client=None,
) -> Optional[Dict[str, str]]:
    if not source_root_dir:
        print("[WARN] source_root_dir is empty")
        return None

    registry_path = s3_join(source_root_dir, "registry", "version.json")

    # ---------- 1. 安全读取 registry ----------
    try:
        registry = load_json_any(
            registry_path,
            filesystem=filesystem,
            s3_client=s3_client,
        )
    except Exception as e:
        print(f"[WARN] Failed to load registry: {registry_path}")
        print(f"[WARN] Error: {e}")
        return None

    if not registry:
        print(f"[WARN] Empty registry: {registry_path}")
        return None

    # ---------- 2. 找 latest ----------
    latest_version = registry.get("latest_version") or registry.get("latest")
    if not latest_version:
        print(f"[WARN] No latest version in registry: {registry_path}")
        return None

    # ---------- 3. 找对应 entry ----------
    latest_entry = None
    versions = registry.get("versions")

    if isinstance(versions, list):
        for item in versions:
            if item.get("version") == latest_version:
                latest_entry = item
                break
    elif registry.get("version") == latest_version:
        latest_entry = registry

    if latest_entry is None:
        print(f"[WARN] Latest version '{latest_version}' not found in registry")
        return None

    # ---------- 4. 解析路径 ----------
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
        return {
            "source_registry": registry_path,
            "source_version": latest_version,
            "input_source": s3_join(version_dir, "parts"),
            "input_kind": "version_dir_fallback",
        }

    print(f"[WARN] Cannot resolve dataset path from registry: {registry_path}")
    return None

# ============================================================
# Arrow dataset helpers
# ============================================================
def build_arrow_dataset(input_source: str, input_kind: str, filesystem: pafs.FileSystem) -> ds.Dataset:
    input_source = strip_s3_scheme(input_source)
    if input_kind == "parts_prefix":
        return ds.dataset(input_source.rstrip("/") + "/", format="parquet", filesystem=filesystem)

    if input_kind in {"data_parquet", "version_dir_fallback"}:
        if input_source.endswith(".parquet"):
            return ds.dataset(input_source, format="parquet", filesystem=filesystem)
        return ds.dataset(input_source.rstrip("/") + "/", format="parquet", filesystem=filesystem)

    raise ValueError(f"Unsupported input_kind: {input_kind}")


def count_dataset_rows(dataset: ds.Dataset, batch_size: int = DEFAULT_BATCH_SIZE) -> int:
    total = 0
    scanner = dataset.scanner(batch_size=batch_size)
    for batch in scanner.to_batches():
        total += batch.num_rows
    return total


# ============================================================
# Label transforms
# ============================================================
def deterministic_jitter(user_id: Any, movie_id: Any, amplitude: float) -> float:
    key = f"{user_id}:{movie_id}".encode("utf-8")
    h = hashlib.blake2b(key, digest_size=8).digest()
    u = int.from_bytes(h, byteorder="big", signed=False) / float(2**64 - 1)
    return u * (2.0 * amplitude) - amplitude


def transform_batch_offline(
    table: pa.Table,
    user_id_col: str,
    movie_id_col: str,
    user_embedding_col: str,
    movie_embedding_col: str,
    label_col: str,
    sigmoid_k: float,
    sigmoid_c: float,
    jitter_amplitude: float,
    keep_source_label: bool = True,
) -> pa.Table:
    pyd = table.to_pydict()

    user_ids = pyd[user_id_col]
    movie_ids = pyd[movie_id_col]
    raw_values = pyd[label_col]

    label_raw = []
    label_jittered = []
    label_final = []

    for u, m, x in zip(user_ids, movie_ids, raw_values):
        r = float(x)
        j = deterministic_jitter(u, m, jitter_amplitude)
        rj = min(5.0, max(0.5, r + j))
        rn = (rj - 0.5) / 4.5
        y = 1.0 / (1.0 + math.exp(-sigmoid_k * (rn - sigmoid_c)))

        label_raw.append(r)
        label_jittered.append(rj)
        label_final.append(y)

    return pa.table(
        {
            "user_id": pa.array(user_ids),
            "movie_id": pa.array(movie_ids),
            "user_embedding": table[user_embedding_col],
            "movie_embedding": table[movie_embedding_col],
            "label": pa.array(label_final, type=pa.float64()),
        }
    )


def transform_batch_online(
    table: pa.Table,
    user_id_col: str,
    movie_id_col: str,
    user_embedding_col: str,
    movie_embedding_col: str,
    label_col: str,
    play_time_cap: float,
    keep_source_label: bool = True,
) -> pa.Table:
    pyd = table.to_pydict()
    raw_values = pyd[label_col]

    play_time_raw = []
    play_time_capped = []
    label_final = []

    for x in raw_values:
        v = max(0.0, float(x))
        c = min(play_time_cap, v)
        y = min(1.0, math.log1p(c) / math.log1p(play_time_cap))

        play_time_raw.append(v)
        play_time_capped.append(c)
        label_final.append(y)

    pyd = table.to_pydict()
    return pa.table(
        {
            "user_id": pa.array(pyd[user_id_col]),
            "movie_id": pa.array(pyd[movie_id_col]),
            "user_embedding": table[user_embedding_col],
            "movie_embedding": table[movie_embedding_col],
            "label": pa.array(label_final, type=pa.float64()),
        }
    )


# ============================================================
# Split + write
# ============================================================
def slice_table(table: pa.Table, offset: int, length: int) -> pa.Table:
    return table.slice(offset, length) if length > 0 else table.slice(0, 0)


def write_table_to_parquet_dataset(
    table: pa.Table,
    filesystem: pafs.FileSystem,
    out_dir: str,
    part_idx: int,
    s3_client=None,
    max_retries: int = 5,
    base_sleep_seconds: float = 1.0,
) -> str:
    if table.num_rows == 0:
        return ""

    filename = f"part-{part_idx:06d}.parquet"

    if is_s3_path(out_dir):
        if s3_client is None:
            raise ValueError("s3_client is required for S3 parquet write")

        out_path = s3_join(out_dir, filename)
        bucket, key = split_s3_uri(out_path)

        buf = io.BytesIO()
        pq.write_table(table, buf, compression="zstd")
        payload = buf.getvalue()

        upload_bytes_with_retry(
            s3_client=s3_client,
            bucket=bucket,
            key=key,
            payload=payload,
            content_type="application/octet-stream",
            max_retries=max_retries,
            base_sleep_seconds=base_sleep_seconds,
        )
    else:
        ensure_dir(out_dir)
        path = os.path.join(out_dir, filename)
        pq.write_table(table, path, compression="zstd")

    return filename

def build_split_dataset(
    cfg: Dict[str, Any],
    input_source: str,
    input_kind: str,
    output_dir: str,
    dataset_type: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    label_col: str = "label",
    user_id_col: str = "user_id",
    movie_id_col: str = "movie_id",
    user_embedding_col: str = "user_embedding",
    movie_embedding_col: str = "movie_embedding",
    offline_sigmoid_k: float = DEFAULT_OFFLINE_SIGMOID_K,
    offline_sigmoid_c: float = DEFAULT_OFFLINE_SIGMOID_C,
    offline_jitter_amplitude: float = DEFAULT_OFFLINE_JITTER_AMPLITUDE,
    online_play_time_cap: float = DEFAULT_ONLINE_PLAY_TIME_CAP,
    batch_size: int = DEFAULT_BATCH_SIZE,
    target_rows_per_file: int = 100_000,
    keep_source_label: bool = True,
    s3_client=None,
    upload_max_retries: int = 5,
    upload_base_sleep_seconds: float = 1.0,
    checkpoint_path: Optional[str] = None,
    resume_state: Optional[Dict[str, Any]] = None,
    job_name: str = "split_dataset_online_offline_streaming",
) -> Dict[str, int]:
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-8:
        raise ValueError(f"train_ratio + val_ratio + test_ratio must equal 1.0, got {total_ratio}")

    use_s3 = is_s3_path(input_source) or is_s3_path(output_dir)
    fs = build_arrow_s3_filesystem(cfg) if use_s3 else None
    filesystem = fs if fs is not None else pafs.LocalFileSystem()

    dataset = build_arrow_dataset(input_source, input_kind, filesystem=filesystem)

    schema_names = set(dataset.schema.names)
    required_cols = {
        user_id_col,
        movie_id_col,
        user_embedding_col,
        movie_embedding_col,
        label_col,
    }
    missing = sorted(c for c in required_cols if c not in schema_names)
    if missing:
        raise ValueError(f"Missing required columns in input dataset: {missing}. Available: {sorted(schema_names)}")

    total_rows = count_dataset_rows(dataset, batch_size=batch_size)
    train_cutoff = int(total_rows * train_ratio)
    val_cutoff = int(total_rows * (train_ratio + val_ratio))

    resume_row = 0
    start_batch_idx = 0

    train_rows = 0
    val_rows = 0
    test_rows = 0

    train_parts = 0
    val_parts = 0
    test_parts = 0

    if resume_state:
        progress = resume_state.get("progress", {})
        meta = resume_state.get("meta", {})
        output_meta = resume_state.get("output", {})
        last_write = resume_state.get("last_write", {}) or {}

        # safety checks to prevent resuming into a different dataset or with different split config
        ckpt_total_rows = int(meta.get("total_rows", total_rows))
        ckpt_train_cutoff = int(meta.get("train_cutoff", train_cutoff))
        ckpt_val_cutoff = int(meta.get("val_cutoff", val_cutoff))

        if ckpt_total_rows != total_rows:
            raise RuntimeError(
                f"Checkpoint total_rows mismatch: ckpt={ckpt_total_rows}, current={total_rows}"
            )
        if ckpt_train_cutoff != train_cutoff or ckpt_val_cutoff != val_cutoff:
            raise RuntimeError(
                "Checkpoint split cutoffs mismatch, current config may have changed"
            )

        if output_meta.get("version_dir") != output_dir:
            raise RuntimeError(
                f"Checkpoint version_dir mismatch: ckpt={output_meta.get('version_dir')}, current={output_dir}"
            )

        resume_row = int(progress.get("resume_row", 0))
        start_batch_idx = int(progress.get("batch_idx", 0))

        train_rows = int(progress.get("train_rows", 0))
        val_rows = int(progress.get("val_rows", 0))
        test_rows = int(progress.get("test_rows", 0))

        train_parts = int(progress.get("train_parts", 0))
        val_parts = int(progress.get("val_parts", 0))
        test_parts = int(progress.get("test_parts", 0))

        # if last time failed during writing a part file, we should re-write that part file (overwrite) 
        print(
            f"[RESUME] checkpoint found: "
            f"resume_row={resume_row}, batch_idx={start_batch_idx}, "
            f"train_parts={train_parts}, val_parts={val_parts}, test_parts={test_parts}, "
            f"last_write_status={last_write.get('status')}, last_write_file={last_write.get('file')}"
        )

    train_dir = s3_join(output_dir, "train")
    val_dir = s3_join(output_dir, "val")
    test_dir = s3_join(output_dir, "test")

    scanner = dataset.scanner(
        columns=[
            user_id_col,
            movie_id_col,
            user_embedding_col,
            movie_embedding_col,
            label_col,
        ],
        batch_size=batch_size,
    )

    seen = 0
    idx = 0


    train_buffer = []
    train_buffer_rows = 0
    train_buffer_start_row = None

    val_buffer = []
    val_buffer_rows = 0
    val_buffer_start_row = None

    test_buffer = []
    test_buffer_rows = 0
    test_buffer_start_row = None

    def compute_resume_row(current_seen: int) -> int:
        candidates = []
        if train_buffer_start_row is not None:
            candidates.append(train_buffer_start_row)
        if val_buffer_start_row is not None:
            candidates.append(val_buffer_start_row)
        if test_buffer_start_row is not None:
            candidates.append(test_buffer_start_row)
        if candidates:
            return min(candidates)
        return current_seen

    def persist_checkpoint(
        current_seen: int,
        current_batch_idx: int,
        last_write_split: Optional[str] = None,
        last_write_part_id: Optional[int] = None,
        last_write_file: Optional[str] = None,
        last_write_status: Optional[str] = None,
        last_write_error: Optional[str] = None,
    ) -> None:
        if not checkpoint_path:
            return

        ckpt = build_checkpoint_payload(
            job_name=job_name,
            input_source=input_source,
            input_kind=input_kind,
            version_dir=output_dir,
            dataset_type=dataset_type,
            resume_row=compute_resume_row(current_seen),
            batch_idx=current_batch_idx,
            train_rows=train_rows,
            val_rows=val_rows,
            test_rows=test_rows,
            train_parts=train_parts,
            val_parts=val_parts,
            test_parts=test_parts,
            train_buffer_start_row=train_buffer_start_row,
            val_buffer_start_row=val_buffer_start_row,
            test_buffer_start_row=test_buffer_start_row,
            total_rows=total_rows,
            train_cutoff=train_cutoff,
            val_cutoff=val_cutoff,
            last_write_split=last_write_split,
            last_write_part_id=last_write_part_id,
            last_write_file=last_write_file,
            last_write_status=last_write_status,
            last_write_error=last_write_error,
        )
        save_local_checkpoint(checkpoint_path, ckpt)

    for batch in scanner.to_batches():
        idx += 1
        table = pa.Table.from_batches([batch])

        batch_rows_raw = table.num_rows
        batch_start_raw = seen
        batch_end_raw = seen + batch_rows_raw

        # if entire batch is before resume_row, skip it
        if batch_end_raw <= resume_row:
            seen += batch_rows_raw
            if idx % 10 == 0:
                print(f"[RESUME] skipping batch {idx}, seen={seen}/{total_rows}")
            continue

        # if batch overlaps with resume_row, slice off the already processed part
        if batch_start_raw < resume_row < batch_end_raw:
            skip_rows = resume_row - batch_start_raw
            table = slice_table(table, skip_rows, batch_rows_raw - skip_rows)

        if dataset_type == "offline":
            table = transform_batch_offline(
                table=table,
                user_id_col=user_id_col,
                movie_id_col=movie_id_col,
                user_embedding_col=user_embedding_col,
                movie_embedding_col=movie_embedding_col,
                label_col=label_col,
                sigmoid_k=offline_sigmoid_k,
                sigmoid_c=offline_sigmoid_c,
                jitter_amplitude=offline_jitter_amplitude,
                keep_source_label=keep_source_label,
            )
        elif dataset_type == "online":
            table = transform_batch_online(
                table=table,
                user_id_col=user_id_col,
                movie_id_col=movie_id_col,
                user_embedding_col=user_embedding_col,
                movie_embedding_col=movie_embedding_col,
                label_col=label_col,
                play_time_cap=online_play_time_cap,
                keep_source_label=keep_source_label,
            )
        else:
            raise ValueError(f"Unsupported dataset_type: {dataset_type}")

        batch_rows = table.num_rows
        batch_start = max(batch_start_raw, resume_row)
        batch_end = batch_start + batch_rows

        train_len = max(0, min(batch_end, train_cutoff) - batch_start)
        val_len = max(0, min(batch_end, val_cutoff) - max(batch_start, train_cutoff))
        test_len = max(0, batch_end - max(batch_start, val_cutoff))

        cursor = 0
        

        if train_len > 0:
            t = slice_table(table, cursor, train_len)
            if train_buffer_rows == 0:
                train_buffer_start_row = batch_start + cursor
            train_buffer.append(t)
            train_buffer_rows += t.num_rows
            train_rows += t.num_rows
            cursor += train_len
        if train_buffer_rows >= target_rows_per_file:
            merged = pa.concat_tables(train_buffer)
            part_id = train_parts
            filename = f"part-{part_id:06d}.parquet"

            persist_checkpoint(
                batch_start,
                idx,
                last_write_split="train",
                last_write_part_id=part_id,
                last_write_file=filename,
                last_write_status="writing",
                last_write_error=None,
            )

            try:
                written_file = write_table_to_parquet_dataset(
                    merged,
                    filesystem,
                    train_dir,
                    part_id,
                    s3_client=s3_client,
                    max_retries=upload_max_retries,
                    base_sleep_seconds=upload_base_sleep_seconds,
                )
            except Exception as e:
                persist_checkpoint(
                    batch_start,
                    idx,
                    last_write_split="train",
                    last_write_part_id=part_id,
                    last_write_file=filename,
                    last_write_status="failed",
                    last_write_error=str(e),
                )
                raise

            del merged
            train_parts += 1
            train_buffer = []
            train_buffer_rows = 0
            train_buffer_start_row = None

            persist_checkpoint(
                batch_start,
                idx,
                last_write_split="train",
                last_write_part_id=part_id,
                last_write_file=written_file,
                last_write_status="success",
                last_write_error=None,
            )
            

        if val_len > 0:
            v = slice_table(table, cursor, val_len)
            if val_buffer_rows == 0:
                val_buffer_start_row = batch_start + cursor
            val_buffer.append(v)
            val_buffer_rows += v.num_rows
            val_rows += v.num_rows
            cursor += val_len          
        if val_buffer_rows >= target_rows_per_file:
            merged = pa.concat_tables(val_buffer)
            part_id = val_parts
            filename = f"part-{part_id:06d}.parquet"

            persist_checkpoint(
                batch_start,
                idx,
                last_write_split="val",
                last_write_part_id=part_id,
                last_write_file=filename,
                last_write_status="writing",
                last_write_error=None,
            )

            try:
                written_file = write_table_to_parquet_dataset(
                    merged,
                    filesystem,
                    val_dir,
                    part_id,
                    s3_client=s3_client,
                    max_retries=upload_max_retries,
                    base_sleep_seconds=upload_base_sleep_seconds,
                )
            except Exception as e:
                persist_checkpoint(
                    batch_start,
                    idx,
                    last_write_split="val",
                    last_write_part_id=part_id,
                    last_write_file=filename,
                    last_write_status="failed",
                    last_write_error=str(e),
                )
                raise

            del merged
            val_parts += 1
            val_buffer = []
            val_buffer_rows = 0
            val_buffer_start_row = None

            persist_checkpoint(
                batch_start,
                idx,
                last_write_split="val",
                last_write_part_id=part_id,
                last_write_file=written_file,
                last_write_status="success",
                last_write_error=None,
            )


        if test_len > 0:
            te = slice_table(table, cursor, test_len)
            if test_buffer_rows == 0:
                test_buffer_start_row = batch_start + cursor
            test_buffer.append(te)
            test_buffer_rows += te.num_rows
            test_rows += te.num_rows
            cursor += test_len       
        if test_buffer_rows >= target_rows_per_file:
            merged = pa.concat_tables(test_buffer)
            part_id = test_parts
            filename = f"part-{part_id:06d}.parquet"

            persist_checkpoint(
                batch_start,
                idx,
                last_write_split="test",
                last_write_part_id=part_id,
                last_write_file=filename,
                last_write_status="writing",
                last_write_error=None,
            )

            try:
                written_file = write_table_to_parquet_dataset(
                    merged,
                    filesystem,
                    test_dir,
                    part_id,
                    s3_client=s3_client,
                    max_retries=upload_max_retries,
                    base_sleep_seconds=upload_base_sleep_seconds,
                )
            except Exception as e:
                persist_checkpoint(
                    batch_start,
                    idx,
                    last_write_split="test",
                    last_write_part_id=part_id,
                    last_write_file=filename,
                    last_write_status="failed",
                    last_write_error=str(e),
                )
                raise

            del merged
            test_parts += 1
            test_buffer = []
            test_buffer_rows = 0
            test_buffer_start_row = None

            persist_checkpoint(
                batch_start,
                idx,
                last_write_split="test",
                last_write_part_id=part_id,
                last_write_file=written_file,
                last_write_status="success",
                last_write_error=None,
            )

        import gc
        gc.collect()
        seen = batch_end
        persist_checkpoint(
            seen,
            idx,
            last_write_split=None,
            last_write_part_id=None,
            last_write_file=None,
            last_write_status=None,
            last_write_error=None,
        )

        if idx % 10 == 0 or seen >= total_rows:
            print(
                f"Processed batch {idx}: total={seen}/{total_rows}, "
                f"train={train_rows}, val={val_rows}, test={test_rows}"
            )
    if train_buffer_rows > 0:
        merged = pa.concat_tables(train_buffer)
        part_id = train_parts
        filename = f"part-{part_id:06d}.parquet"

        persist_checkpoint(
            seen,
            idx,
            last_write_split="train",
            last_write_part_id=part_id,
            last_write_file=filename,
            last_write_status="writing",
            last_write_error=None,
        )

        try:
            written_file = write_table_to_parquet_dataset(
                merged,
                filesystem,
                train_dir,
                part_id,
                s3_client=s3_client,
                max_retries=upload_max_retries,
                base_sleep_seconds=upload_base_sleep_seconds,
            )
        except Exception as e:
            persist_checkpoint(
                seen,
                idx,
                last_write_split="train",
                last_write_part_id=part_id,
                last_write_file=filename,
                last_write_status="failed",
                last_write_error=str(e),
            )
            raise

        train_parts += 1
        train_buffer = []
        train_buffer_rows = 0
        train_buffer_start_row = None
        del merged

        persist_checkpoint(
            seen,
            idx,
            last_write_split="train",
            last_write_part_id=part_id,
            last_write_file=written_file,
            last_write_status="success",
            last_write_error=None,
        )

    if val_buffer_rows > 0:
        merged = pa.concat_tables(val_buffer)
        part_id = val_parts
        filename = f"part-{part_id:06d}.parquet"

        persist_checkpoint(
            seen,
            idx,
            last_write_split="val",
            last_write_part_id=part_id,
            last_write_file=filename,
            last_write_status="writing",
            last_write_error=None,
        )

        try:
            written_file = write_table_to_parquet_dataset(
                merged,
                filesystem,
                val_dir,
                part_id,
                s3_client=s3_client,
                max_retries=upload_max_retries,
                base_sleep_seconds=upload_base_sleep_seconds,
            )
        except Exception as e:
            persist_checkpoint(
                seen,
                idx,
                last_write_split="val",
                last_write_part_id=part_id,
                last_write_file=filename,
                last_write_status="failed",
                last_write_error=str(e),
            )
            raise

        val_parts += 1
        val_buffer = []
        val_buffer_rows = 0
        val_buffer_start_row = None
        del merged

        persist_checkpoint(
            seen,
            idx,
            last_write_split="val",
            last_write_part_id=part_id,
            last_write_file=written_file,
            last_write_status="success",
            last_write_error=None,
        )

    if test_buffer_rows > 0:
        merged = pa.concat_tables(test_buffer)
        part_id = test_parts
        filename = f"part-{part_id:06d}.parquet"

        persist_checkpoint(
            seen,
            idx,
            last_write_split="test",
            last_write_part_id=part_id,
            last_write_file=filename,
            last_write_status="writing",
            last_write_error=None,
        )

        try:
            written_file = write_table_to_parquet_dataset(
                merged,
                filesystem,
                test_dir,
                part_id,
                s3_client=s3_client,
                max_retries=upload_max_retries,
                base_sleep_seconds=upload_base_sleep_seconds,
            )
        except Exception as e:
            persist_checkpoint(
                seen,
                idx,
                last_write_split="test",
                last_write_part_id=part_id,
                last_write_file=filename,
                last_write_status="failed",
                last_write_error=str(e),
            )
            raise

        test_parts += 1
        test_buffer = []
        test_buffer_rows = 0
        test_buffer_start_row = None
        del merged

        persist_checkpoint(
            seen,
            idx,
            last_write_split="test",
            last_write_part_id=part_id,
            last_write_file=written_file,
            last_write_status="success",
            last_write_error=None,
        )
        

    if checkpoint_path:
        delete_local_checkpoint(checkpoint_path)
    
    return {
        "train_rows": train_rows,
        "val_rows": val_rows,
        "test_rows": test_rows,
        "total_rows": train_rows + val_rows + test_rows,
        "train_parts": train_parts,
        "val_parts": val_parts,
        "test_parts": test_parts,
    }


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)

    job_name = deep_get(cfg, ["job", "name"], "split_dataset_online_offline_streaming")
    created_at = datetime.now(timezone.utc).isoformat()

    dataset_type = deep_get(cfg, ["dataset", "type"], "offline")
    if dataset_type not in {"offline", "online"}:
        raise ValueError("dataset.type must be either 'offline' or 'online'")

    input_override_source = deep_get(cfg, ["input", "source"])
    input_override_kind = deep_get(cfg, ["input", "source_kind"])

    upload_max_retries = int(deep_get(cfg, ["runtime", "upload_max_retries"], 5))
    upload_base_sleep_seconds = float(deep_get(cfg, ["runtime", "upload_base_sleep_seconds"], 1.0))

    output_root = deep_get(cfg, ["output", "root_dir"])
    require_value("output.root_dir", output_root)

    use_s3_for_meta = is_s3_path(output_root)
    s3_client = build_boto3_s3_client(cfg) if use_s3_for_meta else None
    fs = build_arrow_s3_filesystem(cfg) if use_s3_for_meta else None

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
        source_root_dir = deep_get(cfg, ["input", f"{dataset_type}_root_dir"])
        require_value(f"input.{dataset_type}_root_dir", source_root_dir)

        input_is_s3 = is_s3_path(source_root_dir)
        input_fs = build_arrow_s3_filesystem(cfg) if input_is_s3 else None
        input_s3_client = build_boto3_s3_client(cfg) if input_is_s3 else None

        input_info = resolve_latest_input_source(
            source_root_dir,
            filesystem=input_fs,
            s3_client=input_s3_client,
        )

    split_cfg = deep_get(cfg, ["split"], {}) or {}
    train_ratio = float(split_cfg.get("train_ratio", 0.7))
    val_ratio = float(split_cfg.get("val_ratio", 0.1))
    test_ratio = float(split_cfg.get("test_ratio", 0.2))

    columns_cfg = deep_get(cfg, ["columns"], {}) or {}
    label_col = columns_cfg.get("label", "label")
    user_id_col = columns_cfg.get("user_id", "user_id")
    movie_id_col = columns_cfg.get("movie_id", "movie_id")
    user_embedding_col = columns_cfg.get("user_embedding", "user_embedding")
    movie_embedding_col = columns_cfg.get("movie_embedding", "movie_embedding")

    offline_k = float(deep_get(cfg, ["label_transform", "offline", "sigmoid_k"], DEFAULT_OFFLINE_SIGMOID_K))
    offline_c = float(deep_get(cfg, ["label_transform", "offline", "sigmoid_c"], DEFAULT_OFFLINE_SIGMOID_C))
    offline_jitter = float(deep_get(cfg, ["label_transform", "offline", "jitter_amplitude"], DEFAULT_OFFLINE_JITTER_AMPLITUDE))
    online_cap = float(deep_get(cfg, ["label_transform", "online", "play_time_cap"], DEFAULT_ONLINE_PLAY_TIME_CAP))
    batch_size = int(deep_get(cfg, ["runtime", "batch_size"], DEFAULT_BATCH_SIZE))
    target_rows_per_file = int(deep_get(cfg, ["runtime", "target_rows_per_file"], 100_000))
    checkpoint_path = deep_get(
    cfg, ["runtime", "local_checkpoint_path"], "scripts/checkpoints/split_dataset_checkpoint.json"
)
    keep_source_label = bool(deep_get(cfg, ["label_transform", "keep_source_label"], True))

    dataset_root = s3_join(output_root)
    registry_path = s3_join(dataset_root, "registry", "version.json")
    latest_dir = s3_join(dataset_root, "latest")

    try:
        registry = load_json_any(registry_path, filesystem=fs, s3_client=s3_client)
    except Exception as e:
        print(f"Error occurred while loading registry: {e}")
        registry = {"versions": []}

    existing_versions = registry.get("versions", []) if isinstance(registry.get("versions"), list) else []
    resume_state = load_local_checkpoint(checkpoint_path)

    if resume_state:
        output_meta = resume_state.get("output", {}) or {}
        input_meta = resume_state.get("input", {}) or {}

        version_dir = output_meta.get("version_dir")
        ckpt_dataset_type = output_meta.get("dataset_type")
        ckpt_input_source = input_meta.get("input_source")
        ckpt_input_kind = input_meta.get("input_kind")

        # if not version_dir:
        #     raise ValueError("Checkpoint missing output.version_dir")

        # if ckpt_dataset_type and ckpt_dataset_type != dataset_type:
        #     raise ValueError(
        #         f"Checkpoint dataset_type mismatch: ckpt={ckpt_dataset_type}, current={dataset_type}"
        #     )

        # if ckpt_input_source and ckpt_input_source != input_info["input_source"]:
        #     raise ValueError(
        #         f"Checkpoint input_source mismatch: "
        #         f"ckpt={ckpt_input_source}, current={input_info['input_source']}"
        #     )

        # if ckpt_input_kind and ckpt_input_kind != input_info["input_kind"]:
        #     raise ValueError(
        #         f"Checkpoint input_kind mismatch: "
        #         f"ckpt={ckpt_input_kind}, current={input_info['input_kind']}"
        #     )

        version_name = version_dir.rstrip("/").split("/")[-1]
        print(f"[RESUME] resume from existing version_dir={version_dir}")
    else:
        version_name = next_version_name(existing_versions)
        version_dir = s3_join(dataset_root, "versions", version_name)
        print(f"[START] start new transfer version_dir={version_dir}")

    stats = build_split_dataset(
        cfg=cfg,
        input_source=input_info["input_source"],
        input_kind=input_info["input_kind"],
        output_dir=version_dir,
        dataset_type=dataset_type,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        label_col=label_col,
        user_id_col=user_id_col,
        movie_id_col=movie_id_col,
        user_embedding_col=user_embedding_col,
        movie_embedding_col=movie_embedding_col,
        offline_sigmoid_k=offline_k,
        offline_sigmoid_c=offline_c,
        offline_jitter_amplitude=offline_jitter,
        online_play_time_cap=online_cap,
        batch_size=batch_size,
        target_rows_per_file=target_rows_per_file,
        keep_source_label=keep_source_label,
        s3_client=s3_client,
        upload_max_retries=upload_max_retries,
        upload_base_sleep_seconds=upload_base_sleep_seconds,
        checkpoint_path=checkpoint_path,
        resume_state=resume_state,
        job_name=job_name,
    )

    manifest = {
        "job_name": job_name,
        "created_at": created_at,
        "dataset_type": dataset_type,
        "version": version_name,
        "input": {
            "source_root_dir": source_root_dir,
            "source_registry": input_info.get("source_registry"),
            "source_version": input_info.get("source_version"),
            "input_source": input_info["input_source"],
            "input_kind": input_info["input_kind"],
        },
        "split": {
            "strategy": "file_order_ratio",
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "batch_size": batch_size,
        },
        "label_transform": {
            "input_label_column": label_col,
            "keep_source_label": keep_source_label,
            "offline": {
                "strategy": "label_jitter_sigmoid",
                "sigmoid_k": offline_k,
                "sigmoid_c": offline_c,
                "jitter_amplitude": offline_jitter,
            },
            "online": {
                "strategy": "play_time_log1p",
                "play_time_cap": online_cap,
            },
        },
        "output": {
            "root_dir": output_root,
            "dataset_root": dataset_root,
            "version_dir": version_dir,
            "latest_dir": latest_dir,
            "train_dataset": s3_join(version_dir, "train"),
            "val_dataset": s3_join(version_dir, "val"),
            "test_dataset": s3_join(version_dir, "test"),
        },
        "schema_note": {
            "index_fields": [user_id_col, movie_id_col],
            "feature_fields": [user_embedding_col, movie_embedding_col],
            "label_field": "label",
        },
        "stats": stats,
    }

    manifest_path = s3_join(version_dir, "manifest.json")
    latest_manifest_path = s3_join(latest_dir, "manifest.json")
    save_json_any(manifest_path, manifest, filesystem=fs, s3_client=s3_client)
    save_json_any(latest_manifest_path, manifest, filesystem=fs, s3_client=s3_client)

    registry_entry = {
        "version": version_name,
        "created_at": created_at,
        "dataset_type": dataset_type,
        "source_version": input_info.get("source_version"),
        "version_dir": version_dir,
        "manifest": manifest_path,
        "data_parts_prefix": version_dir,
        "row_count": stats["total_rows"],
    }

    registry["versions"] = existing_versions

    replaced = False
    for i, item in enumerate(registry["versions"]):
        if item.get("version") == version_name:
            registry["versions"][i] = registry_entry
            replaced = True
            break

    if not replaced:
        registry["versions"].append(registry_entry)

    registry["latest"] = version_name
    save_json_any(registry_path, registry, filesystem=fs, s3_client=s3_client)

    print(f"[{job_name}] done")
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
