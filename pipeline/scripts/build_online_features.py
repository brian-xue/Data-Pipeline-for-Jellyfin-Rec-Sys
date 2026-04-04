from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import s3fs
import yaml


INPUT_VERSION_PATTERNS = ("%y%m%d%H%M%S", "%Y%m%d%H%M%S")
EMBEDDING_VERSION_PATTERNS = ("%y-%m-%d-%H-%M", "%Y-%m-%d-%H-%M")
DEFAULT_VERSION_FORMAT = "%y%m%d%H%M%S"


def load_yaml_config(path: str) -> dict[str, Any]:
	with open(path, "r", encoding="utf-8") as handle:
		return yaml.safe_load(handle) or {}


def deep_get(data: dict[str, Any], keys: list[str], default: Any = None) -> Any:
	current: Any = data
	for key in keys:
		if not isinstance(current, dict) or key not in current:
			return default
		current = current[key]
	return current


def is_s3_uri(path: str) -> bool:
	return isinstance(path, str) and path.startswith("s3://")


def split_s3_uri(uri: str) -> tuple[str, str]:
	tail = uri[len("s3://") :]
	bucket, _, key = tail.partition("/")
	return bucket, key


def s3_to_key(uri: str) -> str:
	bucket, key = split_s3_uri(uri)
	return f"{bucket}/{key}".rstrip("/")


def parse_version(value: str, patterns: tuple[str, ...]) -> datetime | None:
	for pattern in patterns:
		try:
			return datetime.strptime(value, pattern)
		except ValueError:
			continue
	return None


def parse_version_index(version: str) -> int | None:
	if not isinstance(version, str):
		return None
	try:
		return int(version)
	except ValueError:
		return None


def make_next_version(existing_versions: list[dict[str, Any]], version_format: str) -> str:
	try:
		return datetime.now(timezone.utc).strftime(version_format)
	except Exception:
		return datetime.now(timezone.utc).strftime("%y%m%d%H%M%S")


def normalize_path_config(raw: Any, label: str) -> str:
	if isinstance(raw, str) and raw.strip():
		return raw.strip()

	if isinstance(raw, dict):
		for key in ("path", "uri", "root", "dir", "value"):
			value = raw.get(key)
			if isinstance(value, str) and value.strip():
				return value.strip()

		if len(raw) == 1:
			key = next(iter(raw.keys()))
			if isinstance(key, str) and key.strip():
				return key.strip()

	raise ValueError(f"Invalid path config for {label}: {raw}")


@dataclass
class IOContext:
	s3: s3fs.S3FileSystem | None

	def exists(self, path: str) -> bool:
		if is_s3_uri(path):
			if self.s3 is None:
				return False
			return self.s3.exists(s3_to_key(path))
		return os.path.exists(path)

	def makedirs(self, path: str) -> None:
		if is_s3_uri(path):
			if self.s3 is None:
				raise ValueError("S3 filesystem is not configured")
			self.s3.makedirs(s3_to_key(path), exist_ok=True)
			return
		os.makedirs(path, exist_ok=True)

	def list_dirs(self, path: str) -> list[str]:
		if is_s3_uri(path):
			if self.s3 is None or not self.s3.exists(s3_to_key(path)):
				return []
			infos = self.s3.ls(s3_to_key(path), detail=True)
			out: list[str] = []
			for info in infos:
				if info.get("type") == "directory":
					out.append(f"s3://{info['name']}")
			return out

		if not os.path.isdir(path):
			return []
		return [os.path.join(path, name) for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]

	def list_parquet_recursive(self, path: str) -> list[str]:
		if is_s3_uri(path):
			if self.s3 is None or not self.s3.exists(s3_to_key(path)):
				return []
			files = self.s3.find(s3_to_key(path))
			return [f"s3://{item}" for item in files if item.endswith(".parquet")]

		if not os.path.exists(path):
			return []
		out: list[str] = []
		for root, _, files in os.walk(path):
			for name in files:
				if name.endswith(".parquet"):
					out.append(os.path.join(root, name))
		return out

	def read_json(self, path: str, default: dict[str, Any]) -> dict[str, Any]:
		if not self.exists(path):
			return default
		if is_s3_uri(path):
			if self.s3 is None:
				return default
			with self.s3.open(s3_to_key(path), "rb") as handle:
				return json.loads(handle.read().decode("utf-8"))
		with open(path, "r", encoding="utf-8") as handle:
			return json.load(handle)

	def write_json(self, path: str, payload: dict[str, Any]) -> None:
		data = json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8")
		parent = parent_dir(path)
		if parent:
			self.makedirs(parent)

		if is_s3_uri(path):
			if self.s3 is None:
				raise ValueError("S3 filesystem is not configured")
			with self.s3.open(s3_to_key(path), "wb") as handle:
				handle.write(data)
			return

		with open(path, "wb") as handle:
			handle.write(data)

	def open_parquet_file(self, path: str) -> pq.ParquetFile:
		if is_s3_uri(path):
			if self.s3 is None:
				raise ValueError("S3 filesystem is not configured")
			return pq.ParquetFile(self.s3.open(s3_to_key(path), "rb"))
		return pq.ParquetFile(path)

	def write_parquet_table(self, path: str, table: pa.Table) -> None:
		parent = parent_dir(path)
		if parent:
			self.makedirs(parent)

		if is_s3_uri(path):
			if self.s3 is None:
				raise ValueError("S3 filesystem is not configured")
			with self.s3.open(s3_to_key(path), "wb") as handle:
				pq.write_table(table, handle)
			return

		pq.write_table(table, path)


def parent_dir(path: str) -> str:
	if is_s3_uri(path):
		bucket, key = split_s3_uri(path)
		if not key or "/" not in key:
			return f"s3://{bucket}"
		return f"s3://{bucket}/{key.rsplit('/', 1)[0]}"
	return os.path.dirname(path)


def path_join(base: str, *parts: str) -> str:
	cleaned_parts = [part.strip("/") for part in parts if part]
	if is_s3_uri(base):
		out = base.rstrip("/")
		for part in cleaned_parts:
			out += f"/{part}"
		return out
	return os.path.join(base, *cleaned_parts)


def basename(path: str) -> str:
	if is_s3_uri(path):
		_, key = split_s3_uri(path)
		return key.rstrip("/").rsplit("/", 1)[-1]
	return os.path.basename(path.rstrip("/"))


def configure_environment(cfg: dict[str, Any]) -> None:
	env_cfg = deep_get(cfg, ["environment"], {}) or {}
	for key, value in env_cfg.items():
		if value is not None:
			os.environ[str(key)] = str(value)


def build_s3_filesystem() -> s3fs.S3FileSystem:
	endpoint = os.getenv("MINIO_ENDPOINT")
	access_key = os.getenv("AWS_ACCESS_KEY_ID")
	secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
	region = os.getenv("AWS_REGION", "us-east-1")

	if not endpoint:
		raise ValueError("MINIO_ENDPOINT is required for S3 paths")
	if not access_key or not secret_key:
		raise ValueError("AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY are required for S3 paths")

	return s3fs.S3FileSystem(
		key=access_key,
		secret=secret_key,
		client_kwargs={"endpoint_url": endpoint, "region_name": region},
	)


def discover_input_versions(io_ctx: IOContext, input_root: str) -> list[tuple[str, datetime]]:
    versions_root = path_join(input_root, "versions")
    print(f"[online build] discovering input versions in: {versions_root}")
    items = io_ctx.list_dirs(versions_root)
    out: list[tuple[str, datetime]] = []
    for item in items:
        version = basename(item)
        parsed = parse_version(version, INPUT_VERSION_PATTERNS)
        if parsed is None:
            continue
        out.append((version, parsed))
    out.sort(key=lambda pair: pair[1])
    return out


def discover_embedding_versions(io_ctx: IOContext, embedding_root: str) -> list[tuple[str, datetime]]:
	versions_root = path_join(embedding_root, "versions")
	items = io_ctx.list_dirs(versions_root)
	out: list[tuple[str, datetime]] = []
	for item in items:
		version = basename(item)
		parsed = parse_version(version, EMBEDDING_VERSION_PATTERNS)
		if parsed is None:
			continue
		out.append((version, parsed))
	out.sort(key=lambda pair: pair[1])
	return out


def normalize_embedding(value: Any) -> list[float] | None:
	if value is None:
		return None
	if isinstance(value, list):
		try:
			return [float(x) for x in value]
		except Exception:
			return None
	if isinstance(value, tuple):
		try:
			return [float(x) for x in value]
		except Exception:
			return None
	if isinstance(value, str):
		text = value.strip()
		if not text:
			return None
		try:
			parsed = json.loads(text)
		except Exception:
			return None
		if isinstance(parsed, list):
			try:
				return [float(x) for x in parsed]
			except Exception:
				return None
	return None


def load_movie_embeddings(io_ctx: IOContext, movie_embedding_path: str) -> dict[int, list[float]]:
    parquet_file = io_ctx.open_parquet_file(movie_embedding_path)
    mapping: dict[int, list[float]] = {}

    for batch in parquet_file.iter_batches(batch_size=100_000):
        table = pa.Table.from_batches([batch])
        columns = set(table.column_names)

        movie_id_col = None
        for candidate in ("movie_id", "movieId", "movieID"):
            if candidate in columns:
                movie_id_col = candidate
                break

        if movie_id_col is None:
            raise ValueError(
                f"movie embedding parquet must contain movie id column, "
                f"candidates: movie_id/movieId/movieID, actual: {sorted(columns)}"
            )

        embedding_col = None
        for candidate in ("movie_embedding", "embedding", "vector"):
            if candidate in columns:
                embedding_col = candidate
                break

        if embedding_col is None:
            raise ValueError(
                f"movie embedding parquet missing embedding column "
                f"(movie_embedding/embedding/vector), actual: {sorted(columns)}"
            )

        movie_ids = table[movie_id_col].to_pylist()
        vectors = table[embedding_col].to_pylist()

        for movie_id, vector in zip(movie_ids, vectors):
            try:
                mid = int(movie_id)
            except Exception:
                continue 

            normalized = normalize_embedding(vector)
            if normalized is None:
                continue

            mapping[mid] = normalized

    return mapping


def load_user_embedding_version(
	io_ctx: IOContext,
	embedding_root: str,
	version: str,
) -> dict[int, list[float]]:
	version_dir = path_join(embedding_root, "versions", version)
	parquet_files = io_ctx.list_parquet_recursive(version_dir)
	out: dict[int, list[float]] = {}
	for file_path in parquet_files:
		parquet_file = io_ctx.open_parquet_file(file_path)
		for batch in parquet_file.iter_batches(batch_size=100_000):
			table = pa.Table.from_batches([batch])
			columns = set(table.column_names)
			if "user_id" not in columns:
				continue
			embedding_col = None
			for candidate in ("user_embedding", "embedding", "long_term_embedding"):
				if candidate in columns:
					embedding_col = candidate
					break
			if embedding_col is None:
				continue

			user_ids = table["user_id"].to_pylist()
			vectors = table[embedding_col].to_pylist()
			for user_id, vector in zip(user_ids, vectors):
				normalized = normalize_embedding(vector)
				if normalized is not None:
					out[int(user_id)] = normalized
	return out


def gather_input_user_event_files(io_ctx: IOContext, input_root: str, input_version: str) -> list[str]:
	base = path_join(input_root, "versions", input_version)
	candidates = [path_join(base, "user_event"), path_join(base, "user_events")]
	out: list[str] = []
	for candidate in candidates:
		out.extend(io_ctx.list_parquet_recursive(candidate))
	return sorted(set(out))


def scan_file_user_distribution(io_ctx: IOContext, parquet_path: str) -> tuple[int, dict[int, int]]:
	parquet_file = io_ctx.open_parquet_file(parquet_path)
	total_rows = 0
	per_user: dict[int, int] = {}

	for batch in parquet_file.iter_batches(batch_size=100_000, columns=["user_id"]):
		user_ids = batch.column(0).to_pylist()
		total_rows += len(user_ids)
		for user_id in user_ids:
			uid = int(user_id)
			per_user[uid] = per_user.get(uid, 0) + 1

	return total_rows, per_user


class StreamingParquetSink:
	def __init__(self, io_ctx: IOContext, output_dir: str, batch_size: int):
		self.io_ctx = io_ctx
		self.output_dir = output_dir
		self.batch_size = max(1, int(batch_size))
		self.buffer: list[dict[str, Any]] = []
		self.part_id = 0
		self.total_rows = 0
		self.paths: list[str] = []

	def _flush(self) -> None:
		if not self.buffer:
			return
		target = path_join(self.output_dir, "parts", f"part_{self.part_id + 1:05d}.parquet")
		table = pa.Table.from_pylist(self.buffer)
		self.io_ctx.write_parquet_table(target, table)
		self.paths.append(target)
		self.total_rows += len(self.buffer)
		self.buffer = []
		self.part_id += 1

	def write_rows(self, rows: list[dict[str, Any]]) -> None:
		if not rows:
			return
		self.buffer.extend(rows)
		if len(self.buffer) >= self.batch_size:
			self._flush()

	def close(self) -> None:
		self._flush()


def build_online_positive_samples(config_path: str) -> dict[str, Any]:
	cfg = load_yaml_config(config_path)
	configure_environment(cfg)

	input_root = normalize_path_config(deep_get(cfg, ["input"]), "input")
	embedding_root = normalize_path_config(deep_get(cfg, ["embedding_path"]), "embedding_path")
	movie_embedding_path = normalize_path_config(deep_get(cfg, ["movie_embedding_path"]), "movie_embedding_path")
	warehouse_dir = normalize_path_config(deep_get(cfg, ["output", "warehouse_dir"]), "output.warehouse_dir")

	batch_size = int(deep_get(cfg, ["runtime", "profile_batch_size"], 100_000))
	write_parquet = bool(deep_get(cfg, ["output", "write_parquet"], True))
	coverage_threshold = float(deep_get(cfg, ["runtime", "user_embedding_coverage_threshold"], 0.8))
	version_format = str(deep_get(cfg, ["output", "version_format"], DEFAULT_VERSION_FORMAT))

	needs_s3 = any(is_s3_uri(path) for path in (input_root, embedding_root, movie_embedding_path, warehouse_dir))
	s3_client = build_s3_filesystem() if needs_s3 else None
	io_ctx = IOContext(s3=s3_client)

	print("[online build] loading movie embeddings ...")
	movie_embeddings = load_movie_embeddings(io_ctx, movie_embedding_path)
	if not movie_embeddings:
		raise ValueError("movie embedding dictionary is empty")
	print(f"[online build] movie embeddings loaded: {len(movie_embeddings)}")

	metadata_path = path_join(warehouse_dir, "metadata.json")
	metadata = io_ctx.read_json(
		metadata_path,
		{
			"latest_input_version": "0",
			"latest_positive_version": None,
			"history": [],
		},
	)

	latest_input_done = str(metadata.get("latest_input_version", "0") or "0")
	latest_input_done_dt = None if latest_input_done == "0" else parse_version(latest_input_done, INPUT_VERSION_PATTERNS)

	print(f"[online build] latest input version done: {latest_input_done}")
	input_versions = discover_input_versions(io_ctx, input_root)
	if not input_versions:
		return {
			"status": "no_input_versions",
			"latest_input_version": latest_input_done,
		}

	latest_input_available = input_versions[-1][0]
	selected_input_versions = [
		(version, dt)
		for version, dt in input_versions
		if latest_input_done_dt is None or dt > latest_input_done_dt
	]

	if not selected_input_versions:
		return {
			"status": "already_up_to_date",
			"latest_input_version": latest_input_done,
			"latest_available_input_version": latest_input_available,
		}

	embedding_versions = discover_embedding_versions(io_ctx, embedding_root)
	if not embedding_versions:
		raise ValueError("No embedding versions found under embedding_path/versions")

	registry_path = path_join(warehouse_dir, "registry", "version.json")
	registry = io_ctx.read_json(registry_path, {"versions": [], "latest": None})
	existing_versions = list(registry.get("versions", [])) if isinstance(registry.get("versions"), list) else []
	positive_version = make_next_version(existing_versions, version_format)
	output_version_dir = path_join(warehouse_dir, "versions", positive_version)
	latest_dir = path_join(warehouse_dir, "latest")

	sink = (
		StreamingParquetSink(io_ctx=io_ctx, output_dir=output_version_dir, batch_size=batch_size)
		if write_parquet
		else None
	)
	embedding_cache: dict[str, dict[int, list[float]]] = {}

	processed_versions: list[str] = []
	total_input_rows = 0
	total_output_rows = 0
	total_user_match_rows = 0

	for input_version, input_dt in selected_input_versions:
		files = gather_input_user_event_files(io_ctx, input_root, input_version)
		if not files:
			print(f"[online build] skip input version {input_version}: no parquet under user_event(s)")
			processed_versions.append(input_version)
			continue

		eligible_embeddings = [
			(version, dt)
			for version, dt in embedding_versions
			if dt <= input_dt
		]
		if not eligible_embeddings:
			print(f"[online build] skip input version {input_version}: no embedding version <= input version")
			processed_versions.append(input_version)
			continue

		eligible_embeddings.sort(key=lambda item: item[1], reverse=True)

		for file_path in files:
			file_total_rows, user_counts = scan_file_user_distribution(io_ctx, file_path)
			if file_total_rows <= 0:
				continue

			matched_user_embeddings: dict[int, list[float]] = {}
			matched_row_count = 0
			unresolved_users = set(user_counts.keys())
			used_embedding_versions: list[str] = []

			for emb_version, _ in eligible_embeddings:
				if not unresolved_users:
					break

				if emb_version not in embedding_cache:
					embedding_cache[emb_version] = load_user_embedding_version(io_ctx, embedding_root, emb_version)
				emb_map = embedding_cache[emb_version]
				used_embedding_versions.append(emb_version)

				newly_matched_rows = 0
				for user_id in list(unresolved_users):
					vector = emb_map.get(user_id)
					if vector is None:
						continue
					matched_user_embeddings[user_id] = vector
					unresolved_users.remove(user_id)
					newly_matched_rows += user_counts.get(user_id, 0)

				matched_row_count += newly_matched_rows
				coverage = matched_row_count / max(1, file_total_rows)
				if coverage >= coverage_threshold:
					break

			coverage = matched_row_count / max(1, file_total_rows)
			total_input_rows += file_total_rows
			total_user_match_rows += matched_row_count

			parquet_file = io_ctx.open_parquet_file(file_path)
			for batch in parquet_file.iter_batches(batch_size=batch_size):
				table = pa.Table.from_batches([batch])
				needed_columns = {"user_id", "movie_id", "watch_duration_seconds"}
				missing = needed_columns.difference(table.column_names)
				if missing:
					raise ValueError(f"Missing required columns in {file_path}: {sorted(missing)}")

				user_ids = table["user_id"].to_pylist()
				movie_ids = table["movie_id"].to_pylist()
				durations = table["watch_duration_seconds"].to_pylist()

				out_rows: list[dict[str, Any]] = []
				for uid_raw, mid_raw, dur_raw in zip(user_ids, movie_ids, durations):
					uid = int(uid_raw)
					mid = int(mid_raw)

					user_vector = matched_user_embeddings.get(uid)
					if user_vector is None:
						continue

					movie_vector = movie_embeddings.get(mid)
					if movie_vector is None:
						continue

					out_rows.append(
						{
							"user_id": uid,
							"movie_id": mid,
							"user_embedding": user_vector,
							"movie_embedding": movie_vector,
							"label": float(dur_raw),
						}
					)

				total_output_rows += len(out_rows)
				if write_parquet and sink is not None:
					sink.write_rows(out_rows)

			print(
				f"[online build] input={input_version} file={basename(file_path)} "
				f"rows={file_total_rows} user_match_rows={matched_row_count} "
				f"coverage={coverage:.4f} used_embedding_versions={used_embedding_versions}"
			)

		processed_versions.append(input_version)

	if write_parquet and sink is not None:
		sink.close()

	latest_processed_input = processed_versions[-1] if processed_versions else latest_input_done
	latest_positive_version = positive_version if total_output_rows > 0 else metadata.get("latest_positive_version")

	created_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
	data_parts_prefix = path_join(output_version_dir, "parts")
	num_parts = len(sink.paths) if sink is not None else 0

	iceberg_namespace = deep_get(cfg, ["iceberg", "namespace"], "recsys")
	iceberg_table_name = deep_get(cfg, ["iceberg", "table_name"], "online_positive_samples")
	manifest = {
		"job_name": deep_get(cfg, ["job", "name"], "build_online_positive_samples"),
		"created_at": created_at,
		"version": positive_version,
		"input": {
			"input_root": input_root,
			"processed_input_versions": processed_versions,
			"movie_embedding_path": movie_embedding_path,
			"embedding_path": embedding_root,
		},
		"user_embedding": {
			"coverage_threshold": coverage_threshold,
			"fallback_rule": "fallback to earlier embedding versions until matched_row_coverage >= threshold or versions exhausted",
		},
		"output": {
			"data_parts_prefix": data_parts_prefix,
			"num_parts": num_parts,
			"version_dir": output_version_dir,
			"latest_dir": latest_dir,
		},
		"iceberg": {
			"enabled": False,
			"catalog_name": deep_get(cfg, ["iceberg", "catalog_name"], "default"),
			"namespace": iceberg_namespace,
			"table_name": iceberg_table_name,
			"table_identifier": f"{iceberg_namespace}.{iceberg_table_name}",
		},
		"stats": {
			"online_positive_samples_rows": total_output_rows,
			"num_parts": num_parts,
		},
	}

	manifest_path = path_join(output_version_dir, "manifest.json")
	io_ctx.write_json(manifest_path, manifest)
	io_ctx.write_json(path_join(latest_dir, "manifest.json"), manifest)

	history = metadata.get("history", [])
	history.append(
		{
			"built_at": manifest["created_at"],
			"latest_input_version": latest_processed_input,
			"latest_positive_version": latest_positive_version,
			"manifest": manifest_path,
			"output_rows": total_output_rows,
		}
	)

	metadata["latest_input_version"] = latest_processed_input
	metadata["latest_positive_version"] = latest_positive_version
	metadata["updated_at"] = created_at
	metadata["history"] = history[-200:]
	io_ctx.write_json(metadata_path, metadata)

	registry_versions = list(registry.get("versions", []))
	registry_versions = [item for item in registry_versions if str(item.get("version")) != positive_version]
	registry_versions.append(
		{
			"version": positive_version,
			"created_at": manifest["created_at"],
			"dataset_type": "online",
			"source_version": latest_processed_input if latest_processed_input else "None",
			"data_parts_prefix": data_parts_prefix,
			"manifest": manifest_path,
			"row_count": total_output_rows,
		}
	)
	registry["versions"] = registry_versions
	registry["latest"] = positive_version
	registry["updated_at"] = created_at
	io_ctx.write_json(registry_path, registry)

	result = {
		"status": "ok",
		"latest_input_available": latest_input_available,
		"latest_input_processed": latest_processed_input,
		"latest_positive_version": latest_positive_version,
		"output_rows": total_output_rows,
		"manifest": manifest_path,
		"metadata": metadata_path,
		"registry": registry_path,
	}
	print(json.dumps(result, indent=2, ensure_ascii=False))
	return result


def main() -> None:
	parser = argparse.ArgumentParser(description="Build online positive samples with incremental input versions")
	parser.add_argument("--config", required=True, help="Path to config yaml")
	args = parser.parse_args()

	res_dict = build_online_positive_samples(args.config)
	print(json.dumps(res_dict, indent=2, ensure_ascii=False))
    


if __name__ == "__main__":
	main()
