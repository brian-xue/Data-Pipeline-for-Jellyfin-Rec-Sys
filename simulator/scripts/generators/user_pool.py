from __future__ import annotations

import os
import random
from urllib.parse import urlparse

import pyarrow.parquet as pq

try:
	import s3fs  # type: ignore
except Exception:  # pragma: no cover - optional dependency
	s3fs = None


def generate_user_pool(
	user_pool_size: int,
	min_user_id: int,
	max_user_id: int,
	random_seed: int = 42,
) -> list[int]:
	if user_pool_size <= 0:
		return []
	if max_user_id < min_user_id:
		raise ValueError("max_user_id must be >= min_user_id")

	population_size = max_user_id - min_user_id + 1
	if user_pool_size > population_size:
		raise ValueError("user_pool_size is larger than available user id range")

	generator = random.Random(random_seed)
	return generator.sample(range(min_user_id, max_user_id + 1), user_pool_size)


def load_profile_user_embeddings(
	profile_path: str | None,
	sample_size: int,
	random_seed: int = 42,
	batch_size: int = 5000,
) -> dict[int, list[float]]:
	if not profile_path:
		return {}
	if not os.path.exists(profile_path):
		return {}
	if sample_size <= 0:
		return {}

	def _open_profile_source(path: str):
		if path.startswith("s3://"):
			if s3fs is None:
				raise RuntimeError("s3fs is required to read simulator base profile from s3:// paths")

			parsed = urlparse(path)
			bucket = parsed.netloc
			key = parsed.path.lstrip("/")
			endpoint = os.getenv("MINIO_ENDPOINT")
			access_key = os.getenv("MINIO_ROOT_USER") or os.getenv("AWS_ACCESS_KEY_ID")
			secret_key = os.getenv("MINIO_ROOT_PASSWORD") or os.getenv("AWS_SECRET_ACCESS_KEY")

			fs = s3fs.S3FileSystem(
				key=access_key,
				secret=secret_key,
				client_kwargs={"endpoint_url": endpoint} if endpoint else None,
			)
			return fs.open(f"{bucket}/{key}", "rb")

		if not os.path.exists(path):
			return None
		return open(path, "rb")

	source = _open_profile_source(profile_path)
	if source is None:
		return {}

	reservoir: list[tuple[int, list[float]]] = []
	seen = 0
	generator = random.Random(random_seed)
	try:
		parquet_file = pq.ParquetFile(source)

		for batch in parquet_file.iter_batches(batch_size=batch_size, columns=["user_id", "embedding"]):
			user_ids = batch.column(0).to_pylist()
			embeddings = batch.column(1).to_pylist()
			for user_id, embedding in zip(user_ids, embeddings):
				if embedding is None:
					continue
				item = (int(user_id), [float(value) for value in embedding])
				seen += 1
				if len(reservoir) < sample_size:
					reservoir.append(item)
				else:
					index = generator.randint(0, seen - 1)
					if index < sample_size:
						reservoir[index] = item
	finally:
		source.close()

	return {user_id: embedding for user_id, embedding in reservoir}


def build_simulator_user_pool(
	profile_path: str | None,
	online_user_sample_size: int,
	fallback_user_pool_size: int,
	min_user_id: int,
	max_user_id: int,
	random_seed: int = 42,
) -> tuple[set[int], set[int], dict[int, list[float]]]:
	profiles = load_profile_user_embeddings(
		profile_path=profile_path,
		sample_size=online_user_sample_size,
		random_seed=random_seed,
	)
	if profiles:
		user_ids = set(profiles.keys())
		return set(), set(user_ids), profiles

	user_ids = generate_user_pool(
		user_pool_size=fallback_user_pool_size,
		min_user_id=min_user_id,
		max_user_id=max_user_id,
		random_seed=random_seed,
	)
	return set(), set(user_ids), {}

