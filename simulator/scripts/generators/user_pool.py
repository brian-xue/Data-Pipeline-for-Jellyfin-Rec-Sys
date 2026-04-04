from __future__ import annotations

import random


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

