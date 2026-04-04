from __future__ import annotations

import time
from typing import Callable


class TickScheduler:
	def __init__(self, tick_seconds: float):
		self.tick_seconds = tick_seconds

	def run(self, total_ticks: int, tick_fn: Callable[[int], None]) -> None:
		for tick_index in range(1, total_ticks + 1):
			started_at = time.time()
			tick_fn(tick_index)
			elapsed = time.time() - started_at
			sleep_seconds = max(0.0, self.tick_seconds - elapsed)
			time.sleep(sleep_seconds)

