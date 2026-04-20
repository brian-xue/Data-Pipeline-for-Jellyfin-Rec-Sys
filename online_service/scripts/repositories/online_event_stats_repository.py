from __future__ import annotations

from datetime import datetime


class OnlineEventStatsRepository:
	def __init__(self, conn):
		self.conn = conn

	def upsert_window_stats(
		self,
		window_start_time: datetime,
		window_end_time: datetime,
		total_count: int,
		abnormal_count: int,
	) -> None:
		with self.conn.cursor() as cur:
			cur.execute(
				"""
				INSERT INTO online_event_stats (
					window_start_time,
					window_end_time,
					total_count,
					abnormal_count,
					updated_at
				)
				VALUES (%s, %s, %s, %s, NOW())
				ON CONFLICT (window_start_time)
				DO UPDATE SET
					window_end_time = EXCLUDED.window_end_time,
					total_count = online_event_stats.total_count + EXCLUDED.total_count,
					abnormal_count = online_event_stats.abnormal_count + EXCLUDED.abnormal_count,
					updated_at = NOW()
				""",
				(
					window_start_time,
					window_end_time,
					int(total_count),
					int(abnormal_count),
				),
			)
