from __future__ import annotations

import json
from datetime import datetime
from typing import Any


class UserEmbeddingSnapshotRepository:
	def __init__(self, conn):
		self.conn = conn

	def insert_snapshot(
		self,
		user_id: int,
		embedding_uri: str,
		embedding_version: str,
		embedding_updated_at: datetime,
		model_version: str | None = None,
		source_event_max_id: int | None = None,
		source_event_count: int | None = None,
		metadata_json: dict[str, Any] | None = None,
	) -> None:
		with self.conn.cursor() as cur:
			cur.execute(
				"""
				INSERT INTO user_embedding_snapshots (
					user_id,
					embedding_uri,
					embedding_version,
					model_version,
					source_event_max_id,
					source_event_count,
					embedding_updated_at,
					metadata_json
				)
				VALUES (%s, %s, %s, %s, %s, %s, %s, CAST(%s AS jsonb))
				""",
				(
					user_id,
					embedding_uri,
					embedding_version,
					model_version,
					source_event_max_id,
					source_event_count,
					embedding_updated_at,
					json.dumps(metadata_json or {}),
				),
			)
