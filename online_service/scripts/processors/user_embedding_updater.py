from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import datetime, timezone

from scripts.services.export_service import export_rows_to_parquet, write_json
import time

from scripts.db.connection import get_connection
from scripts.repositories.checkpoint_repository import CheckpointRepository
from psycopg2.errors import DeadlockDetected

from scripts.repositories.user_embedding_snapshot_repository import UserEmbeddingSnapshotRepository
from scripts.repositories.user_event_repository import UserEventRepository
from scripts.repositories.user_repository import UserRepository
from scripts.services.embedding_service import compute_weighted_user_embedding
from scripts.utils.logger import get_logger


LOGGER = get_logger("online_service.user_embedding_updater")
JOB_NAME = "online_service_user_embedding"


@dataclass
class UserEmbeddingUpdate:
    user_id: int
    embedding: list[float]
    previous_version: str | None
    source_event_count: int
    source_event_max_id: int | None


class UserEmbeddingUpdater:
    def __init__(self, config):
        self.config = config
        self.interval_seconds = config.processor_intervals.user_embedding_updater_seconds
        LOGGER.info("UserEmbeddingUpdater interval configured: %s seconds", self.interval_seconds)

    def collect_affected_users(self, event_repo: UserEventRepository, last_user_event_id: int) -> list[int]:
        return event_repo.fetch_affected_user_ids_after(last_user_event_id)

    def _mock_movie_embedding(self, movie_id: int) -> list[float]:
        base = float(movie_id % 1000) / 1000.0
        return [base, base / 2.0, base / 3.0, base / 4.0]

    def _join(self, *parts: str) -> str:
        cleaned = [p.strip("/") for p in parts if p]
        root = cleaned[0]
        tail = cleaned[1:]
        if root.startswith("s3://"):
            return root.rstrip("/") + "/" + "/".join(tail)
        return "/".join([root.rstrip("/")] + tail)

    def _current_version_hour(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d-%H-%M")

    def _persist_user_update_with_retry(
        self,
        item: UserEmbeddingUpdate,
        embedding_uri: str,
        version: str,
        embedding_updated_at: datetime,
        manifest_path: str,
        version_path: str,
        max_retries: int = 3,
    ) -> bool:
        for attempt in range(1, max_retries + 1):
            try:
                with get_connection(self.config) as conn:
                    user_repo = UserRepository(conn)
                    snapshot_repo = UserEmbeddingSnapshotRepository(conn)
                    user_repo.update_user_embedding(
                        user_id=item.user_id,
                        embedding_uri=embedding_uri,
                        embedding_version=version,
                        embedding_updated_at=embedding_updated_at,
                    )
                    snapshot_repo.insert_snapshot(
                        user_id=item.user_id,
                        embedding_uri=embedding_uri,
                        embedding_version=version,
                        source_event_max_id=item.source_event_max_id,
                        source_event_count=item.source_event_count,
                        embedding_updated_at=embedding_updated_at,
                        metadata_json={
                            "manifest": manifest_path,
                            "version_file": version_path,
                            "previous_version": item.previous_version,
                        },
                    )
                return True
            except DeadlockDetected:
                if attempt >= max_retries:
                    LOGGER.exception(
                        "Failed to persist embedding update for user_id=%s due to deadlock after %s attempts",
                        item.user_id,
                        attempt,
                    )
                    return False
                backoff_seconds = 0.1 * attempt
                LOGGER.warning(
                    "Deadlock while persisting user_id=%s embedding (attempt %s/%s), retrying in %.2fs",
                    item.user_id,
                    attempt,
                    max_retries,
                    backoff_seconds,
                )
                time.sleep(backoff_seconds)

    def _save_embedding_payload(
        self,
        updates: list[UserEmbeddingUpdate],
        version: str,
        last_user_event_id: int,
    ) -> tuple[str, str, str, str]:
        snapshot_root = self.config.embedding.snapshot_root.rstrip("/")
        version_root = self._join(snapshot_root, "versions", version)
        output_path = self._join(version_root, f"embeddings_upto_{last_user_event_id}.parquet")
        manifest_path = self._join(version_root, "manifest.json")
        version_path = self._join(snapshot_root, "version.json")
        created_at = datetime.now(timezone.utc).isoformat()

        rows = [
            {
                "user_id": item.user_id,
                "version": version,
                "previous_version": item.previous_version,
                "embedding": item.embedding,
                "created_at": created_at,
            }
            for item in updates
        ]
        export_rows_to_parquet(rows=rows, output_path=output_path, storage=self.config.object_storage)

        manifest_payload = {
            "version": version,
            "created_at": created_at,
            "embedding_uri": output_path,
            "records": len(rows),
            "users": [
                {
                    "user_id": item.user_id,
                    "previous_version": item.previous_version,
                    "previous_manifest": (
                        self._join(snapshot_root, "versions", item.previous_version, "manifest.json")
                        if item.previous_version
                        else None
                    ),
                    "source_event_count": item.source_event_count,
                    "source_event_max_id": item.source_event_max_id,
                }
                for item in updates
            ],
        }
        write_json(manifest_payload, manifest_path, self.config.object_storage)

        previous_global_version = None
        if updates:
            previous_global_version = max(
                [item.previous_version for item in updates if item.previous_version is not None],
                default=None,
            )

        version_payload = {
            "latest": version,
            "previous": previous_global_version,
            "records": len(rows),
            "updated_at": created_at,
            "manifest": manifest_path,
            "embedding_uri": output_path,
        }
        write_json(version_payload, version_path, self.config.object_storage)
        return output_path, manifest_path, version_path, created_at

    def recompute_user_embedding(
        self,
        user_id: int,
        event_repo: UserEventRepository,
        user_repo: UserRepository,
    ) -> UserEmbeddingUpdate | None:
        history_cfg = self.config.user_history_query
        rows = event_repo.fetch_recent_user_events_within_window(
            user_id=user_id,
            limit=history_cfg.recent_limit,
            window_hours=history_cfg.recent_window_hours,
        )
        if not rows:
            return None

        movie_embeddings: list[list[float]] = []
        weights: list[float] = []
        for row in rows:
            movie_embeddings.append(self._mock_movie_embedding(int(row["movie_id"])))
            weights.append(max(float(row["watch_duration_seconds"]), self.config.embedding.min_watch_duration_seconds))

        vector = compute_weighted_user_embedding(movie_embeddings, weights)
        if vector is None:
            return None

        current_user = user_repo.get_user_by_id(user_id)
        previous_version = None if not current_user else current_user.get("embedding_version")

        source_event_max_id: int | None = None
        for row in rows:
            event_id = row.get("event_id")
            if event_id is None:
                continue
            event_id_int = int(event_id)
            if source_event_max_id is None or event_id_int > source_event_max_id:
                source_event_max_id = event_id_int

        return UserEmbeddingUpdate(
            user_id=user_id,
            embedding=vector,
            previous_version=previous_version,
            source_event_count=len(rows),
            source_event_max_id=source_event_max_id,
        )

    def run_once(self) -> int:
        updated = 0
        with get_connection(self.config) as conn:
            checkpoint_repo = CheckpointRepository(conn)
            event_repo = UserEventRepository(conn)
            user_repo = UserRepository(conn)

            checkpoint = checkpoint_repo.get_checkpoint(JOB_NAME)
            last_user_event_id = int(checkpoint["last_user_event_id"])
            latest_events = event_repo.fetch_user_events_after(last_user_event_id=last_user_event_id)
            if not latest_events:
                return 0

            affected_users = sorted({int(row["user_id"]) for row in latest_events})
            if not affected_users:
                return 0

            updates: list[UserEmbeddingUpdate] = []
            for user_id in affected_users:
                user_update = self.recompute_user_embedding(user_id=user_id, event_repo=event_repo, user_repo=user_repo)
                if user_update is not None:
                    updates.append(user_update)

            if updates:
                version = self._current_version_hour()
                last_event = latest_events[-1]
                embedding_uri, manifest_path, version_path, created_at_iso = self._save_embedding_payload(
                    updates=updates,
                    version=version,
                    last_user_event_id=int(last_event["event_id"]),
                )
                embedding_updated_at = datetime.fromisoformat(created_at_iso)

                for item in updates:
                    if self._persist_user_update_with_retry(
                        item=item,
                        embedding_uri=embedding_uri,
                        version=version,
                        embedding_updated_at=embedding_updated_at,
                        manifest_path=manifest_path,
                        version_path=version_path,
                    ):
                        updated += 1
                updated = len(updates)

            if latest_events:
                last_event = latest_events[-1]
                checkpoint_repo.update_user_event_checkpoint(
                    job_name=JOB_NAME,
                    last_user_event_id=int(last_event["event_id"]),
                    last_user_event_time=last_event["event_time"],
                    status="idle",
                )

        return updated

    def run_loop(self, stop_event: threading.Event) -> None:
        LOGGER.info("UserEmbeddingUpdater loop started (interval=%ss)", self.interval_seconds)
        while not stop_event.is_set():
            try:
                updated = self.run_once()
                if updated > 0:
                    LOGGER.info("UserEmbeddingUpdater updated %s users", updated)
            except Exception:
                LOGGER.exception("UserEmbeddingUpdater run_once failed")
            stop_event.wait(self.interval_seconds)
