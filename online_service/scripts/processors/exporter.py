from __future__ import annotations

import threading
from datetime import datetime, timezone

from scripts.db.connection import get_connection
from scripts.repositories.auth_event_repository import AuthEventRepository
from scripts.repositories.checkpoint_repository import CheckpointRepository
from scripts.repositories.user_event_repository import UserEventRepository
from scripts.services.export_service import (
	build_export_manifest,
	export_rows_to_parquet,
	read_json_or_default,
	write_json,
)
from scripts.utils.logger import get_logger


LOGGER = get_logger("online_service.exporter")
JOB_NAME = "online_service_export"


class ExporterProcessor:
	def __init__(self, config):
		self.config = config
		self.interval_seconds = config.processor_intervals.exporter_seconds

	def _version(self) -> str:
		return datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")

	def _join(self, *parts: str) -> str:
		cleaned = [p.strip("/") for p in parts if p]
		root = cleaned[0]
		tail = cleaned[1:]
		if root.startswith("s3://"):
			return root.rstrip("/") + "/" + "/".join(tail)
		return "/".join([root.rstrip("/")] + tail)

	def export_auth_events(self, rows: list[dict], version_root: str) -> str | None:
		if not rows:
			return None
		output_path = self._join(version_root, self.config.exporter.auth_events_prefix, "data.parquet")
		return export_rows_to_parquet(rows=rows, output_path=output_path, storage=self.config.object_storage)

	def export_user_events(self, rows: list[dict], version_root: str) -> str | None:
		if not rows:
			return None
		output_path = self._join(version_root, self.config.exporter.user_events_prefix, "data.parquet")
		return export_rows_to_parquet(rows=rows, output_path=output_path, storage=self.config.object_storage)

	def run_once(self) -> int:
		with get_connection(self.config) as conn:
			checkpoint_repo = CheckpointRepository(conn)
			auth_repo = AuthEventRepository(conn)
			user_event_repo = UserEventRepository(conn)

			checkpoint = checkpoint_repo.get_checkpoint(JOB_NAME)
			last_auth_event_id = int(checkpoint["last_auth_event_id"])
			last_user_event_id = int(checkpoint["last_user_event_id"])

			auth_rows = auth_repo.fetch_auth_events_after(last_auth_event_id)
			user_rows = user_event_repo.fetch_user_events_after(last_user_event_id)

			if not auth_rows and not user_rows:
				return 0

			version = self._version()
			root = self.config.exporter.snapshot_root
			version_root = self._join(root, "versions", version)
			latest_root = self._join(root, "latest")
			registry_path = self._join(root, "registry", "versions.json")

			exported_paths: dict[str, str] = {}

			auth_path = self.export_auth_events(auth_rows, version_root)
			if auth_path:
				exported_paths["auth_events"] = auth_path

			user_path = self.export_user_events(user_rows, version_root)
			if user_path:
				exported_paths["user_events"] = user_path

			stats = {
				"auth_events": len(auth_rows),
				"user_events": len(user_rows),
			}
			manifest = build_export_manifest(version=version, exported_paths=exported_paths, stats=stats)

			version_manifest_path = self._join(version_root, "manifest.json")
			latest_manifest_path = self._join(latest_root, "manifest.json")
			write_json(manifest, version_manifest_path, self.config.object_storage)
			write_json(manifest, latest_manifest_path, self.config.object_storage)

			registry = read_json_or_default(registry_path, {"versions": [], "latest": None}, self.config.object_storage)
			versions = registry.get("versions", [])
			versions.append(
				{
					"version": version,
					"created_at": manifest["created_at"],
					"manifest": version_manifest_path,
					"stats": stats,
					"exports": exported_paths,
				}
			)
			registry["versions"] = versions
			registry["latest"] = version
			write_json(registry, registry_path, self.config.object_storage)

			if auth_rows:
				last_auth = auth_rows[-1]
				checkpoint_repo.update_auth_checkpoint(
					job_name=JOB_NAME,
					last_auth_event_id=int(last_auth["auth_event_id"]),
					last_auth_event_time=last_auth["event_time"],
					status="idle",
				)
			if user_rows:
				last_user = user_rows[-1]
				checkpoint_repo.update_user_event_checkpoint(
					job_name=JOB_NAME,
					last_user_event_id=int(last_user["event_id"]),
					last_user_event_time=last_user["event_time"],
					status="idle",
				)

			return len(auth_rows) + len(user_rows)

	def run_loop(self, stop_event: threading.Event) -> None:
		LOGGER.info("ExporterProcessor loop started")
		while not stop_event.is_set():
			try:
				exported = self.run_once()
				if exported > 0:
					LOGGER.info("ExporterProcessor exported %s rows", exported)
			except Exception:
				LOGGER.exception("ExporterProcessor run_once failed")
			stop_event.wait(self.interval_seconds)
