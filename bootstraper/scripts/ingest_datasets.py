from __future__ import annotations

import hashlib
import argparse
import os
import socket
import subprocess
import tempfile
import zipfile
import shutil
from pathlib import Path

import requests

from minio_s3 import list_csv_files, path_exists, to_s3_uri, upload_file_to_path


MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-32m.zip"
MOVIELENS_CHECKSUMS = {
    "links.csv": "8f033867bcb4e6be8792b21468b4fa6e",
    "movies.csv": "0df90835c19151f9d819d0822e190797",
    "ratings.csv": "cf12b74f9ad4b94a011f079e26d4270a",
    "tags.csv": "963bf4fa4de6b8901868fddd3eb54567",
}
TMDB_DATASET = "asaniczka/tmdb-movies-dataset-2023-930k-movies"
NON_S3_RAW_DIR = Path("/data/raw/non_s3")


def md5_for_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def download_file(url: str, destination: Path) -> None:
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with destination.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)


def copy_csvs(csv_paths: list[Path], destination_dir: Path) -> None:
    for csv_path in csv_paths:
        target = destination_dir / csv_path.name
        uploaded_uri = upload_file_to_path(csv_path, target)
        print(f"Uploaded {csv_path.name} -> {uploaded_uri}")


def save_local_only_csv(csv_path: Path, destination_dir: Path) -> Path:
    ensure_dir(destination_dir)
    target = destination_dir / csv_path.name
    shutil.copy2(csv_path, target)
    print(f"Saved local-only CSV -> {target}")
    return target


def ingest_movielens(raw_dir: Path, work_dir: Path) -> None:
    movielens_dir = raw_dir
    expected_files = [movielens_dir / name for name in MOVIELENS_CHECKSUMS]

    if all(path_exists(path) for path in expected_files):
        print(f"MovieLens already present in {to_s3_uri(movielens_dir)}")
        # return

    zip_path = work_dir / "ml-32m.zip"
    extract_dir = ensure_dir(work_dir / "ml-32m")

    print(f"Downloading MovieLens 32M from {MOVIELENS_URL}")
    download_file(MOVIELENS_URL, zip_path)

    print(f"Unzipping {zip_path}")
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(extract_dir)

    extracted_root = extract_dir / "ml-32m"
    csv_paths: list[Path] = []
    for filename, expected_md5 in MOVIELENS_CHECKSUMS.items():
        csv_path = extracted_root / filename
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing expected MovieLens file: {csv_path}")

        actual_md5 = md5_for_file(csv_path)
        if actual_md5 != expected_md5:
            raise ValueError(
                f"Checksum mismatch for {filename}: expected {expected_md5}, got {actual_md5}"
            )

        print(f"Checksum verified for {filename}")
        save_local_only_csv(csv_path, NON_S3_RAW_DIR)
        csv_paths.append(csv_path)

    copy_csvs(csv_paths, movielens_dir)


def ingest_tmdb(raw_dir: Path, work_dir: Path, local_only_csv: bool = False) -> None:
    tmdb_dir = raw_dir
    existing_csvs = list_csv_files(tmdb_dir)
    if existing_csvs:
        print(f"TMDB dataset already present in {to_s3_uri(tmdb_dir)}")
        # return

    download_dir = ensure_dir(work_dir / "tmdb")
    command = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        TMDB_DATASET,
        "--unzip",
        "-p",
        str(download_dir),
    ]

    print(f"Running Kaggle download for {TMDB_DATASET}")
    subprocess.run(command, check=True)

    csv_paths = sorted(download_dir.rglob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found after Kaggle download in {download_dir}")

    save_local_only_csv(csv_paths[0], NON_S3_RAW_DIR)

    if local_only_csv:
        print("Local-only mode enabled; skipping S3 upload for TMDB CSVs.")
        return

    copy_csvs(csv_paths, tmdb_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest datasets into MinIO-backed storage.")
    parser.add_argument(
        "--tmdb-local-only-csv",
        action="store_true",
        help="Only save one TMDB CSV under /mnt/object/minio_data/raw/non_s3/ and skip S3 upload for TMDB CSVs.",
    )
    args = parser.parse_args()

    warehouse_dir = ensure_dir(Path("/data/warehouse"))
    artifacts_dir = ensure_dir(Path("/data/artifacts"))
    object_store_mount = ensure_dir(Path("/data/"))
    raw_dir = ensure_dir(object_store_mount / "raw")

    print("Pipeline started.")
    print(f"Hostname: {socket.gethostname()}")
    print(f"Postgres host: {os.getenv('POSTGRES_HOST', 'postgres')}")
    print(f"Postgres port: {os.getenv('POSTGRES_PORT', '5432')}")
    print(f"Object store mount: {object_store_mount}")
    print(f"Raw dataset dir: {raw_dir}")
    print(f"Warehouse dir: {warehouse_dir}")
    print(f"Artifacts dir: {artifacts_dir}")

    with tempfile.TemporaryDirectory(prefix="ingest_", dir=artifacts_dir) as temp_dir:
        work_dir = Path(temp_dir)
        ingest_movielens(raw_dir, work_dir)
        ingest_tmdb(raw_dir, work_dir, local_only_csv=args.tmdb_local_only_csv)

    print("Dataset ingestion completed.")


if __name__ == "__main__":
    main()
