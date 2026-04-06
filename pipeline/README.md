# Pipeline

This module builds training-ready datasets from bootstrap artifacts and online-service exports.

## What It Does

- `run_offline.sh`: builds offline positive samples from the bootstrap user profiles, remaining user events, and movie embeddings.
- `run_online.sh`: builds online features from exported live events and user embedding snapshots.
- `split_dataset_new.py`: splits generated datasets into downstream train and evaluation sets.

## Main Inputs

- Bootstrap artifacts in MinIO under `artifacts/...`
- Movie embeddings in `embedding/...`
- Online-service exports in `artifacts/online-service-export/...`
- Online user embedding snapshots in `artifacts/online-service-user-embeddings/...`

## Run

Inside the container:

```bash
bash /app/run_offline.sh
bash /app/run_online.sh
```

From the project root:

```bash
docker compose run --rm pipeline bash /app/run_offline.sh
docker compose run --rm pipeline bash /app/run_online.sh
```

For debugging, open a shell from the project root:

```bash
docker run -it --rm -v $(pwd):/app <image> /bin/bash
```

## Important Config Files

- `scripts/config_offline_build.yaml`
- `scripts/config_offline_split.yaml`
- `scripts/config_online_build.yaml`
- `scripts/config_online_split.yaml`
- `scripts/config_dataset_profile.yaml`

These files define input paths, output locations, batching, and storage settings.
