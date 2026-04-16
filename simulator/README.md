# Simulator

This module generates synthetic login, logout, impression, and watch-like events for the recommendation pipeline.

## What It Does

- loads a sampled subset of `simulator_base_profile.parquet` as the online user pool when available
- keeps each sampled user's embedding as a simulator-side reference preference
- triggers the candidate API once during the online process and reuses the returned items
- chooses watch events mainly from candidate results, with a small random fallback ratio
- writes events either directly to PostgreSQL or to the ingest API
- periodically runs garbage collection for long-lived simulations

## Data Paths

The simulator expects embedding inputs under `/data/embedding/`:

- `s3://artifacts/simulator_base_profile/simulator_base_profile.parquet`
- `/data/embedding/embedding.npy`
- `/data/embedding/ids.npy`

## Run

Inside the container:

```bash
python scripts/main.py --config scripts/config.yaml
```

From the project root:

```bash
docker compose exec simulator python scripts/main.py --config scripts/config.yaml
```

## Main Config

`scripts/config.yaml` controls:

- sampled profile size and fallback user pool size
- target and max online users
- tick interval and total number of ticks
- one-shot candidate request settings
- movie embedding paths and random movie fallback ratio
- watch duration ranges
- ingest API settings

Adjust this file to make the simulation lighter, heavier, shorter, or more API-focused.
