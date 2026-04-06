# Simulator

This module generates synthetic user login, logout, and watch events for the recommendation system.

## What It Does

- creates a random user pool
- simulates users coming online and going offline over time
- emits movie watch events on each tick
- writes events either directly to PostgreSQL or to the ingest API
- can optionally call the online candidate API during a run

## Default Flow

The default configuration sends generated events to:

- ingest API: `http://api:8080/ingest/events`
- candidate API: `http://online_service:18080/candidates` when incremental requests are enabled

The main entrypoint is `scripts/main.py`, which loads `scripts/config.yaml` and runs the tick-based simulator.

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

- PostgreSQL connection settings
- user pool size and user ID range
- target and max online users
- tick interval and total number of ticks
- login, logout, and event generation rates
- ingest API settings
- optional incremental candidate requests

Adjust this file to make the simulation lighter, heavier, shorter, or more API-focused.
