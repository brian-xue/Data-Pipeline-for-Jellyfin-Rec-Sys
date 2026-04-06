# Online Service

This module handles online recommendation logic. It processes recent events, maintains user-level state, exports snapshots, and serves recommendation candidates through a Flask API.

## What It Does

- reads auth and user events from PostgreSQL
- updates popular-movie statistics
- rebuilds user embeddings from recent behavior
- exports processed snapshots to object storage
- serves `/candidates` for real-time candidate retrieval

## API

Main endpoints:

- `GET /health`
- `GET /candidates?user_id=<id>&top_k=<k>`

The candidate API returns a response containing the selected category, item list, and result count for the requested user.

## Run

The service is usually started by Docker Compose with Gunicorn:

```bash
gunicorn -w 2 -b 0.0.0.0:18080 "scripts.api.candidate_api:create_app()"
```

For the threaded worker entrypoint inside the container:

```bash
python scripts/main.py --config scripts/config/config.yaml
```

## Main Config

`scripts/config/config.yaml` defines:

- PostgreSQL and MinIO connection settings
- processor polling intervals
- embedding and candidate retrieval parameters
- export snapshot locations
- optional built-in Flask API settings

## Key Components

- `scripts/processors/`: background processors for auth events, user events, exports, popular movies, and embedding refresh
- `scripts/api/candidate_api.py`: Flask API for candidate retrieval
- `scripts/services/` and `scripts/repositories/`: recommendation logic and data access layers
