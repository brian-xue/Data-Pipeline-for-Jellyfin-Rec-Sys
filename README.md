# Data Pipeline for Jellyfin Recommendation System

This repository contains an end-to-end MLOps workflow for a movie recommendation system. It covers bootstrap data preparation, online event ingestion, online candidate generation, workload simulation, and dataset building for model training.

## System Overview

The project is organized around a few connected services:

- `bootstraper`: downloads external datasets, builds movie embeddings, creates FAISS indexes, and prepares initial user profile artifacts.
- `pipeline`: builds offline and online training datasets from stored artifacts and exported online events.
- `api`: ingests login, logout, and watch events into PostgreSQL.
- `online_service`: processes recent events, updates user embeddings, exports snapshots, and serves recommendation candidates.
- `simulator`: generates synthetic user sessions and playback events to stress-test the online path.
- `postgres` and `minio`: provide relational storage and object storage for datasets and artifacts.

## End-to-End Flow

1. `bootstraper` prepares the initial movie and user artifacts.
2. `pipeline/run_offline.sh` creates offline training samples from bootstrap outputs.
3. `simulator` sends events to `api`, which writes them into PostgreSQL.
4. `online_service` reads those events, updates user state, exports snapshots, and serves `/candidates`.
5. `pipeline/run_online.sh` converts exported online data into online training datasets.

## Repository Structure

```text
.
├── bootstraper/      # initial dataset + embedding preparation
├── pipeline/         # offline and online dataset builders
├── online_service/   # online processors and candidate API
├── simulator/        # synthetic traffic generator
├── api/              # event ingest API
├── db/               # database schema and initialization
└── docker-compose.yml
```
