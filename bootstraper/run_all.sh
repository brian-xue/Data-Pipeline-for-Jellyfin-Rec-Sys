#!/bin/sh
set -e

python scripts/ingest_datasets.py
python scripts/build_embedding_text.py
python scripts/emdding.py
python scripts/build_initial_user.py --config scripts/config.yaml