#!/bin/bash
set -euo pipefail

cd /app

echo "[STEP] ingest_datasets"
python3 scripts/ingest_datasets.py