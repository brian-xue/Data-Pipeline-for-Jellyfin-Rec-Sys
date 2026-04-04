#!/bin/bash
set -euo pipefail

cd /app

echo "[STEP] ingest_datasets"
python3 scripts/ingest_datasets.py

echo "[STEP] build_embedding_text"
python3 scripts/build_embedding_text.py

echo "[STEP] embedding"
python3 scripts/embedding.py

echo "[STEP] build_embedding_index"
python3 scripts/build_embedding_index.py

echo "[STEP] build_initial_user"
python3 scripts/build_initial_user.py --config scripts/config.yaml

echo "[DONE] bootstraper pipeline finished"