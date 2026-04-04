#!/bin/bash
set -euo pipefail

cd /app

echo "[STEP] Build Online Features"
python scripts/build_online_features.py --config scripts/config_online_build.yaml

echo "[STEP] Split Dataset"
python scripts/split_dataset_new.py --config scripts/config_online_split.yaml

echo "[DONE] finished"