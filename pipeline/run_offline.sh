#!/bin/bash
set -euo pipefail

cd /app

echo "[STEP] Build Offline Samples"
python scripts/build_offline_samples.py --config scripts/config_offline_build.yaml

echo "[STEP] Split Dataset"
python scripts/split_dataset_new.py --config scripts/config_offline_split.yaml

echo "[DONE] finished"