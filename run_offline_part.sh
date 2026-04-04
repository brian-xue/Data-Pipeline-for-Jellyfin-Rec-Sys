#! /bin/bash

set -euo pipefail

echo "START"

sleep 2


echo "===== Extract Dataset, Data processing ====="
docker compose run --rm bootstraper bash /app/run_all.sh

sleep 5

echo "===== Build Dataset ====="
docker compose run --rm pipeline bash /app/run_offline.sh

echo "======= Done ========="
