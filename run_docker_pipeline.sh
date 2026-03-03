#!/usr/bin/env bash
set -euo pipefail

docker build -t temporal-moe-pipeline .
docker run --rm \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -e REFRESH_DATA="${REFRESH_DATA:-1}" \
  -v "$(pwd)/results:/app/results" \
  -v "$(pwd)/data_store:/app/data_store" \
  temporal-moe-pipeline
