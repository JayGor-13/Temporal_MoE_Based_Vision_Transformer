#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-temporal-moe-pipeline}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
HF_CACHE_DIR="${HF_CACHE_DIR:-$(pwd)/.cache/huggingface}"

mkdir -p "$HF_CACHE_DIR" "$(pwd)/results" "$(pwd)/data_store"

docker build -t "${IMAGE_NAME}:${IMAGE_TAG}" .
docker run --rm \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -e REFRESH_DATA="${REFRESH_DATA:-1}" \
  -v "$(pwd)/results:/app/results" \
  -v "$(pwd)/data_store:/app/data_store" \
  -v "$HF_CACHE_DIR:/root/.cache/huggingface" \
  "${IMAGE_NAME}:${IMAGE_TAG}"
