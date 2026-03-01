#!/usr/bin/env bash
set -euo pipefail
docker build -t temporal-moe-pipeline .
docker run --rm -v "$(pwd)/results:/app/results" temporal-moe-pipeline
