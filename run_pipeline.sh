#!/usr/bin/env bash
set -euo pipefail

REFRESH_DATA="${REFRESH_DATA:-1}"
args=(--root data_store --datasets msvd msrvtt)

if [[ "$REFRESH_DATA" == "1" ]]; then
  args+=(--refresh)
fi

if [[ -n "${HF_TOKEN:-}" ]]; then
  args+=(--hf-token "$HF_TOKEN")
fi

python scripts/prepare_datasets.py "${args[@]}"
python main.py
