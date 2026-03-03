#!/usr/bin/env bash
set -euo pipefail
if [[ -n "${HF_TOKEN:-}" ]]; then
  python scripts/prepare_datasets.py --root data_store --hf-token "$HF_TOKEN"
else
  python scripts/prepare_datasets.py --root data_store
fi
python main.py
