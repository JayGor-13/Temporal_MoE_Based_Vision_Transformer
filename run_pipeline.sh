#!/usr/bin/env bash
set -euo pipefail
python scripts/prepare_datasets.py --root data_store
python main.py
