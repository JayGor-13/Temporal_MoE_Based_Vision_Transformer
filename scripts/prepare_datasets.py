"""Create dataset directory structure and optional metadata pulls.

Usage:
  python scripts/prepare_datasets.py --root data_store
"""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data_store")
    args = ap.parse_args()

    root = Path(args.root)
    specs = {
        "msvd": {"train": 1200, "val": 100, "test": 670},
        "msrvtt": {"train": 6513, "val": 497, "test": 2990},
        "vatex": {"train": 25991, "val": 3000, "test": 6000},
    }

    for ds, splits in specs.items():
        d = root / ds
        d.mkdir(parents=True, exist_ok=True)
        (d / "README.txt").write_text(
            "Put official files here or use external APIs (HF/Kaggle) to generate split JSON manifests.\n"
        )
        for split, n in splits.items():
            sample = [{"video_id": f"{ds}_{split}_{i}", "caption": f"placeholder caption {i}"} for i in range(min(32, n))]
            (d / f"{split}.json").write_text(json.dumps(sample, indent=2))

    print(f"Created dataset structure under: {root}")


if __name__ == "__main__":
    main()
