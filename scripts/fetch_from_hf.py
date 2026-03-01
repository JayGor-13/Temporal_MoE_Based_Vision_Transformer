"""Optional helper to fetch manifests from HuggingFace datasets API.
Requires: pip install datasets and authentication if needed.
"""
from pathlib import Path
import json


def export_subset(ds, out_path, caption_key="caption", id_key="video_id", limit=200):
    rows = []
    for i, ex in enumerate(ds):
        if i >= limit:
            break
        rows.append({"video_id": str(ex.get(id_key, i)), "caption": str(ex.get(caption_key, ""))})
    Path(out_path).write_text(json.dumps(rows, indent=2))


def main():
    from datasets import load_dataset

    Path("data_store/msrvtt").mkdir(parents=True, exist_ok=True)
    train = load_dataset("friedrichor/MSR-VTT", "train_9k", split="train")
    test = load_dataset("friedrichor/MSR-VTT", "test_1k", split="train")
    export_subset(train, "data_store/msrvtt/train.json")
    export_subset(test, "data_store/msrvtt/test.json")
    export_subset(test.select(range(min(64, len(test)))), "data_store/msrvtt/val.json")


if __name__ == "__main__":
    main()
