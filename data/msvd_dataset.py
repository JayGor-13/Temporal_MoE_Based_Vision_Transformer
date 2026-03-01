import json
import random
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import Dataset

from config import CFG


class TinyVideoCaptionDataset(Dataset):
    """Small synthetic/video-metadata based dataset used for smoke running full pipeline."""

    def __init__(self, samples: List[Dict], tokenizer, num_frames: int = 16, max_len: int = 20):
        self.samples = samples
        self.tokenizer = tokenizer
        self.num_frames = num_frames
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        tokenized = self.tokenizer(item["caption"], max_length=self.max_len)
        video = torch.rand(self.num_frames, 3, 224, 224)
        return {
            "video": video,
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "caption": item["caption"],
            "video_id": item["video_id"],
        }


def _slice_fraction(items: List[Dict]):
    k = max(1, int(len(items) * CFG.data_fraction))
    return items[:k]


def load_manifest(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text())


def build_dataset(dataset_name: str, split: str, tokenizer):
    manifest_path = Path(CFG.data_root) / dataset_name / f"{split}.json"
    samples = load_manifest(manifest_path)
    if not samples:
        # fallback synthetic entries so pipeline always runs
        random.seed(CFG.seed)
        samples = [{"video_id": f"{dataset_name}_{split}_{i}", "caption": f"a sample caption {i}"} for i in range(32)]
    samples = _slice_fraction(samples)
    return TinyVideoCaptionDataset(samples, tokenizer, num_frames=CFG.num_frames, max_len=CFG.max_len)
