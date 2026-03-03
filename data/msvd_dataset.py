import json
from pathlib import Path
from typing import Dict, Iterable, List

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from config import CFG


class VideoCaptionDataset(Dataset):
    """Video-caption dataset backed by real video files listed in manifest JSON."""

    def __init__(
        self,
        samples: List[Dict],
        tokenizer,
        dataset_root: Path,
        num_frames: int = 16,
        max_len: int = 20,
        frame_size: int = 224,
    ):
        self.samples = samples
        self.tokenizer = tokenizer
        self.dataset_root = dataset_root
        self.num_frames = num_frames
        self.max_len = max_len
        self.frame_size = frame_size

    def __len__(self):
        return len(self.samples)

    def _resolve_video_path(self, item: Dict) -> Path:
        video_path = item.get("video_path")
        if not video_path:
            raise KeyError(f"Missing 'video_path' in manifest row for video_id={item.get('video_id')}")
        p = Path(video_path)
        return p if p.is_absolute() else (self.dataset_root / p)

    def _decode_frames(self, video_path: Path) -> torch.Tensor:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            cap.release()
            raise RuntimeError(f"Video has no readable frames: {video_path}")

        target_idx = np.linspace(0, frame_count - 1, num=self.num_frames, dtype=np.int64)
        frames = []
        last_frame = None
        for idx in target_idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok or frame is None:
                if last_frame is None:
                    continue
                frame = last_frame
            else:
                last_frame = frame

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (self.frame_size, self.frame_size), interpolation=cv2.INTER_AREA)
            frames.append(frame)

        cap.release()
        if not frames:
            raise RuntimeError(f"Unable to decode sampled frames from video: {video_path}")

        while len(frames) < self.num_frames:
            frames.append(frames[-1])

        arr = np.stack(frames, axis=0).astype(np.float32) / 255.0  # [T,H,W,C]
        return torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()

    def __getitem__(self, idx):
        item = self.samples[idx]
        tokenized = self.tokenizer(item["caption"], max_length=self.max_len)
        video = self._decode_frames(self._resolve_video_path(item))
        return {
            "video": video,
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "caption": item["caption"],
            "video_id": item["video_id"],
        }


def _slice_fraction(items: List[Dict]):
    if CFG.data_fraction >= 1.0:
        return items
    k = max(1, int(len(items) * CFG.data_fraction))
    return items[:k]


def load_manifest(path: Path) -> List[Dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing manifest: {path}")
    return json.loads(path.read_text())


def collect_captions(dataset_names: Iterable[str], split: str = "train") -> List[str]:
    captions: List[str] = []
    for name in dataset_names:
        path = Path(CFG.data_root) / name / f"{split}.json"
        for row in load_manifest(path):
            caption = str(row.get("caption", "")).strip()
            if caption:
                captions.append(caption)
    return captions


def build_dataset(dataset_name: str, split: str, tokenizer):
    dataset_root = Path(CFG.data_root) / dataset_name
    manifest_path = dataset_root / f"{split}.json"
    samples = load_manifest(manifest_path)
    samples = _slice_fraction(samples)
    if not samples:
        raise ValueError(f"Empty split after applying data_fraction: {manifest_path}")
    return VideoCaptionDataset(samples, tokenizer, dataset_root=dataset_root, num_frames=CFG.num_frames, max_len=CFG.max_len)
