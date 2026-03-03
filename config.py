from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass
class Config:
    seed: int = 42
    seeds: List[int] = (42,)
    datasets: Tuple[str, ...] = ("msvd", "msrvtt")
    run_dense_baseline: bool = False

    # data
    data_root: str = "data_store"
    results_dir: str = "results"
    data_fraction: float = 1.0
    download_fraction_by_dataset: Dict[str, float] = field(default_factory=lambda: {"msvd": 0.01, "msrvtt": 0.01})
    max_videos_per_dataset: Dict[str, int] = field(default_factory=lambda: {"msvd": 3, "msrvtt": 3})
    train_videos_per_dataset: Dict[str, int] = field(default_factory=lambda: {"msvd": 1, "msrvtt": 1})
    val_videos_per_dataset: Dict[str, int] = field(default_factory=lambda: {"msvd": 1, "msrvtt": 1})
    test_videos_per_dataset: Dict[str, int] = field(default_factory=lambda: {"msvd": 1, "msrvtt": 1})
    max_captions_per_video_by_dataset: Dict[str, int] = field(default_factory=lambda: {"msvd": 1, "msrvtt": 1})
    num_frames: int = 2
    max_len: int = 20
    vocab_size: int = 5000

    # model
    embed_dim: int = 128
    num_heads: int = 4
    num_layers: int = 2
    num_experts: int = 8
    top_k: int = 2
    dropout: float = 0.1
    length_penalty: float = 0.7

# Tokenizer
MAX_LEN = 20
VOCAB_SIZE = 5000
NUM_SAMPLES = 1970


RESUME_CHECKPOINT_PATH = None 
CHECKPOINT_SAVE_DIR = "/kaggle/working/checkpoints/"
