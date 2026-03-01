from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List


@dataclass
class Config:
    seed: int = 42
    seeds: List[int] = (42, 123, 999)

    # data
    data_root: str = "data_store"
    results_dir: str = "results"
    data_fraction: float = 0.01  # intentionally tiny for local smoke runs
    num_frames: int = 16
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

    # optimization
    pretrain_batch_size: int = 64
    finetune_batch_size: int = 32
    pretrain_lr: float = 3e-4
    finetune_lr: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    pretrain_epochs: int = 1
    finetune_epochs: int = 1
    grad_clip: float = 1.0
    tau: float = 0.07
    label_smoothing: float = 0.1
    aux_loss_weight: float = 0.01


CFG = Config()
Path(CFG.results_dir).mkdir(parents=True, exist_ok=True)


def as_dict():
    return asdict(CFG)
