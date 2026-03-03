from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass
class Config:
    seed: int = 42
    seeds: List[int] = (42, 43, 44)
    datasets: Tuple[str, ...] = ("msvd", "msrvtt")
    run_dense_baseline: bool = True
    run_ablations: bool = True

    # data
    data_root: str = "data_store"
    results_dir: str = "results"
    data_fraction: float = 1.0
    download_fraction_by_dataset: Dict[str, float] = field(default_factory=lambda: {"msvd": 1.0, "msrvtt": 1.0})
    max_videos_per_dataset: Dict[str, int] = field(default_factory=lambda: {"msvd": 2000, "msrvtt": 11000})
    train_videos_per_dataset: Dict[str, int] = field(default_factory=lambda: {"msvd": 1200, "msrvtt": 9000})
    val_videos_per_dataset: Dict[str, int] = field(default_factory=lambda: {"msvd": 100, "msrvtt": 1000})
    test_videos_per_dataset: Dict[str, int] = field(default_factory=lambda: {"msvd": 670, "msrvtt": 1000})
    max_captions_per_video_by_dataset: Dict[str, int] = field(default_factory=lambda: {"msvd": 20, "msrvtt": 20})
    num_frames: int = 8
    max_len: int = 24
    vocab_size: int = 12000

    # model
    embed_dim: int = 128
    num_heads: int = 4
    num_layers: int = 2
    num_experts: int = 8
    top_k: int = 2
    dropout: float = 0.1
    length_penalty: float = 0.7

    # optimization
    pretrain_batch_size: int = 32
    finetune_batch_size: int = 8
    pretrain_lr: float = 3e-4
    finetune_lr: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    pretrain_epochs: int = 3
    finetune_epochs: int = 10
    grad_clip: float = 1.0
    tau: float = 0.07
    label_smoothing: float = 0.1
    aux_loss_weight: float = 0.01
    contrastive_loss_weight: float = 0.1

    # execution
    num_workers: int = 2
    pin_memory: bool = True

    # ablation toggles
    no_caption_conditioning: bool = False
    no_aux_loss: bool = False
    no_temporal_bias: bool = False
    no_cross_modal_gating: bool = False
    no_contrastive_pretrain: bool = False


CFG = Config()
Path(CFG.results_dir).mkdir(parents=True, exist_ok=True)


def as_dict():
    return asdict(CFG)
