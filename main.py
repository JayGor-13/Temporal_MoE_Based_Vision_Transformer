from pathlib import Path
import json

import torch

from config import CFG, as_dict
from data.tokenizer import SimpleTokenizer
from data.msvd_dataset import build_dataset, collect_captions
from models.video_captioning_moe import VideoCaptioningMoE
from train.train_loop import run_training
from utils.misc import set_seed


ABLATIONS = {
    "A1_dense_only": {"dense_only": True},
    "A2_no_caption_conditioning": {},
    "A3_no_aux_loss": {},
    "A4_no_temporal_bias": {},
    "A5_no_cross_modal_gating": {},
    "A6_no_contrastive_pretrain": {},
}


def ensure_results_tree():
    root = Path(CFG.results_dir)
    for sub in ["metrics", "raw", "plots", "benchmarks"]:
        (root / sub).mkdir(parents=True, exist_ok=True)


def save_benchmark_template():
    path = Path(CFG.results_dir) / "benchmarks" / "reported_baselines.json"
    baselines = {
        "CLIP4Clip": {"citation": "fill_me", "CIDEr": None, "METEOR": None, "SPICE": None},
        "BLIP-2 (video)": {"citation": "fill_me", "CIDEr": None, "METEOR": None, "SPICE": None},
        "Frozen": {"citation": "fill_me", "CIDEr": None, "METEOR": None, "SPICE": None},
        "Video Swin Transformer": {"citation": "fill_me", "CIDEr": None, "METEOR": None, "SPICE": None},
    }
    path.write_text(json.dumps(baselines, indent=2))


def run_all():
    ensure_results_tree()
    save_benchmark_template()

    dataset_names = list(CFG.datasets)
    tokenizer = SimpleTokenizer(CFG.vocab_size)
    vocab_texts = collect_captions(dataset_names, split="train")
    if not vocab_texts:
        raise RuntimeError("No training captions found. Run scripts/prepare_datasets.py first.")
    tokenizer.build_vocab(vocab_texts)

    summary = []
    for dataset in dataset_names:
        train_ds = build_dataset(dataset, "train", tokenizer)
        test_ds = build_dataset(dataset, "test", tokenizer)

        for seed in CFG.seeds:
            set_seed(seed)
            model = VideoCaptioningMoE(vocab_size=CFG.vocab_size)
            run_name = f"{dataset}_seed{seed}"
            metrics = run_training(model, train_ds, test_ds, seed, run_name)
            metrics["dataset"] = dataset
            summary.append(metrics)
            print(
                f"[{dataset}][seed={seed}] "
                f"train_loss={metrics.get('train_loss', 0):.4f} "
                f"CIDEr={metrics.get('CIDEr', 0):.4f} METEOR={metrics.get('METEOR', 0):.4f}"
            )

        if CFG.run_dense_baseline:
            set_seed(CFG.seed)
            dense_model = VideoCaptioningMoE(vocab_size=CFG.vocab_size, dense_only=True)
            _ = run_training(dense_model, train_ds, test_ds, CFG.seed, f"{dataset}_dense_baseline")

    Path(CFG.results_dir, "metrics", "all_runs.json").write_text(json.dumps(summary, indent=2))
    Path(CFG.results_dir, "raw", "config.json").write_text(json.dumps(as_dict(), indent=2, default=str))


if __name__ == "__main__":
    run_all()
