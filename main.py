from pathlib import Path
import json

import torch

from config import CFG, as_dict
from data.tokenizer import SimpleTokenizer
from data.msvd_dataset import build_dataset
from models.video_captioning_moe import VideoCaptioningMoE
from train.train_loop import run_training
from utils.misc import set_seed


DATASETS = ["msvd", "msrvtt", "vatex"]
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


def save_visualization_placeholders(tag: str):
    plots = Path(CFG.results_dir) / "plots"
    placeholders = [
        "expert_load_histogram.csv",
        "routing_entropy_curve.csv",
        "framewise_expert_heatmap.csv",
        "attention_over_time_map.csv",
        "tsne_expert_outputs.csv",
        "cider_vs_active_params.csv",
    ]
    for name in placeholders:
        (plots / f"{tag}_{name}").write_text("step,value\n0,0\n")


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
    tokenizer = SimpleTokenizer(CFG.vocab_size)

    summary = []
    for dataset in DATASETS:
        train_ds = build_dataset(dataset, "train", tokenizer)
        val_ds = build_dataset(dataset, "val", tokenizer)

        for seed in CFG.seeds:
            set_seed(seed)
            model = VideoCaptioningMoE(vocab_size=CFG.vocab_size)
            run_name = f"{dataset}_seed{seed}"
            metrics = run_training(model, train_ds, val_ds, seed, run_name)
            metrics["dataset"] = dataset
            summary.append(metrics)
            save_visualization_placeholders(run_name)

        # dense baseline
        set_seed(CFG.seed)
        dense_model = VideoCaptioningMoE(vocab_size=CFG.vocab_size, dense_only=True)
        _ = run_training(dense_model, train_ds, val_ds, CFG.seed, f"{dataset}_dense_baseline")

    Path(CFG.results_dir, "metrics", "all_runs.json").write_text(json.dumps(summary, indent=2))
    Path(CFG.results_dir, "raw", "config.json").write_text(json.dumps(as_dict(), indent=2, default=str))


if __name__ == "__main__":
    run_all()
