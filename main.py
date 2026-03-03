from collections import defaultdict
import json

from config import *
from data.tokenizer import SimpleTokenizer
from data.msvd_dataset import MSVDVideoCaptionDataset
from models.video_captioning_moe import VideoCaptioningMoE
from train.train_loop import train_model

from config import CFG, as_dict
from data.tokenizer import SimpleTokenizer
from data.msvd_dataset import build_dataset, collect_captions
from models.video_captioning_moe import VideoCaptioningMoE
from train.train_loop import run_training
from utils.misc import set_seed


    dataset = MSVDVideoCaptionDataset(
        video_dir="./dataset/video_files/video_files",
        csv_path="./dataset/video_corpus.csv",
        txt_path="./dataset/annotations.txt",
        tokenizer=tokenizer,
        frames=NUM_FRAMES
    )


def ensure_results_tree():
    root = Path(CFG.results_dir)
    for sub in ["metrics", "raw", "plots", "benchmarks", "ablations"]:
        (root / sub).mkdir(parents=True, exist_ok=True)

    train_model(model, loader, None, tokenizer, device=device, epochs=EPOCHS)

def save_benchmark_template():
    path = Path(CFG.results_dir) / "benchmarks" / "reported_baselines.json"
    if path.exists():
        return

    baselines = {
        "CLIP4Clip": {"citation": "fill_me", "CIDEr": None, "METEOR": None, "SPICE": None},
        "BLIP-2 (video)": {"citation": "fill_me", "CIDEr": None, "METEOR": None, "SPICE": None},
        "Frozen": {"citation": "fill_me", "CIDEr": None, "METEOR": None, "SPICE": None},
        "Video Swin Transformer": {"citation": "fill_me", "CIDEr": None, "METEOR": None, "SPICE": None},
    }
    path.write_text(json.dumps(baselines, indent=2))


def _mean_std(values):
    values = [float(v) for v in values]
    if not values:
        return {"mean": 0.0, "std": 0.0}
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return {"mean": mean, "std": math.sqrt(var)}


def aggregate_results(all_runs):
    metric_keys = ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "METEOR", "ROUGE-L", "CIDEr", "SPICE"]
    grouped = defaultdict(list)
    for row in all_runs:
        grouped[(row["dataset"], row["variant"])].append(row)

    aggregates = []
    for (dataset, variant), rows in sorted(grouped.items()):
        out = {
            "dataset": dataset,
            "variant": variant,
            "num_seeds": len(rows),
        }
        for mk in metric_keys:
            stats = _mean_std([r.get(mk, 0.0) for r in rows])
            out[f"{mk}_mean"] = stats["mean"]
            out[f"{mk}_std"] = stats["std"]
        out["train_loss_mean"] = _mean_std([r.get("train_loss", 0.0) for r in rows])["mean"]
        out["active_params_mean"] = _mean_std([r.get("active_params", 0.0) for r in rows])["mean"]
        out["flops_approx_mean"] = _mean_std([r.get("flops_approx", 0.0) for r in rows])["mean"]
        aggregates.append(out)
    return aggregates


def write_benchmark_reports(all_runs):
    benchmark_dir = Path(CFG.results_dir) / "benchmarks"
    ablation_dir = Path(CFG.results_dir) / "ablations"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    ablation_dir.mkdir(parents=True, exist_ok=True)

    aggregates = aggregate_results(all_runs)
    (benchmark_dir / "model_aggregates.json").write_text(json.dumps(aggregates, indent=2))

    ablations = [a for a in aggregates if a["variant"].startswith("A")]
    (ablation_dir / "ablation_aggregates.json").write_text(json.dumps(ablations, indent=2))

    baselines_path = benchmark_dir / "reported_baselines.json"
    baselines = json.loads(baselines_path.read_text()) if baselines_path.exists() else {}

    full_model = [a for a in aggregates if a["variant"] == "full_moe"]
    comparison = []
    for row in full_model:
        dataset = row["dataset"]
        ours = {
            "model": "Temporal_MoE_ViT (this work)",
            "dataset": dataset,
            "CIDEr": row.get("CIDEr_mean"),
            "METEOR": row.get("METEOR_mean"),
            "SPICE": row.get("SPICE_mean"),
        }
        comparison.append(ours)
        for baseline_name, vals in baselines.items():
            comparison.append(
                {
                    "model": baseline_name,
                    "dataset": dataset,
                    "CIDEr": vals.get("CIDEr"),
                    "METEOR": vals.get("METEOR"),
                    "SPICE": vals.get("SPICE"),
                    "citation": vals.get("citation"),
                }
            )

    (benchmark_dir / "benchmark_comparison.json").write_text(json.dumps(comparison, indent=2))


def run_all():
    ensure_results_tree()
    save_benchmark_template()

    dataset_names = list(CFG.datasets)
    tokenizer = SimpleTokenizer(CFG.vocab_size)
    vocab_texts = collect_captions(dataset_names, split="train")
    if not vocab_texts:
        raise RuntimeError("No training captions found. Run scripts/prepare_datasets.py first.")
    tokenizer.build_vocab(vocab_texts)

    run_specs = [{"name": "full_moe", "model": {}, "train": {}}]
    if CFG.run_ablations:
        for name, spec in ABLATIONS.items():
            run_specs.append({"name": name, **spec})
    elif CFG.run_dense_baseline:
        run_specs.append({"name": "dense_baseline", "model": {"dense_only": True}, "train": {}})

    summary = []
    for dataset in dataset_names:
        train_ds = build_dataset(dataset, "train", tokenizer)
        val_ds = build_dataset(dataset, "val", tokenizer)
        test_ds = build_dataset(dataset, "test", tokenizer)

        for spec in run_specs:
            for seed in CFG.seeds:
                set_seed(seed)

                model = VideoCaptioningMoE(vocab_size=CFG.vocab_size, **spec.get("model", {}))
                run_name = f"{dataset}_{spec['name']}_seed{seed}"
                metrics = run_training(
                    model,
                    train_ds,
                    val_ds,
                    test_ds,
                    seed,
                    run_name,
                    tokenizer,
                    no_aux_loss=spec.get("train", {}).get("no_aux_loss", CFG.no_aux_loss),
                    no_contrastive_pretrain=spec.get("train", {}).get("no_contrastive_pretrain", CFG.no_contrastive_pretrain),
                )
                metrics["dataset"] = dataset
                metrics["variant"] = spec["name"]
                summary.append(metrics)

                print(
                    f"[{dataset}][{spec['name']}][seed={seed}] "
                    f"CIDEr={metrics.get('CIDEr', 0):.4f} METEOR={metrics.get('METEOR', 0):.4f} "
                    f"SPICE={metrics.get('SPICE', 0):.4f}"
                )

    Path(CFG.results_dir, "metrics", "all_runs.json").write_text(json.dumps(summary, indent=2))
    Path(CFG.results_dir, "raw", "config.json").write_text(json.dumps(as_dict(), indent=2, default=str))
    write_benchmark_reports(summary)


if __name__ == "__main__":
    run_all()
