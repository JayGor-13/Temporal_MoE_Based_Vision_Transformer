"""Prepare real train/val/test manifests from Hugging Face datasets.

This script downloads official metadata splits and video archives for supported
datasets, then writes manifests under:
  data_store/<dataset>/{train,val,test}.json

Each manifest row has:
  {
    "video_id": "...",
    "video_path": "videos/<filename>.mp4",
    "caption": "..."
  }
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import CFG
from datasets import load_dataset
from huggingface_hub import hf_hub_download


@dataclass(frozen=True)
class DatasetSpec:
    repo_id: str
    video_archive: str
    train_config: Optional[str]
    train_split: str
    test_config: Optional[str]
    test_split: str
    val_config: Optional[str]
    val_split: Optional[str]
    video_field: str
    caption_field: str


SPECS: Dict[str, DatasetSpec] = {
    "msvd": DatasetSpec(
        repo_id="friedrichor/MSVD",
        video_archive="MSVD_Videos.zip",
        train_config=None,
        train_split="train",
        test_config=None,
        test_split="test",
        val_config=None,
        val_split="validation",
        video_field="video",
        caption_field="caption",
    ),
    "msrvtt": DatasetSpec(
        repo_id="friedrichor/MSR-VTT",
        video_archive="MSRVTT_Videos.zip",
        train_config="train_9k",
        train_split="train",
        test_config="test_1k",
        test_split="test",
        val_config=None,  # derived from train split
        val_split=None,
        video_field="video",
        caption_field="caption",
    ),
}

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm")


def _manifest_has_video_paths(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        rows = json.loads(path.read_text())
    except Exception:
        return False
    if not rows:
        return False
    return all(isinstance(r, dict) and bool(str(r.get("video_path", "")).strip()) for r in rows)


def _is_prepared(dataset_dir: Path) -> bool:
    manifests_ok = all(_manifest_has_video_paths(dataset_dir / f"{s}.json") for s in ("train", "val", "test"))
    videos_dir = dataset_dir / "videos"
    videos_ok = videos_dir.exists() and any(videos_dir.rglob("*"))
    return manifests_ok and videos_ok

def _load_split(repo_id: str, config_name: Optional[str], split_name: str, token: Optional[str]) -> List[Dict[str, Any]]:
    if config_name:
        ds = load_dataset(repo_id, config_name, split=split_name, token=token)
    else:
        ds = load_dataset(repo_id, split=split_name, token=token)
    return [dict(row) for row in ds]


def _extract_selected_videos(archive_path: Path, videos_dir: Path, wanted_video_names: Set[str], force: bool = False) -> None:
    wanted = {n.lower() for n in wanted_video_names if n}
    if not wanted:
        raise RuntimeError("No target videos selected for extraction.")

    if force and videos_dir.exists():
        shutil.rmtree(videos_dir)
    videos_dir.mkdir(parents=True, exist_ok=True)

    existing = {p.name.lower() for p in videos_dir.rglob("*") if p.is_file()}
    missing = wanted - existing
    if not missing:
        return

    with zipfile.ZipFile(archive_path) as zf:
        for info in zf.infolist():
            basename = Path(info.filename).name.lower()
            if basename in missing:
                zf.extract(info, videos_dir)

    existing = {p.name.lower() for p in videos_dir.rglob("*") if p.is_file()}
    still_missing = wanted - existing
    if still_missing:
        raise RuntimeError(f"Failed to extract {len(still_missing)} videos from archive: {sorted(list(still_missing))[:5]}")


def _normalize_captions(raw_value: Any) -> List[str]:
    if raw_value is None:
        return []
    if isinstance(raw_value, str):
        txt = raw_value.strip()
        return [txt] if txt else []
    if isinstance(raw_value, (list, tuple)):
        out = []
        for item in raw_value:
            if item is None:
                continue
            txt = str(item).strip()
            if txt:
                out.append(txt)
        return out
    txt = str(raw_value).strip()
    return [txt] if txt else []


def _build_video_index(videos_dir: Path) -> tuple[Dict[str, Path], Dict[str, Path]]:
    name_index: Dict[str, Path] = {}
    stem_index: Dict[str, Path] = {}
    for p in videos_dir.rglob("*"):
        if not p.is_file():
            continue
        lower_name = p.name.lower()
        lower_stem = p.stem.lower()
        name_index.setdefault(lower_name, p)
        stem_index.setdefault(lower_stem, p)
    return name_index, stem_index


def _row_video_name(row: Dict[str, Any], video_field: str) -> str:
    return Path(str(row.get(video_field, "")).strip()).name


def _choose_video_names(
    rows: Sequence[Dict[str, Any]],
    video_field: str,
    max_videos: int,
    fraction: float,
    seed: int,
    exclude: Optional[Set[str]] = None,
) -> List[str]:
    exclude = exclude or set()
    pool = []
    seen = set()
    for row in rows:
        name = _row_video_name(row, video_field)
        if not name or name in exclude or name in seen:
            continue
        seen.add(name)
        pool.append(name)

    if not pool or max_videos <= 0:
        return []

    rnd = random.Random(seed)
    rnd.shuffle(pool)
    frac = min(max(fraction, 0.0), 1.0)
    n_frac = max(1, int(len(pool) * frac)) if frac > 0 else 0
    candidate = pool[:n_frac]
    return candidate[: min(max_videos, len(candidate))]


def _filter_rows_by_video_names(rows: Iterable[Dict[str, Any]], video_field: str, names: Set[str]) -> List[Dict[str, Any]]:
    return [row for row in rows if _row_video_name(row, video_field) in names]


def _resolve_video_path(
    video_ref: str,
    videos_dir: Path,
    name_index: Dict[str, Path],
    stem_index: Dict[str, Path],
) -> Optional[Path]:
    ref = video_ref.strip()
    if not ref:
        return None

    lower_name = Path(ref).name.lower()
    if lower_name in name_index:
        return name_index[lower_name]

    stem = Path(ref).stem.lower()
    if Path(ref).suffix and lower_name not in name_index:
        return None

    for ext in VIDEO_EXTS:
        key = f"{stem}{ext}"
        if key in name_index:
            return name_index[key]

    if stem in stem_index:
        return stem_index[stem]
    return None


def _rows_to_manifest(
    rows: Iterable[Dict[str, Any]],
    spec: DatasetSpec,
    videos_dir: Path,
    name_index: Dict[str, Path],
    stem_index: Dict[str, Path],
    max_captions_per_video: Optional[int] = None,
) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for row in rows:
        video_ref = str(row.get(spec.video_field, "")).strip()
        resolved = _resolve_video_path(video_ref, videos_dir, name_index, stem_index)
        if resolved is None:
            continue

        captions = _normalize_captions(row.get(spec.caption_field))
        if not captions:
            continue
        if max_captions_per_video is not None and max_captions_per_video > 0:
            captions = captions[:max_captions_per_video]

        video_id = Path(video_ref).stem if video_ref else resolved.stem
        rel_video_path = resolved.relative_to(videos_dir.parent).as_posix()
        for caption in captions:
            out.append({"video_id": video_id, "video_path": rel_video_path, "caption": caption})
    return out


def _write_manifest(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2))


def prepare_dataset(
    name: str,
    root: Path,
    token: Optional[str],
    refresh: bool,
    split_seed: int,
) -> None:
    if name not in SPECS:
        raise ValueError(f"Unsupported dataset '{name}'. Supported: {sorted(SPECS)}")

    spec = SPECS[name]
    dataset_dir = root / name
    videos_dir = dataset_dir / "videos"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    if _is_prepared(dataset_dir) and not refresh:
        print(f"[{name}] already prepared, skipping (use --refresh to rebuild).")
        return

    frac = float(CFG.download_fraction_by_dataset.get(name, 1.0))
    max_total = int(CFG.max_videos_per_dataset.get(name, 10))
    train_budget = int(CFG.train_videos_per_dataset.get(name, 4))
    val_budget = int(CFG.val_videos_per_dataset.get(name, 1))
    test_budget = int(CFG.test_videos_per_dataset.get(name, 2))
    max_captions_per_video = int(CFG.max_captions_per_video_by_dataset.get(name, 1))
    if train_budget + val_budget + test_budget > max_total:
        raise ValueError(
            f"[{name}] split video budgets exceed max_videos_per_dataset: "
            f"{train_budget}+{val_budget}+{test_budget} > {max_total}"
        )

    train_raw = _load_split(spec.repo_id, spec.train_config, spec.train_split, token=token)
    test_raw = _load_split(spec.repo_id, spec.test_config, spec.test_split, token=token)
    if spec.val_split is not None:
        val_raw = _load_split(spec.repo_id, spec.val_config, spec.val_split, token=token)
    else:
        val_raw = train_raw

    if spec.val_split is None:
        # Validation split is derived from the train split, so combined budgets
        # cannot exceed available train videos.
        unique_train = len({_row_video_name(r, spec.video_field) for r in train_raw if _row_video_name(r, spec.video_field)})
        if train_budget + val_budget > unique_train:
            raise ValueError(
                f"[{name}] train+val budgets exceed available train videos for derived val split: "
                f"{train_budget}+{val_budget} > {unique_train}. "
                "Adjust config.train_videos_per_dataset / val_videos_per_dataset for full split coverage."
            )

    archive = hf_hub_download(
        repo_id=spec.repo_id,
        filename=spec.video_archive,
        repo_type="dataset",
        token=token,
    )

    train_video_names = _choose_video_names(
        train_raw, spec.video_field, max_videos=train_budget, fraction=frac, seed=split_seed
    )
    val_video_names = _choose_video_names(
        val_raw,
        spec.video_field,
        max_videos=val_budget,
        fraction=frac,
        seed=split_seed + 1,
        exclude=set(train_video_names),
    )
    test_video_names = _choose_video_names(
        test_raw, spec.video_field, max_videos=test_budget, fraction=frac, seed=split_seed + 2
    )
    selected_video_names = set(train_video_names) | set(val_video_names) | set(test_video_names)
    if len(selected_video_names) > max_total:
        raise RuntimeError(f"[{name}] selected {len(selected_video_names)} videos, exceeds max_total={max_total}")

    _extract_selected_videos(Path(archive), videos_dir, wanted_video_names=selected_video_names, force=refresh)
    name_index, stem_index = _build_video_index(videos_dir)

    train_rows = _rows_to_manifest(
        _filter_rows_by_video_names(train_raw, spec.video_field, set(train_video_names)),
        spec,
        videos_dir,
        name_index,
        stem_index,
        max_captions_per_video=max_captions_per_video,
    )
    val_rows = _rows_to_manifest(
        _filter_rows_by_video_names(val_raw, spec.video_field, set(val_video_names)),
        spec,
        videos_dir,
        name_index,
        stem_index,
        max_captions_per_video=max_captions_per_video,
    )
    test_rows = _rows_to_manifest(
        _filter_rows_by_video_names(test_raw, spec.video_field, set(test_video_names)),
        spec,
        videos_dir,
        name_index,
        stem_index,
        max_captions_per_video=max_captions_per_video,
    )

    if not train_rows or not val_rows or not test_rows:
        raise RuntimeError(
            f"[{name}] produced empty splits. train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}. "
            "Check HF download and video extraction."
        )

    _write_manifest(dataset_dir / "train.json", train_rows)
    _write_manifest(dataset_dir / "val.json", val_rows)
    _write_manifest(dataset_dir / "test.json", test_rows)
    (dataset_dir / "README.txt").write_text(
        "Generated from Hugging Face dataset API and local video archive.\n"
    )
    print(
        f"[{name}] videos(train/val/test)="
        f"{len(set(train_video_names))}/{len(set(val_video_names))}/{len(set(test_video_names))} "
        f"| rows(train/val/test)={len(train_rows)}/{len(val_rows)}/{len(test_rows)}"
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data_store")
    ap.add_argument(
        "--datasets",
        nargs="+",
        default=sorted(SPECS),
        help=f"Datasets to prepare. Supported: {', '.join(sorted(SPECS))}",
    )
    ap.add_argument("--hf-token", default=None, help="HF token for private/gated datasets (optional).")
    ap.add_argument("--refresh", action="store_true", help="Rebuild manifests and re-extract archives.")
    ap.add_argument("--split-seed", type=int, default=42, help="Seed for deterministic train/val split derivation.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    requested = [d.lower() for d in args.datasets]

    for name in requested:
        prepare_dataset(
            name=name,
            root=root,
            token=args.hf_token,
            refresh=args.refresh,
            split_seed=args.split_seed,
        )

    print(f"Prepared datasets under: {root}")


if __name__ == "__main__":
    main()

