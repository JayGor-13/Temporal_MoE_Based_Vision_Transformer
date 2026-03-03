import json
import importlib.util
import sys
import types
import zipfile
from pathlib import Path

if "datasets" not in sys.modules:
    fake_datasets = types.ModuleType("datasets")
    fake_datasets.load_dataset = lambda *args, **kwargs: None
    sys.modules["datasets"] = fake_datasets

if "huggingface_hub" not in sys.modules:
    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.hf_hub_download = lambda *args, **kwargs: ""
    sys.modules["huggingface_hub"] = fake_hub

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "prepare_datasets.py"
sys.path.insert(0, str(MODULE_PATH.parent.parent))
SPEC = importlib.util.spec_from_file_location("prepare_datasets", MODULE_PATH)
prepare_datasets = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = prepare_datasets
SPEC.loader.exec_module(prepare_datasets)
prepare_dataset = prepare_datasets.prepare_dataset


def test_prepare_datasets_from_hf_layout(tmp_path, monkeypatch):
    root = tmp_path / "data_store"
    fake_zip = tmp_path / "videos.zip"

    with zipfile.ZipFile(fake_zip, "w") as zf:
        zf.writestr("clip_0001.mp4", b"")
        zf.writestr("clip_0002.mp4", b"")
        zf.writestr("clip_0003.mp4", b"")

    def fake_load_split(repo_id, config_name, split_name, token):
        if split_name == "train":
            return [{"video": "clip_0001.mp4", "caption": ["a person walks"]}]
        if split_name == "validation":
            return [{"video": "clip_0002.mp4", "caption": ["a person runs"]}]
        if split_name == "test":
            return [{"video": "clip_0003.mp4", "caption": ["a person jumps"]}]
        raise AssertionError(f"Unexpected split: {split_name}")

    monkeypatch.setattr(prepare_datasets, "hf_hub_download", lambda **_: str(fake_zip))
    monkeypatch.setattr(prepare_datasets, "_load_split", fake_load_split)

    prepare_dataset(
        name="msvd",
        root=root,
        token=None,
        refresh=True,
        split_seed=42,
    )

    train_manifest = root / "msvd" / "train.json"
    val_manifest = root / "msvd" / "val.json"
    test_manifest = root / "msvd" / "test.json"
    assert train_manifest.exists()
    assert val_manifest.exists()
    assert test_manifest.exists()

    sample = json.loads(train_manifest.read_text())[0]
    assert sample["video_path"].startswith("videos/")
    assert sample["caption"] == "a person walks"
