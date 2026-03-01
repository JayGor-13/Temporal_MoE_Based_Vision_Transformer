from pathlib import Path

from scripts.prepare_datasets import main as prep_main


def test_prepare_datasets(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    import sys
    sys.argv = ["prepare_datasets.py", "--root", "data_store"]
    prep_main()
    assert Path("data_store/msvd/train.json").exists()
