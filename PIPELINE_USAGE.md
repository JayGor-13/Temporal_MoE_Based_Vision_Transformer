# Pipeline Usage

## 1) Prepare real datasets from Hugging Face
```bash
python scripts/prepare_datasets.py --root data_store
```
Supported datasets:
- `msvd` (`friedrichor/MSVD`)
- `msrvtt` (`friedrichor/MSR-VTT`)

Generated files:
- `data_store/<dataset>/videos/*`
- `data_store/<dataset>/{train,val,test}.json`

Sampling controls are in `config.py` (per dataset):
- `download_fraction_by_dataset`
- `max_videos_per_dataset`
- `train_videos_per_dataset`
- `val_videos_per_dataset`
- `test_videos_per_dataset`

Default config is laptop-friendly: max `10` videos per dataset, train on `4`, validate on `1`, test on `2`.

If you need gated access, pass an HF token:
```bash
python scripts/prepare_datasets.py --root data_store --hf-token <your_token>
```

## 2) Run end-to-end locally
```bash
bash run_pipeline.sh
```

## 3) Run end-to-end in Docker (single command wrapper)
```bash
bash run_docker_pipeline.sh
```

All artifacts are written under `results/`.
