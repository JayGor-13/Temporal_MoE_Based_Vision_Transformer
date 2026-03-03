# Pipeline Usage

## 1) Prepare full MSVD + MSR-VTT datasets from Hugging Face
```bash
python scripts/prepare_datasets.py --root data_store --datasets msvd msrvtt --refresh
```
If needed for gated access:
```bash
python scripts/prepare_datasets.py --root data_store --datasets msvd msrvtt --refresh --hf-token <your_token>
```

Generated files:
- `data_store/<dataset>/videos/*`
- `data_store/<dataset>/{train,val,test}.json`

## 2) Configure training/ablations
Main controls live in `config.py`:
- full-data sampling knobs (`download_fraction_by_dataset`, split budgets)
- training schedule (`pretrain_epochs`, `finetune_epochs`, batch sizes)
- ablation toggles (`run_ablations`, `run_dense_baseline`)
- multi-seed eval (`seeds`)
- defaults are set for full MSVD and full MSR-VTT split coverage

## 3) Run end-to-end locally
```bash
bash run_pipeline.sh
```
Notes:
- `REFRESH_DATA=1` by default in `run_pipeline.sh`.
- Set `REFRESH_DATA=0` to reuse existing manifests/videos.
- Set `HF_TOKEN=<token>` when needed.

## 4) Run in Docker
```bash
bash run_docker_pipeline.sh
```
This mounts both `results/` and `data_store/`.

## 5) Outputs
All artifacts are written under `results/`:
- per-run folders: checkpoints, metrics, history, predictions
- `results/metrics/all_runs.json`
- `results/benchmarks/model_aggregates.json`
- `results/benchmarks/benchmark_comparison.json`
- `results/ablations/ablation_aggregates.json`


## 6) Team handoff / sequence guide
For a full collaborator-focused Docker + data-download playbook, see `COLLAB_DOCKER_RUN_GUIDE.md`.
