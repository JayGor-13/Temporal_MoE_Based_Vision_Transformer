# Pipeline Usage

## 1) Create local dataset directories (no heavy downloads)
```bash
python scripts/prepare_datasets.py --root data_store
```
This creates:
- `data_store/msvd/{train,val,test}.json`
- `data_store/msrvtt/{train,val,test}.json`
- `data_store/vatex/{train,val,test}.json`

These manifests are lightweight placeholders so the full pipeline can run on a small machine.

## 2) Optional: pull subset metadata from HuggingFace
```bash
python scripts/fetch_from_hf.py
```
For VATEX Kaggle files, place JSON files under `data_store/vatex/` and keep only English captions (`enCap`).

## 3) Run end-to-end locally
```bash
bash run_pipeline.sh
```

## 4) Run end-to-end in Docker (single command wrapper)
```bash
bash run_docker_pipeline.sh
```

All artifacts are written under `results/`.
