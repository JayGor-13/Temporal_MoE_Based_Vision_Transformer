# Docker Run Guide (Commands Only)

```bash
# 0) Clone and enter repo
git clone <repo_url>
cd Temporal_MoE_Based_Vision_Transformer
```

```bash
# 1) Optional: set HF token (required for gated/private HF datasets)
export HF_TOKEN=<your_hf_token>
```

```bash
# 2) First run: prepare full datasets
export REFRESH_DATA=1
```

```bash
# 3) Optional: set shared/local HF cache path (for large downloads)
export HF_CACHE_DIR=$(pwd)/.cache/huggingface
```

```bash
# 4) Build + run pipeline in Docker
bash run_docker_pipeline.sh
```

```bash
# 5) Re-runs: skip re-download and reuse prepared data
REFRESH_DATA=0 bash run_docker_pipeline.sh
```

```bash
# 6) Optional: custom image name/tag
IMAGE_NAME=temporal-moe-pipeline IMAGE_TAG=v1 bash run_docker_pipeline.sh
```

```bash
# 7) Outputs on host
ls -lah results/
ls -lah data_store/
```

```bash
# 8) Main output files
ls -lah results/metrics/all_runs.json
ls -lah results/benchmarks/model_aggregates.json
ls -lah results/benchmarks/benchmark_comparison.json
ls -lah results/ablations/ablation_aggregates.json
```

```bash
# 9) Share prebuilt image (maintainer side)
docker build -t <registry>/temporal-moe-pipeline:<tag> .
docker push <registry>/temporal-moe-pipeline:<tag>
```

```bash
# 10) Run prebuilt image (collaborator side)
docker pull <registry>/temporal-moe-pipeline:<tag>
docker run --rm \
  -e HF_TOKEN="$HF_TOKEN" \
  -e REFRESH_DATA="${REFRESH_DATA:-0}" \
  -v "$(pwd)/results:/app/results" \
  -v "$(pwd)/data_store:/app/data_store" \
  -v "${HF_CACHE_DIR:-$(pwd)/.cache/huggingface}:/root/.cache/huggingface" \
  <registry>/temporal-moe-pipeline:<tag>
```
