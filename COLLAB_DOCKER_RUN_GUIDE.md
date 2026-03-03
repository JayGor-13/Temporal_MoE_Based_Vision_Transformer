# Collaboration Guide: Dockerized Training Pipeline

This guide explains how your team can run the full Temporal MoE Vision Transformer pipeline in a reproducible way using Docker, including how to handle **large (GB-scale) dataset downloads**.

---

## 1) What you should provide to collaborators

Share these items with your team:

1. **Source code repository** (this project).
2. **Pinned dependencies** (`requirements.txt`) and Docker setup (`Dockerfile`, `run_docker_pipeline.sh`).
3. **Runtime configuration expectations**:
   - `HF_TOKEN` (if any dataset is gated/private on Hugging Face).
   - `REFRESH_DATA` (whether to re-download/rebuild data manifests).
4. **Storage expectations**:
   - Enough free disk for downloaded datasets, extracted videos, model checkpoints, and logs.
5. **Execution sequence** (documented below), so everyone runs the same steps.

In practice, teams usually share either:
- a **Docker image** (already built and pushed to a registry), or
- the **repo + Dockerfile** and ask each person to build locally.

For frequent collaboration, pushing a prebuilt image is usually faster and avoids “works on my machine” dependency drift.

---

## 2) One-time prerequisites for each collaborator

- Install **Docker Engine** (or Docker Desktop).
- Ensure enough local disk (datasets are large).
- Clone this repository.

Optional but recommended:
- Get a Hugging Face token if your organization uses gated datasets.

---

## 3) Standard run sequence (recommended)

From the repository root:

### Step A — Configure environment variables

```bash
export HF_TOKEN=<your_hf_token_if_needed>
export REFRESH_DATA=0
```

Notes:
- Use `REFRESH_DATA=1` if you want to force dataset preparation again.
- Keep `REFRESH_DATA=0` for normal reruns to avoid unnecessary recomputation.

### Step B — Run with Docker helper script

```bash
bash run_docker_pipeline.sh
```

What this does:
- builds image `temporal-moe-pipeline:latest` (or custom name/tag),
- mounts `results/` so outputs persist,
- mounts `data_store/` so extracted videos/manifests persist,
- mounts `.cache/huggingface/` so large Hugging Face downloads are reused across runs.

---

## 4) Why cache mounting matters for GB-scale dataset downloads

Large dataset archives can be expensive to re-download repeatedly.

This project now mounts host cache:
- Host: `./.cache/huggingface`
- Container: `/root/.cache/huggingface`

So if one run downloads archive files once, subsequent runs and teammates on the same machine can reuse cached artifacts.

> Team best practice: keep a shared high-capacity disk path for cache (if on a shared server) and point `HF_CACHE_DIR` to it.

Example:

```bash
export HF_CACHE_DIR=/mnt/shared/hf-cache
bash run_docker_pipeline.sh
```

---

## 5) Sharing options inside a team

### Option 1: Share source + local build (simple)
Each developer runs:
```bash
git pull
bash run_docker_pipeline.sh
```

### Option 2: Share prebuilt image via registry (faster onboarding)
Owner builds and pushes:
```bash
docker build -t <registry>/temporal-moe-pipeline:<tag> .
docker push <registry>/temporal-moe-pipeline:<tag>
```

Collaborators pull and run with same volume mounts:
```bash
docker pull <registry>/temporal-moe-pipeline:<tag>
docker run --rm \
  -e HF_TOKEN="$HF_TOKEN" \
  -e REFRESH_DATA=0 \
  -v "$(pwd)/results:/app/results" \
  -v "$(pwd)/data_store:/app/data_store" \
  -v "$(pwd)/.cache/huggingface:/root/.cache/huggingface" \
  <registry>/temporal-moe-pipeline:<tag>
```

### Option 3: Internal shared runner (best for heavy workloads)
A central machine executes training; teammates only review artifacts in `results/` and logs.

---

## 6) Configuration knobs teammates should know

Primary controls:
- `config.py`: model/training/data budgets and ablation toggles.
- `REFRESH_DATA`: refresh prepared datasets or reuse existing local data.
- `HF_TOKEN`: authentication for dataset access.
- `HF_CACHE_DIR`: persistent cache location for HF downloads.

---

## 7) Expected outputs and where to find them

After a run, collaborators should look under:
- `results/metrics/all_runs.json`
- `results/benchmarks/model_aggregates.json`
- `results/benchmarks/benchmark_comparison.json`
- `results/ablations/ablation_aggregates.json`

These are mounted to the host and persist after the container exits.

---

## 8) Troubleshooting checklist

1. **Download/auth errors**: verify `HF_TOKEN` and dataset access permissions.
2. **Disk full**: clean old caches/checkpoints or move `HF_CACHE_DIR` to larger storage.
3. **Re-running downloads unintentionally**: set `REFRESH_DATA=0`.
4. **Container runs but no outputs**: ensure `results/` is mounted and writable.
5. **Slow first run**: expected for large downloads and extraction; subsequent runs should be faster due to cache and mounted data.

---

## 9) Quick commands reference

```bash
# Standard local Docker run
bash run_docker_pipeline.sh

# Force data refresh
REFRESH_DATA=1 bash run_docker_pipeline.sh

# Use a custom image tag
IMAGE_TAG=v1 bash run_docker_pipeline.sh

# Use a larger cache location
HF_CACHE_DIR=/mnt/shared/hf-cache bash run_docker_pipeline.sh
```
