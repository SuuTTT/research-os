# Hardware Requirements: <project_id>

Fill this file during project initialization or benchmark setup. It narrows the
global Vast worker policy for this specific research project.

## Required Runtime

| Field | Requirement |
|---|---|
| Accelerator | NVIDIA CUDA GPU |
| Minimum driver | `>= 580` |
| Minimum CUDA reported by Vast | `>= 13.0` |
| Minimum VRAM | `>= 8GB` |
| Minimum disk | `>= 50GB` |
| Python | `>= 3.11`, prefer `3.12` |

## Project-Specific Needs

- Datasets:
- Expected local dataset size:
- Checkpoint/log budget:
- Required system packages:
- Required Python stack:
- Known incompatible GPUs/drivers:
- Known bad Vast offers/hosts:

## Smoke Test

A worker is not eligible for this project until this passes:

```bash
cd /root/research-worker/<project_id>
source /root/venv/bin/activate
python - <<'PY'
import jax
print(jax.devices())
PY
```

Expected:

```text
[CudaDevice(id=0)]
```

Add project benchmark smoke below:

```bash
# Example:
# python benchmark/smoke.py --tiny
```

## Throughput Record

Record actual project throughput after the first smoke run:

| Worker | GPU | $/h | Benchmark | Throughput | Throughput/$h | Keep? |
|---|---|---:|---|---:|---:|---|
