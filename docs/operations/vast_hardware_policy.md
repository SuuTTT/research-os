# Vast Worker Hardware Policy

Date: 2026-05-26  
Updated: 2026-05-29

This document defines the hardware profiles for Research OS workers. Use the
`vast-hunt` command or `vast_hunter.py` with `--profile` to apply a profile.

---

## Profiles

### `pytorch` — Neural forecasting with PyTorch (NBEATSx-DC and similar)

Use this for any project using PyTorch with small-to-medium models.

| Field | Value | Reason |
|---|---:|---|
| CUDA | ≥ 11.8 | PyTorch 2.x minimum |
| GPU RAM | ≥ 8 GB | Batch=512, lookback=168: activations < 100 MB — 8 GB is safe |
| Disk | ≥ 15 GB | Data CSV + Python env + results |
| Reliability | > 0.95 | Avoid churn during long probes |
| Direct ports | ≥ 1 | SSH only needed |
| DLPerf/$ floor | none | Any Ampere+ card is fine; filter by total cost instead |
| Storage allocation | 20 GB | Used for pricing calculation |

**Hunter command (with cost estimate)**:
```bash
python3 scripts/vast_hunter.py --profile pytorch --max-dph 0.10 \
    --ref-hours 4.5 --ref-dlp 12.4 --data-gb 0.5 \
    --min-dlperf-usd 0 --min-disk-space 15 --min-cuda 11.8

# Via ros.py (reads hardware: section from project.yaml automatically):
python3 scripts/ros.py vast-hunt --project-id nbeatsx_dc --max-dph 0.10 --rent
```

**Key lesson (2026-05-29)**: For PyTorch projects, filter by *total cost* (not
DLPerf/$). The RTX 3070 (DLP=16.7, $0.072/hr) is 5× cheaper in total than a
GTX 1080 (DLP=3.4, same $/hr) even though $/hr is identical. Old Pascal cards
(GTX 1080/1070) appear cheap per-hour but are slow — the run takes 5× longer.

**Best sub-$0.10/hr option found 2026-05-29**:
```
RTX 3070  ID=32711685  $0.072/hr  DLP=16.7  8GB VRAM  172GB disk  102Mbps
Estimated cost for NBEATSx-DC full run (~3.3h): $0.25 total
```

---

### `jax` — JAX/XLA workloads (SE-TS and similar)

Use this for projects using JAX, XLA, or any framework requiring modern CUDA/cuDNN.

| Field | Value | Reason |
|---|---:|---|
| CUDA | ≥ 13.0 | Current JAX CUDA13 wheels need modern drivers |
| Driver | ≥ 580 | Avoid CUDA/JAX mismatch and old cuSPARSE issues |
| GPU RAM | ≥ 8 GB | Smaller cards often fail eval bursts or large batches |
| Disk | ≥ 50 GB | CUDA/JAX wheels plus repo/data/checkpoints make 20 GB fragile |
| Reliability | > 0.95 | Avoid churn during long probes |
| Direct ports | ≥ 2 | Need SSH plus dashboard or service forwarding |
| DLPerf/$ floor | > 200 | Basic cost-efficiency floor |
| pids.max | ≥ 512 | JAX/XLA aborts during compile when thread creation is tightly capped |

**Hunter command**:
```bash
python3 scripts/vast_hunter.py --profile jax --storage 50
```

---

## Setup-Stability Rejection Rules (both profiles)

A worker is rejected even if it passes hardware specs when any of these happens:

- SSH repeatedly times out during banner exchange
- A small repo or benchmark sync repeatedly stalls or fails
- `nvidia-smi` is absent or shows wrong CUDA version
- Available disk after dependency install is too small for logs/checkpoints
- (JAX only) `python -c 'import jax; print(jax.devices())'` does not show CUDA
- (JAX only) Logs contain PJRT/XLA thread creation failures

**Process limit check**:
```bash
ssh -p <port> root@<host> \
  'cat /sys/fs/cgroup/pids.max 2>/dev/null || cat /sys/fs/cgroup/pids/pids.max 2>/dev/null || true'
```

---

## Known-Bad Offers (permanent exclusions)

| Offer ID | Contract | Endpoint | Reason |
|---|---|---|---|
| `34624617` | `37907664` | `ssh4.vast.ai:27665` | SSH route repeatedly timed out; rsync stalled on 21 MB tree |
| — | `37565664` | `ssh4.vast.ai:15665` RTX 2060 12GB | Repeated PJRT pthread failures, `pids.max=256` |

These IDs are hard-coded in `vast_hunter.py` `DEFAULT_EXCLUDED_OFFER_IDS`.

---

## Worker Routing Rules

| Task type | Where to run | Reason |
|---|---|---|
| Any GPU training | Vast.ai worker | AWS t3.small has no GPU and only 2 GB RAM |
| Smoke tests (≤500 rows) | local_cpu | Free, fast |
| Git operations, file edits | local_cpu | Free, always available |
| queue dispatch, check-runs | local_cpu | Orchestration only |
| Large dataset prep (>1 GB) | Vast.ai worker | RAM constraint on t3.small |


```text
research/<project_id>/hardware_requirements.md
```

Use the template:

```text
templates/project/hardware_requirements.md
```

The run queue should match tasks to workers only when:

- global worker policy passes;
- project-specific hardware requirements pass;
- the worker has completed the project smoke test.

## Keep/Reject Decision

Keep a rented worker only if all are true:

1. It passes the setup smoke.
2. It can sync code and artifacts reliably.
3. It has a recorded cost and throughput measurement.
4. It is competitive on either DLPerf/$ or actual project metric throughput per dollar.

Destroy quickly when setup fails. A cheap unstable worker blocks research loops
more than it saves money.
