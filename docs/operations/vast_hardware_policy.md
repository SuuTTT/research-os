# Vast Worker Hardware Policy

Date: 2026-05-26

This document defines the default rental bar for Research OS workers. Projects
can override it in their own `project.yaml`, but the global policy should reject
machines that are cheap on paper and expensive in setup time.

## Default Bar

Use this default for GPU research projects unless the project has a stronger
requirement:

| Field | Default | Reason |
|---|---:|---|
| `cuda_vers` | `>= 13.0` | Current JAX CUDA13 wheels need modern drivers. |
| Driver | `>= 580` | Avoid CUDA/JAX mismatch and old cuSPARSE issues. |
| GPU RAM | `>= 8GB` | Smaller cards often fail eval bursts or large batches. |
| Disk | `>= 50GB` | CUDA/JAX wheels plus repo/data/checkpoints make 20GB fragile. |
| Process/thread cgroup | `pids.max >= 512` or `max` | JAX/XLA can abort during compile when thread creation is tightly capped. |
| Reliability | `> 0.95` | Avoid churn during long probes. |
| Direct ports | `>= 2` | Need SSH plus dashboard or service forwarding. |
| DLPerf/$ | `> 200` | Basic cost-efficiency floor. |
| Setup smoke | required | Must complete code sync, env install, and `jax.devices()` check. |

Default hunter command:

```bash
python3 scripts/vast_hunter.py --storage 50 --min-disk-space 50 --max-dph 0.10 --min-dlperf-usd 200 --min-cuda 13.0
```

## Setup-Stability Bar

A worker is rejected even if it passes DLPerf/$ when any of these happens:

- SSH repeatedly times out during banner exchange.
- A small repo or benchmark sync repeatedly stalls or fails.
- `python -c 'import jax; print(jax.devices())'` does not show a CUDA device.
- The environment cannot be installed without manual one-off fixes.
- Available disk after dependency install is too small for logs/checkpoints.
- The process/thread cgroup limit is below `512`.
- Logs contain PJRT/XLA thread creation failures, especially
  `Thread pjrt_async_work_runner creation via pthread_create() failed`.

Process limit check:

```bash
ssh -p <port> root@<host> 'cat /sys/fs/cgroup/pids.max 2>/dev/null || cat /sys/fs/cgroup/pids/pids.max 2>/dev/null || true'
```

Known bad offer:

| Offer / Contract | Reason |
|---|---|
| Offer `34624617`, contract `37907664`, `ssh4.vast.ai:27665` | Good CUDA13/DLPerf on paper, but SSH repeatedly timed out and rsync of a 21MB benchmark tree stalled/broke. |
| Contract `37565664`, `ssh4.vast.ai:15665`, RTX 2060 12GB | Repeated PJRT pthread creation failures under `pids.max=256`; unsuitable for JAX queue workers. |

## Project-Specific Requirements

Each project should declare hardware requirements in:

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
