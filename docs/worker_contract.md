# Worker Contract

A Research OS worker is a disposable machine that executes tasks from
`queues/run_queue.json`.

## Required Capabilities

Every worker must provide:

- stable `worker_id`;
- health check command;
- busy check command;
- workspace path;
- runner type;
- artifact output paths.

Worker types:

- `local`: run on the master machine.
- `ssh`: run on a remote machine by SSH.
- `docker`: run inside a container.
- `slurm`: submit to an HPC cluster.
- `kubernetes`: submit to a cluster job.

## Task Execution Contract

The run daemon will eventually execute:

```text
cd <workspace>
<env> <command>
```

The task must:

1. Write stdout/stderr to a log file.
2. Write metrics in JSONL or CSV.
3. Exit nonzero on real failure.
4. Save enough metadata to reproduce:
   - project id;
   - run id;
   - commit SHA;
   - config;
   - seed;
   - dataset version;
   - metric parser.

Recommended output layout:

```text
runs/<project_id>/<run_id>/
  run.json
  stdout.log
  metrics.jsonl
  checkpoints/
  figures/
```

## Health Contract

The master must be able to ask:

```bash
heartbeat_command
busy_command
```

For GPU workers, also:

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
```

If heartbeat fails, the worker is unavailable. Do not assign tasks.

## Compute Routing Rules

Always follow this routing table before queuing or running any task:

| Task type | Where to run | Reason |
|---|---|---|
| Full dataset training (ETT, Weather, Traffic, ship-motion) | `vastai_worker_2` | Local has 2GB RAM — OOM guaranteed |
| Model evaluation / rolling inference over full test set | `vastai_worker_2` | Same: large sliding-window arrays |
| Hyperparameter sweeps (≥4 runs) | `vastai_worker_2` | Accumulate memory across runs |
| GPU training (TimesNet, PatchTST, iTransformer, NBEATSx) | `vastai_worker_2` | Requires RTX 3060 VRAM |
| Smoke tests on synthetic data (≤500 rows) | `local_cpu` | Fine, no GPU needed |
| Code edits, file management, git operations | `local_cpu` | No compute required |
| Graph construction / community detection on real data | `vastai_worker_2` | DTW + TMI on long series is RAM-heavy |

**Hard rule**: do not run any script locally that loads a full benchmark CSV and
trains or evaluates a model. The local machine is a t3.small (2GB RAM, no GPU).
Exceeding ~1.5GB working set triggers the OOM killer (exit code 137) with no warning.

To run on a worker (example using `vastai_worker_3`):
```bash
SSH_WORKER="ssh -p 22607 root@ssh4.vast.ai -i ~/.ssh/vastai_id_ed25519"

# rsync project then run (code-only sync, ~few MB, no cost warning needed)
rsync -az --exclude '__pycache__' /home/ubuntu/my-project/ \
  root@ssh4.vast.ai:/root/my-project/ -e "ssh -p 22607 -i ~/.ssh/vastai_id_ed25519"
$SSH_WORKER "cd /root/my-project && nohup python3 -u run_experiments.py > run.log 2>&1 &"
```

**Before syncing datasets or checkpoints**: estimate transfer cost first.
`vastai_worker_3` uplink is 35.2 Mbps / $0.10 per GB (estimated).
A 5 GB dataset sync costs ~$0.50 — right at the warning threshold; confirm with user.

## Crash Contract

If a worker disappears:

1. Mark its `running` tasks as `stale`.
2. Keep any mirrored metrics as partial evidence.
3. Requeue only if the task is retryable.
4. Never count partial/stale tasks as final SOTA evidence unless the project
   manifest explicitly allows fixed-budget partial evaluation.

## Rule: Do Not Disturb Running Workers

If a worker already has a process running (visible via `pgrep`, `ps aux`, or
`nvidia-smi`), **do not kill it, do not modify its environment, and do not
install or upgrade packages** in the same Python/conda environment.

- If you must install new dependencies, use a separate venv or a new worker.
- Only attach to an existing run (e.g. `tail -f nohup.out`) to observe it.
- If the run must be replaced (machine rented anew), destroy the instance first
  and rent a fresh one; never recycle a working instance mid-run.

## Rule: Air-Gapped / Restricted-Network Workers

Some machines (e.g. hosted in China) have no access to PyPI, GitHub, Google,
or HuggingFace. Before connecting or dispatching to any worker:

1. **Check connectivity** at registration time: run
   `curl -s --max-time 5 https://pypi.org` from the worker. If it times out,
   mark the worker `network: restricted` in `workers.yaml`.

2. **Package installation on restricted workers**:
   - Prefer wheels pre-downloaded on the control plane and SCPed over.
   - Use the closest accessible mirror (e.g. `https://pypi.tuna.tsinghua.edu.cn`
     for Chinese nodes; set via `pip install --index-url <mirror>`).
   - Document the mirror URL in the worker's `notes` field.

3. **Bandwidth cost warning**: before any large `rsync` or `scp` transfer,
   estimate the cost using the worker's `bandwidth_$/gb` field (from Vast.ai
   listing or measured). **If the estimated transfer cost exceeds $0.50,
   stop and report the estimate to the user before proceeding.**

   Quick estimate formula:
   ```
   transfer_cost = data_gb × bandwidth_$/gb
   # Vast.ai typical egress: $0.10–0.13/GB
   # Example: 5 GB model checkpoint × $0.13 = $0.65 → WARN
   ```

4. For code syncs (typically < 50 MB), no warning needed. For datasets,
   checkpoints, or Docker images, always estimate first.

