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

## Rule: Unstable Workers (reliability < 95 %)

Some workers have a listed reliability below 95 % (visible in `workers.yaml`
as `reliability_pct`). Treat them differently from stable workers:

1. **Always use `tmux` (or `screen`) for long jobs.** Start the job inside a
   named session:
   ```bash
   tmux new-session -d -s <job_name> 'CUDA_VISIBLE_DEVICES=N python3 -u script.py > log.txt 2>&1'
   ```
   SSH disconnects will **not** kill the process. Reattach later with
   `tmux attach -t <job_name>`.

2. **Save results after every (H, seed) combination**, not only at the end of
   the full run. If the process dies mid-run the partial file is valid and
   the run can be resumed from the last saved combo.

3. **Use `nohup` as a belt-and-suspenders fallback** alongside tmux, so that
   even if the tmux server itself crashes the child process survives.

4. **Assign only resumable or short tasks** (< 30 min per chunk) to unreliable
   workers. Tasks that cannot be easily restarted (e.g. one-shot full sweeps
   with no checkpointing) should go to a stable worker.

5. **Label the worker** in `workers.yaml` with `reliability_pct` so routing
   logic can select accordingly.

## Rule: Multi-GPU Workers (CUDA_VISIBLE_DEVICES)

When a worker has 2+ GPUs but the training code is **single-GPU** (no DDP /
DistributedDataParallel), assign one independent job per GPU via the
`CUDA_VISIBLE_DEVICES` environment variable:

```bash
# GPU 0 — first dataset
tmux new-session -d -s job_gpu0 'CUDA_VISIBLE_DEVICES=0 python3 -u script.py --datasets A > log_A.txt 2>&1'

# GPU 1 — second dataset
tmux new-session -d -s job_gpu1 'CUDA_VISIBLE_DEVICES=1 python3 -u script.py --datasets B > log_B.txt 2>&1'
```

Guidelines:
- Confirm per-GPU VRAM before launch: `nvidia-smi --query-gpu=index,memory.total --format=csv`.
- Never run two jobs on the same physical GPU unless you have measured that their
  combined peak VRAM stays under the device limit.
- Log which GPU each job is using in the run-log filename (e.g.
  `run_log_gpu0_datasetA.txt`).
- Use `nvidia-smi` after both jobs start to verify each is on its assigned GPU
  (`GPU-Util > 0 %`, correct memory used).

## Rule: Large-Disk Workers (≥ 80 GB)

Workers with `disk_gb ≥ 80` in `workers.yaml` can store full dataset copies
locally, avoiding repeated rsyncs:

1. **Sync datasets once** at worker registration; do not re-sync unless the
   local copy is missing or stale (check with `md5sum`).
2. **Store per-run checkpoints** locally at `runs/checkpoints/`. Sync back to
   the control plane only when a run is complete.
3. **Clean up** (remove checkpoints, intermediate logs) only when disk usage
   exceeds 80 % (`df -h`). Never delete raw dataset files automatically.

