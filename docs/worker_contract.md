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

To run on the worker:
```bash
SSH_WORKER="ssh -p 26249 root@ssh8.vast.ai -i ~/.ssh/vastai_id_ed25519"

# rsync project then run
rsync -az --exclude '__pycache__' /home/ubuntu/my-project/ \
  root@ssh8.vast.ai:/root/my-project/ -e "ssh -p 26249 -i ~/.ssh/vastai_id_ed25519"
$SSH_WORKER "cd /root/my-project && python3 -u run_experiments.py"
```

## Crash Contract

If a worker disappears:

1. Mark its `running` tasks as `stale`.
2. Keep any mirrored metrics as partial evidence.
3. Requeue only if the task is retryable.
4. Never count partial/stale tasks as final SOTA evidence unless the project
   manifest explicitly allows fixed-budget partial evaluation.

