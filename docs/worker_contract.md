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

## Crash Contract

If a worker disappears:

1. Mark its `running` tasks as `stale`.
2. Keep any mirrored metrics as partial evidence.
3. Requeue only if the task is retryable.
4. Never count partial/stale tasks as final SOTA evidence unless the project
   manifest explicitly allows fixed-budget partial evaluation.

