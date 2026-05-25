# Build Guide: General Research OS

Date: 2026-05-25

This is a clean-system guide for a brand-new research workflow. It does not
assume TD-MPC-Glass. The control plane should support arbitrary new research,
for example:

```text
I want to integrate structural entropy into time-series forecasting,
based on N-BEATS, N-BEATSx, and TimesNet.
```

The system should capture that idea, build a benchmark, run probes, use metrics
and errors as feedback, and eventually generate a blog and paper locally.

## 0. Mental Model

Research OS has four queues:

1. **Idea queue**: what hypotheses should be explored?
2. **Benchmark queue**: what environments/repos/datasets/baselines need to be
   built or reproduced?
3. **Run queue**: what concrete jobs should workers execute?
4. **Publication queue**: what blog/paper/release artifacts should be produced?

And five persistent stores:

1. **Project manifests**: what is the target and metric?
2. **Evidence store**: what happened?
3. **Artifact store**: checkpoints, tables, figures, logs.
4. **Worker registry**: what machines can run jobs?
5. **Agent logs**: what did each agent decide and why?

## 1. Folder Layout

```text
/root/research-os/
  README.md
  docs/
  schemas/
  queues/
    idea_queue.json
    benchmark_queue.json
    run_queue.json
    publication_queue.json
  research/
    <project_id>/
      project.yaml
      deep_research/
      benchmark/
      probes/
      evidence/
      analysis/
      blog/
      paper/
      artifacts/
  templates/
  scripts/
  workers/
  storage/
```

This folder can become a GitHub repo. Large outputs should not be committed.

## 2. Project Format

Every project starts with:

```text
research/<project_id>/project.yaml
```

The manifest defines:
- idea;
- target claim;
- benchmark suite;
- metrics;
- SOTA baselines;
- datasets;
- worker needs;
- publication plan.

See:

```text
schemas/project.schema.yaml
templates/project/project.yaml
```

## 3. Worker Contract

A worker is any machine that can execute a run task.

Minimum worker contract:

```text
worker_id: unique name
ssh: host + port + user + key
workspace: /root/research-worker
runner: bash / python / docker
resources: cpu, gpu, vram, disk
heartbeat: can report alive/dead
artifact output: writes logs/metrics/checkpoints to agreed paths
```

Every run task must be reproducible from:
- repo commit;
- project id;
- command;
- env vars;
- config path;
- dataset version;
- seed;
- expected metric parser.

Workers should be disposable. The master queue state is durable.

## 4. New Research Flow

### Step A: Capture Idea

Example:

```bash
python3 scripts/ros.py init-project \
  --project-id structural_entropy_timeseries \
  --title "Structural entropy for neural time-series forecasting" \
  --idea "Integrate structural entropy into N-BEATS, N-BEATSx, and TimesNet." \
  --target "Beat strong baselines on ETT, Weather, Traffic, Electricity."
```

This creates:

```text
research/structural_entropy_timeseries/project.yaml
research/structural_entropy_timeseries/deep_research/request.md
research/structural_entropy_timeseries/benchmark/README.md
research/structural_entropy_timeseries/probes/README.md
research/structural_entropy_timeseries/evidence/README.md
research/structural_entropy_timeseries/blog/
research/structural_entropy_timeseries/paper/main.tex
```

### Step B: Deep Research

Use the generated `deep_research/request.md` in a deep-research web app.

The output should identify:
- SOTA papers;
- benchmark datasets;
- official repos;
- metric definitions;
- train/val/test splits;
- known reproduction issues;
- compute budget;
- fair comparison criteria.

Save result:

```text
research/<project_id>/deep_research/report.md
research/<project_id>/deep_research/sota_table.csv
research/<project_id>/deep_research/refs.bib
```

### Step C: Build Benchmark

Create benchmark tasks:

```bash
python3 scripts/ros.py add-benchmark-task \
  --project-id structural_entropy_timeseries \
  --title "Clone and smoke-test TimesNet official repo" \
  --repo-url "https://github.com/..." \
  --metric "MSE/MAE"
```

Benchmark agent responsibilities:
- clone official repos;
- pin commit hashes;
- create environment;
- run smoke tests;
- run one small baseline;
- reproduce at least one published number;
- write `benchmark/reproduction.md`.

If official repos are slow or hard to modify:
- keep official repo as reference;
- build a local clean implementation wrapper;
- compare output against the reference.

### Step D: Generate Probes

For structural entropy time-series, first probes might be:

1. N-BEATS block structural-entropy regularizer.
2. N-BEATSx exogenous-variable graph entropy penalty.
3. TimesNet temporal-period graph entropy module.
4. Structural entropy as data-dependent curriculum.

Each probe needs:
- code change or config-only change;
- target baseline;
- dataset subset;
- metric parser;
- pass/kill rule;
- seed/budget.

### Step E: Push Runs

A run task is concrete:

```json
{
  "project_id": "structural_entropy_timeseries",
  "idea_id": "i...",
  "benchmark_id": "b...",
  "probe_id": "p...",
  "status": "pending",
  "priority": 10,
  "worker_pool": "gpu",
  "command": "bash research/structural_entropy_timeseries/probes/run_nbeats_entropy.sh",
  "env": "SEED=1 DATASET=ETTh1 HORIZON=336",
  "metric_parser": "research/structural_entropy_timeseries/analysis/parse_metrics.py",
  "target_metric": "mse",
  "target_direction": "lower"
}
```

The run daemon claims a worker, launches the command, streams logs, parses
metrics, and writes evidence.

### Step F: Feedback Loop

Each completed run updates:

```text
research/<project_id>/evidence/
queues/idea_queue.json
queues/run_queue.json
```

Decision options:
- fix error and retry;
- kill idea;
- mutate probe;
- add seed;
- promote to benchmark sweep;
- trigger blog/paper update.

### Step G: Publication

When benchmark is built:
- generate blog: "Benchmark reproduction notes".

When SOTA is beaten on a subset:
- generate blog: "First positive signal".

When full benchmark is confirmed:
- generate paper draft and release checklist.

## 5. Agent Design

Agents are not magic. Each agent has a narrow contract.

### Orchestrator Agent

Owns state transitions:
- idea -> benchmark task;
- benchmark done -> probe task;
- probe result -> next decision;
- milestone -> publication task.

### Literature Agent

Consumes deep-research report and fills:
- SOTA table;
- benchmark plan;
- BibTeX;
- fairness checklist.

### Benchmark Agent

Builds local benchmark and validates baselines.

### Coding Agent

Implements probes and smoke tests.

### Run Agent

Pushes run tasks and watches failures.

### Analysis Agent

Parses metrics, updates evidence, recommends next action.

### Publication Agent

Creates blog and LaTeX manuscript drafts from evidence.

## 6. Minimum Viable System

Build in this order:

1. File schemas and CLI.
2. Project scaffold command.
3. Idea queue command.
4. Benchmark queue command.
5. Run queue command.
6. Worker registry.
7. Local worker runner.
8. SSH worker runner.
9. Dashboard.
10. Agent runner.
11. Publication generator.
12. S3/W&B/HF integration.

Do not start with a fully autonomous coding agent. Start with a visible
human-approved queue and add autonomy once the state machine is reliable.

## 7. GitHub Push Plan

If publishing this system as a repo:

```text
/root/research-os
```

Commit:
- docs;
- schemas;
- scripts;
- templates;
- example project.

Do not commit:
- `research/*/artifacts/`
- model checkpoints;
- datasets;
- W&B logs;
- large benchmark outputs.

