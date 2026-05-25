# Queue Schemas

## idea_queue.json

```json
{
  "id": "i123abc",
  "project_id": "structural_entropy_timeseries",
  "status": "new",
  "priority": 5,
  "title": "Structural entropy regularizer",
  "hypothesis": "...",
  "metric": "MSE/MAE",
  "claimed_by": "",
  "probe_specs": [],
  "evidence": [],
  "events": []
}
```

## benchmark_queue.json

```json
{
  "id": "b123abc",
  "project_id": "structural_entropy_timeseries",
  "status": "pending",
  "priority": 5,
  "title": "Clone and run TimesNet baseline",
  "repo_url": "",
  "command": "",
  "pass_rule": "baseline reproduces reported MSE within tolerance",
  "output_path": "research/<project>/benchmark/reproduction.md"
}
```

## run_queue.json

```json
{
  "id": "r123abc",
  "project_id": "structural_entropy_timeseries",
  "idea_id": "i123abc",
  "probe_id": "p123abc",
  "status": "pending",
  "priority": 10,
  "worker_pool": "gpu",
  "command": "bash research/<project>/probes/run_probe.sh",
  "env": "SEED=1 DATASET=ETTh1",
  "metric_parser": "research/<project>/analysis/parse_metrics.py",
  "target_metric": "mse",
  "target_direction": "lower"
}
```

## publication_queue.json

```json
{
  "id": "pub123",
  "project_id": "structural_entropy_timeseries",
  "status": "pending",
  "kind": "blog | paper | release",
  "trigger": "benchmark_built | sota_beaten | confirmation_done",
  "source_evidence": [],
  "output_path": "research/<project>/blog/YYYY-MM-DD-title.md"
}
```

