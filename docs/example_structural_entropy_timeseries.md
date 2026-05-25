# Example: Structural Entropy for Time-Series Forecasting

This is the example project created by the scaffold.

## Human Idea

```text
Integrate structural entropy into N-BEATS, N-BEATSx, and TimesNet.
Use the system to identify SOTA benchmarks, build a local benchmark, run probes,
and generate a paper if the method beats strong baselines.
```

## Project

```text
research/structural_entropy_timeseries/project.yaml
```

Current target:

```text
Beat strong reported baselines on ETT, Weather, Traffic, and Electricity under
fair MSE/MAE evaluation.
```

## Queue State

```bash
cd /root/research-os
python3 scripts/ros.py status
python3 scripts/ros.py list-ideas
```

Expected after scaffold:

```text
idea_queue.json: 2
benchmark_queue.json: 2
run_queue.json: 0
publication_queue.json: 1
```

## First Deep Research Step

Open:

```text
research/structural_entropy_timeseries/deep_research/request.md
```

Paste it into GPT Deep Research. Save output as:

```text
research/structural_entropy_timeseries/deep_research/report.md
research/structural_entropy_timeseries/deep_research/sota_table.csv
research/structural_entropy_timeseries/deep_research/refs.bib
```

## First Benchmark Tasks

Already queued:

1. Build SOTA table for N-BEATS, N-BEATSx, TimesNet.
2. Clone and smoke-test TimesNet baseline.

After deep research, add concrete repo URLs:

```bash
python3 scripts/ros.py add-benchmark-task \
  --project-id structural_entropy_timeseries \
  --title "Clone official N-BEATSx repo and run smoke test" \
  --repo-url "<repo-url-from-deep-research>" \
  --metric "MSE/MAE" \
  --pass-rule "one dataset/horizon trains and emits parseable MSE/MAE"
```

## First Probe Ideas

Start with cheap probes:

1. Add structural entropy regularizer to N-BEATS block activations.
2. Add structural entropy regularizer to N-BEATSx exogenous component graph.
3. Add structural entropy over TimesNet period components.
4. Use entropy as a curriculum term that anneals to zero.

Only enqueue a run once:

- baseline runs locally;
- metric parser exists;
- probe has pass/kill rule;
- output path is fresh.

## Publication Trigger

Publication queue already contains a blog task:

```text
research/structural_entropy_timeseries/blog/benchmark-built.md
```

Generate it once the benchmark reproduction task passes.

