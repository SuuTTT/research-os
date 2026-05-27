# Structural Entropy Timeseries Project Recap (2026-05-26)

## 1) What has been done

### Infrastructure and workflow
- Set up a standalone research repository at `/home/ubuntu/research` and synchronized project files from control plane.
- Kept control-plane orchestration in `/home/ubuntu/research-os` using queue-driven execution (`run_queue.json`) with remote workers only.
- Added series automation scripts and configs for iterative experiments:
  - `series_pipeline/scripts/submit_series_batch.py`
  - `series_pipeline/scripts/collect_series_results.py`
  - `series_pipeline/scripts/run_series_batch_remote.py`
- Added cron-based unattended runner for continuing dispatch/check/collect while away:
  - marker: `ROS_SERIES_V5_AUTORUN`
  - log: `research/structural_entropy_timeseries/series_pipeline/results/series_v5_autorun.log`

### Remote execution policy
- Experiments were queued and executed on VastAI workers only.
- No local model training was used for series batches.
- Logs and parsed metrics were pulled back into project evidence files.

### Series batches executed
- Completed: `series_v1`, `series_v3_retry`, `series_v4_seed_sweep`, `series_v5_longrun`.
- Earlier blocked attempts (`series_v2*`) were preserved as failed/stale queue history and used for diagnostics.

## 2) Current result

### Queue and runtime state now
- No pending runs and no running runs.
- Current queue summary (control plane):
  - done: 37
  - failed: 6
  - stale: 4

### Best run snapshot (series_v5)
- Source: `research/structural_entropy_timeseries/evidence/series_v5_summary.json`
- Best model: `custom_mlp_se`
- Dataset: `ETTh1`
- Horizon: `24`
- Best MSE: `0.046292`
- Best MAE: `0.157999`

### Aggregate baseline vs SE (series_v5, 8 seeds each)
- Baseline model: `custom_mlp_baseline`
  - n = 8
  - mean MSE = 0.054351
  - std MSE = 0.003784
- SE model: `custom_mlp_se`
  - n = 8
  - mean MSE = 0.051532
  - std MSE = 0.005121
- Relative mean improvement (MSE): 5.19%

Interpretation:
- The current custom remote harness shows a positive mean MSE improvement for SE over baseline on ETTh1 horizon 24.
- This is a useful intermediate signal, but not yet a publishable SOTA claim against standardized literature benchmarks.

## 3) Current target

Primary target:
- Demonstrate robust, reproducible improvement from structural-entropy integration over strong baselines on standardized long-horizon forecasting benchmarks (TSLib-compatible protocol), then compare against reported SOTA references under fair settings.

Near-term technical target:
- Move from custom harness evidence to benchmark-comparable evidence (ETTh1/ETTh2/ETTm1/ETTm2, fixed look-back protocol, multi-seed statistics, same budget/fairness controls).

## 4) Plan next

### Phase A: Benchmark-grade runner alignment
1. Build a minimal compatible benchmark environment on remote workers (Python and dependency matrix that avoids incompatible strict pins).
2. Freeze exact runtime snapshots (requirements lock and commit SHAs).
3. Keep queue-only execution and continue collecting parsed metrics into evidence JSON.

### Phase B: Baseline benchmark matrix
1. Run DLinear/TimesNet baseline matrix on ETT subsets and horizons.
2. Collect per-seed metrics and aggregate mean/std.
3. Produce a baseline reference table in `evidence/`.

### Phase C: Structural entropy integration matrix
1. Implement SE regularizer/module variants corresponding to target backbones.
2. Run matched-budget SE vs baseline experiments (same seeds, epochs, data split).
3. Add diagnostics (entropy trajectory, stability, seed sensitivity).

### Phase D: SOTA-gap report
1. Build a machine-readable comparison against the deep-research SOTA table assumptions.
2. Report where SE is better / tied / worse by dataset-horizon.
3. Select best candidate for full-scale run and publication track.

## 5) Immediate execution checklist
- [ ] Verify cron autorun continues to dispatch/check/collect correctly.
- [ ] Generate `series_v5` final table artifact in `analysis/`.
- [ ] Start benchmark-compatible baseline batch (`series_v6`) via queue-only mode.
- [ ] Capture reproducibility metadata (seed list, commit SHA, env lock).

## 6) Key artifacts
- Evidence summary (latest):
  - `research/structural_entropy_timeseries/evidence/series_v5_summary.json`
- Leaderboard (latest):
  - `research/structural_entropy_timeseries/series_pipeline/results/series_v5_leaderboard.json`
- Autorun log:
  - `research/structural_entropy_timeseries/series_pipeline/results/series_v5_autorun.log`
- Deep research base document:
  - `research/structural_entropy_timeseries/deep_research/example_structural_entropy_forecasting_review_en.md`
