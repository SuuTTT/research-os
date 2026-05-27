# Phase A Progress Update (2026-05-27)

## Scope started
- Batch: series_v7_phase_a_matrix_seed1
- Protocol: Look-Back-96 setup (`seq_len=96`), horizons `96/192/336/720`
- Models: DLinear, TimesNet
- Datasets: ETTh1, ETTh2, ETTm1, ETTm2
- Seeds queued: 1
- Total runs queued in this batch: 32

## Automation state
- Autorun cron enabled:
  - marker: ROS_SERIES_V7_AUTORUN
  - script: /home/ubuntu/research-os/scripts/series_v7_autorun.sh
  - log: research/structural_entropy_timeseries/series_pipeline/results/series_v7_autorun.log
- Each tick executes: `check-runs`, `dispatch-runs`, `collect_series_results`, `status`.

## Current results snapshot
From `series_v7_phase_a_matrix_seed1_summary.json`:
- Completed: 2 runs
  - DLinear ETTh1 horizon 96: MSE 0.411866, MAE 0.425835
  - DLinear ETTh1 horizon 192: MSE 0.457468, MAE 0.450346
- Remaining in batch queue:
  - pending: 28
  - running: 2
- Commit SHA captured by runs: `4e938a1`

## Notes
- Worker compatibility path is now stable for Python 3.10 by using a pinned dependency set in the queued commands.
- The batch is still progressing autonomously; this file is a midpoint snapshot, not final matrix results.

## Next immediate step
- Let autorun drain all 32 seed-1 runs, then compute a complete baseline table by dataset-horizon and begin seed expansion (seed 2/3) for mean/std reporting.
