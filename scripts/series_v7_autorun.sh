#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ubuntu/research-os"
LOG="${ROOT}/research/structural_entropy_timeseries/series_pipeline/results/series_v7_autorun.log"
LOCK="/tmp/ros_series_v7_autorun.lock"

{
  flock -n 9 || exit 0
  cd "$ROOT"
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] tick"
  python3 scripts/ros.py check-runs || true
  python3 scripts/ros.py dispatch-runs --worker-pool vastai || true
  python3 research/structural_entropy_timeseries/series_pipeline/scripts/collect_series_results.py --batch-prefix series_v7_phase_a_matrix_seed1 || true
  python3 scripts/ros.py status || true
} 9>"$LOCK" >> "$LOG" 2>&1
