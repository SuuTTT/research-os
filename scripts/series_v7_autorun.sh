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
  # Clean TSLib results/ on worker when idle (between jobs) to prevent disk fill.
  # results/ holds large pred/true .npy arrays (~400MB/run) never needed after metric extraction.
  RUNNING=$(python3 -c "import json,sys; q=json.load(open('queues/run_queue.json')); print(sum(1 for r in q if r.get('status')=='running'))" 2>/dev/null || echo "1")
  if [[ "${RUNNING}" == "0" ]]; then
    ssh -i ~/.ssh/vastai_id_ed25519 -p 26249 -o StrictHostKeyChecking=no -o ConnectTimeout=8 root@ssh8.vast.ai \
      "rm -rf /root/research-worker/Time-Series-Library/results/ 2>/dev/null; true" 2>/dev/null || true
    echo "[cleanup] results/ cleared on worker"
  fi
  python3 scripts/ros.py dispatch-runs --worker-pool vastai || true
  python3 research/structural_entropy_timeseries/series_pipeline/scripts/collect_series_results.py --batch-prefix series_v7_phase_a_v2_seed1 || true
  python3 research/structural_entropy_timeseries/series_pipeline/scripts/collect_series_results.py --batch-prefix series_v7_phase_a_v2_seed23 || true
  python3 research/structural_entropy_timeseries/series_pipeline/scripts/collect_series_results.py --batch-prefix series_v7_phase_b_pilot || true
  python3 research/structural_entropy_timeseries/series_pipeline/scripts/collect_series_results.py --batch-prefix series_v7_phase_b_pilot_v2 || true
  python3 scripts/ros.py status || true
} 9>"$LOCK" >> "$LOG" 2>&1
