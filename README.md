# Research OS

A standalone research automation control plane.

The system turns a high-level research idea into:

1. a project manifest,
2. a deep-research brief,
3. a benchmark implementation plan,
4. probe specs,
5. queued runs on workers,
6. metric/error feedback,
7. milestone blog drafts,
8. a local LaTeX manuscript.

It is intentionally file-backed at first: JSON/YAML/Markdown files are easy to
inspect, commit to GitHub, back up to S3, and repair after crashes.

## Quick Start

```bash
cd /root/research-os

python3 scripts/ros.py init-project \
  --project-id structural_entropy_timeseries \
  --title "Structural entropy for neural time-series forecasting" \
  --idea "Integrate structural entropy into N-BEATS, N-BEATSx, and TimesNet to improve long-horizon forecasting." \
  --target "Beat strong reported baselines on ETT, Weather, Traffic, and Electricity under fair MSE/MAE evaluation."

python3 scripts/ros.py add-idea \
  --project-id structural_entropy_timeseries \
  --title "Structural entropy regularizer for decomposition blocks" \
  --hypothesis "Graph/partition entropy over learned temporal components can improve decomposition diversity and reduce overfitting." \
  --metric "MSE/MAE on long-horizon forecasting benchmarks"

python3 scripts/ros.py list-ideas
```

Read the full build guide:

```text
docs/build_guide.md
```

## Vast GPU Workers

Research OS includes a Vast.ai hunter for disposable GPU workers:

```bash
python3 scripts/vast_hunter.py --storage 50 --max-dph 0.10 --min-dlperf-usd 200 --min-cuda 13.0
```

Read:

```text
docs/operations/vast_hardware_policy.md
templates/project/hardware_requirements.md
```

## Credentials And Smoke Tests

Keep runtime credentials in a local file that is never committed:

```bash
cat > /home/ubuntu/.env.local <<'EOF'
WANDB_API_KEY=...
HF_TOKEN=...
GITHUB_TOKEN=...
VASTAI_API_KEY=...
EOF
chmod 600 /home/ubuntu/.env.local
```

Quick credential smoke tests:

```bash
python3 scripts/ros.py key-smoke --env-file /home/ubuntu/.env.local
```

## Worker Runtime Commands

Research OS supports worker health checks and simple queue dispatch directly
from `ros.py`:

```bash
python3 scripts/ros.py worker-status
python3 scripts/ros.py setup-worker --worker-id vastai_worker_1
python3 scripts/ros.py dispatch-runs --worker-pool vastai
python3 scripts/ros.py check-runs
```

`setup-worker` synchronizes the control-plane `/home/ubuntu/.env.local` to
worker `/root/.env.local`, and dispatched runs auto-load `/root/.env.local`
before executing task commands.
