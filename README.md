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

