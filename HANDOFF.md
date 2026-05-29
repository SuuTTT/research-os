# research-os — Handoff Document

**Date**: 2026-05-29  
**What this is**: A file-backed research automation control plane that manages
ideas, experiments, workers, and publication artifacts for one or more research
projects.

---

## 1. What research-os Does

research-os turns a high-level research idea into a structured workflow:

```
Idea → project manifest → deep-research brief → benchmark plan
     → run queue → worker dispatch → metric collection → paper artifacts
```

Everything is stored as human-readable JSON/YAML/Markdown files — no database,
no daemon required. The system is inspectable, git-committable, and recoverable
after crashes.

The CLI entry point is:

```bash
python3 scripts/ros.py <command>
```

---

## 2. Directory Structure

```
research-os/
├── scripts/
│   └── ros.py                    ← Main CLI (all commands live here)
│
├── queues/                       ← File-backed task queues
│   ├── idea_queue.json           ← Research ideas and hypotheses
│   ├── benchmark_queue.json      ← Baseline reproduction tasks
│   ├── run_queue.json            ← GPU experiment runs (dispatched to workers)
│   └── publication_queue.json    ← Blog posts / paper triggers
│
├── workers/
│   └── workers.yaml              ← Worker registry (capabilities, routing rules)
│
├── schemas/
│   ├── project.schema.yaml       ← Schema for project.yaml files
│   ├── worker.schema.yaml        ← Schema for workers.yaml
│   └── queues.md                 ← Human-readable queue field documentation
│
├── templates/                    ← Scaffolds for new projects and papers
│   ├── project/project.yaml
│   ├── paper/main.tex
│   ├── blog/milestone.md
│   └── agents/deep_research_prompt.md
│
├── docs/
│   ├── build_guide.md            ← Full setup and workflow guide
│   ├── aws_setup_guide.md        ← EC2 control-plane setup
│   ├── worker_contract.md        ← Worker capability contract (required fields)
│   └── operations/
│       └── vast_hardware_policy.md ← GPU rental policy and routing rules
│
└── research/                     ← One sub-folder per research project
    ├── structural_entropy_timeseries/    ← SE-TS paper (COMPLETE — see §4)
    └── nbeatsx_dc/                       ← Older DC prototype (see §5)
```

---

## 3. Key CLI Commands

```bash
# Project management
python3 scripts/ros.py init-project --project-id <id> --title "..." --idea "..."
python3 scripts/ros.py add-idea --project-id <id> --title "..." --hypothesis "..."
python3 scripts/ros.py list-ideas

# Worker operations
python3 scripts/ros.py worker-status
python3 scripts/ros.py setup-worker --worker-id vastai_worker_1
  # ↑ syncs ~/.env.local credentials to worker /root/.env.local

# Run dispatch and monitoring
python3 scripts/ros.py dispatch-runs --worker-pool vastai
python3 scripts/ros.py check-runs

# Credential smoke test
python3 scripts/ros.py key-smoke --env-file /home/ubuntu/.env.local
```

### Environment file (never committed)
```bash
cat > /home/ubuntu/.env.local <<'EOF'
WANDB_API_KEY=...
HF_TOKEN=...
GITHUB_TOKEN=...
VASTAI_API_KEY=...
EOF
chmod 600 /home/ubuntu/.env.local
```

---

## 4. Active Research Project: structural_entropy_timeseries

### Status: EXPERIMENTS COMPLETE — paper needs finishing

This project tracks the paper:
> "Is Spectral Entropy a Useful Regulariser for Neural Time-Series Forecasting?"

The standalone paper source is at `/home/ubuntu/research-paper-se-ts/` (see its
own `HANDOFF.md` for submission checklist). The research-os folder
`research/structural_entropy_timeseries/` holds all experiment data and analysis.

#### What is in this project folder

```
research/structural_entropy_timeseries/
├── project.yaml                  ← Project manifest and current_state
├── analysis/
│   ├── phase_a_progress_2026-05-27.md
│   ├── milestone_phase_b_launch_2026-05-28.md   ← Full Phase A+B results summary
│   ├── sota_target_and_publishability_2026-05-26.md
│   ├── SE-idea.md                ← Method spec for algorithm engineers
│   ├── IDEAS.md                  ← Open questions and design decisions
│   └── CHANGELOG.md
├── evidence/                     ← Experiment result JSON summaries
│   ├── series_v7_phase_a_v2_aggregate_summary.json  ← Phase A final (96 runs)
│   ├── series_v7_phase_b_pilot_v2_analysis.json     ← Phase B final (24 runs)
│   └── ...                       ← Older iterations (v1–v6)
├── series_pipeline/
│   ├── configs/series_v1.json    ← Batch config schema example
│   ├── scripts/
│   │   ├── submit_series_batch.py  ← Enqueues runs from a batch config
│   │   └── collect_series_results.py ← Parses worker output → evidence JSON
│   └── results/                  ← Raw leaderboard JSONs per batch
├── paper/main.tex                ← Paper LaTeX (older version; canonical is research-paper-se-ts/)
└── deep_research/
    └── request.md                ← Deep research brief (SOTA survey)
```

#### Phase summary

| Phase | Runs | Status | Finding |
|---|---|---|---|
| Phase A: baseline reproduction | 96 | ✅ Done | TimesNet ±2.1% of paper; DLinear ETTm gap is by design (fixed L vs. tuned L) |
| Phase B v1: entropy minimisation | 12 | ✅ Done | **NO-GO** — λ=0.10 causes +17% MSE, rank collapse |
| Phase B v2: entropy maximisation | 12 | ✅ Done | **NO-GO** — best case −1.6% MSE, mostly neutral |

**Conclusion**: Pre-projection spectral entropy is not a productive regularisation
target for TimesNet on ETT. Clean null result ready to publish.

#### run_queue.json state
All queued runs have `"status": "done"`. There is nothing pending to dispatch.
The queue is complete.

---

## 5. Older Project: nbeatsx_dc (inside research-os)

`research/nbeatsx_dc/` contains an **early prototype** of the DC feature
selection pipeline, written before the main codebase was set up.

**This is NOT the authoritative code.** The current codebase is at
`/home/ubuntu/nbeatsx-dc/` which supersedes everything here.

What's in this folder:
- `src/` — older versions of backbone.py, community.py, decomposition.py, etc.
  These are outdated. Use `/home/ubuntu/nbeatsx-dc/src/` instead.
- `scripts/run_ett.py` — the ETT Ridge baseline runner (superseded by
  `/home/ubuntu/nbeatsx-dc/run_experiments.py`)
- `results/ett_all.json`, `ETTh1_H96_structural_entropy.json` — preliminary ETT
  results (superseded by `/home/ubuntu/nbeatsx-dc/results/tables/results.json`)

You can ignore this folder entirely. It is kept for git history reference only.

---

## 6. Worker Registry

Current `workers/workers.yaml` has one entry:

| Worker ID | Kind | GPU | Role |
|---|---|---|---|
| `local_cpu` | local | None | Control plane only — smoke tests, file ops, git |

Vast.ai workers were added dynamically during experiments and are no longer
active. To add a new worker:

1. Rent a GPU on Vast.ai (see `/home/ubuntu/nbeatsx-dc/HANDOFF.md §7` for GPU
   selection guidance)
2. Add an entry to `workers/workers.yaml` following `schemas/worker.schema.yaml`
3. Run `python3 scripts/ros.py setup-worker --worker-id <id>` to sync credentials

**Worker routing rule** (from `docs/worker_contract.md`): never run GPU tasks on
the local CPU worker. The AWS t3.small has 2 GB RAM and no GPU. All model
training must go to a Vast.ai or equivalent GPU worker.

---

## 7. How to Start a New Research Project

```bash
cd /home/ubuntu/research-os

# 1. Initialise project scaffold
python3 scripts/ros.py init-project \
  --project-id my_new_project \
  --title "My Research Title" \
  --idea "One-sentence description of the core idea." \
  --target "What metric and threshold constitutes success."

# This creates:
# research/my_new_project/
#   project.yaml, deep_research/, benchmark/, probes/,
#   evidence/, analysis/, blog/, paper/, artifacts/

# 2. Add a hypothesis to the idea queue
python3 scripts/ros.py add-idea \
  --project-id my_new_project \
  --title "Hypothesis A" \
  --hypothesis "Detailed claim..." \
  --metric "MSE/MAE"

# 3. Queue experiment runs
# Edit queues/run_queue.json directly or use submit_series_batch.py pattern
# from the structural_entropy_timeseries series_pipeline as a template.

# 4. Dispatch to a worker
python3 scripts/ros.py dispatch-runs --worker-pool vastai

# 5. Monitor
python3 scripts/ros.py check-runs
```

---

## 8. TODO

### For the SE-TS paper (urgent — ICDM deadline ~mid-June 2026)
- [ ] See `/home/ubuntu/research-paper-se-ts/HANDOFF.md` for the full checklist

### For research-os itself
- [ ] Add the NBEATSx-DC project to research-os tracking:
  ```bash
  python3 scripts/ros.py init-project --project-id nbeatsx_dc \
    --title "NBEATSx-DC: Community-Detection Feature Selection" \
    --idea "See /home/ubuntu/nbeatsx-dc/HANDOFF.md"
  ```
- [ ] Update `workers/workers.yaml` when a new Vast.ai GPU is rented
- [ ] Delete or archive `research/nbeatsx_dc/src/` (superseded by main repo)
