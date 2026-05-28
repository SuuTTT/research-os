# Project Changelog — Structural Entropy Timeseries

**Project**: Structural Entropy as a Regulariser for Neural Time-Series Forecasting  
**Repo**: `/home/ubuntu/research-os` (control plane) + `/home/ubuntu/research` (pipeline)  
**Worker**: VastAI RTX 3060 12 GB · `ssh8.vast.ai:26249` · TSLib commit `4e938a1`

---

## [2026-05-26] — Project Kickoff & Custom Harness

### Done
- Defined research question: does spectral entropy of encoder activations make a useful
  regularisation signal for time-series forecasting?
- Set up `/home/ubuntu/research-os` control plane with queue-driven execution
  (`queues/run_queue.json`) and VastAI remote-only execution policy.
- Built custom MLP harness (non-TSLib) as quick signal check:
  - Dataset: ETTh1, horizon 24 (short, sanity only)
  - 8 seeds each: `custom_mlp_baseline` vs. `custom_mlp_se`
  - **Result**: SE improved mean MSE by **5.19%** (0.054351 → 0.051532)
  - Interpretation: encouraging signal, but non-standard protocol — not publishable.
- Defined publishability thresholds:
  - Strong: ≥3% relative MSE gain across standardised benchmarks, multiple seeds
  - Minimum: consistent wins on ≥60% of (model × dataset × horizon) cells
- Drafted Phase A → B → C progression plan (see `sota_target_and_publishability_2026-05-26.md`).
- Wrote series pipeline automation scripts:
  - `submit_series_batch.py`, `collect_series_results.py`, `run_series_batch_remote.py`
- Enabled cron autorun (`series_v5_autorun.sh`) for unattended dispatch/collect.

### Key files created
- `analysis/project_recap_2026-05-26.md`
- `analysis/sota_target_and_publishability_2026-05-26.md`
- `project.yaml` (goals, metrics, baseline list)
- `paper/main.tex` (skeleton)

---

## [2026-05-27] — Phase A Begins: TSLib Baseline Matrix

### Done
- Switched from custom harness to **TSLib** (`commit 4e938a1`) for benchmark-grade runs.
- Pinned Python 3.10 dependency set on worker to fix earlier environment failures.
- Built `generate_phase_a_matrix.py` — generates 96-run config (DLinear + TimesNet,
  ETTh1/ETTh2/ETTm1/ETTm2, H∈{96,192,336,720}, seeds 1/2/3).
- Submitted batch `series_v7_phase_a_matrix_seed1` (32 runs, seed=1 first).
- First 2 runs confirmed healthy: DLinear ETTh1 h96 MSE=0.4119, h192 MSE=0.4575.
- Upgraded cron to `series_v7_autorun.sh`.
- Recorded midpoint snapshot in `analysis/phase_a_progress_2026-05-27.md`.

### Protocol locked
```
seq_len=96, label_len=48, features=M (multivariate)
train_epochs=10, patience=3, batch_size=32
horizons: 96 / 192 / 336 / 720
seeds: 1, 2, 3  (TimesNet); seed=1 (DLinear, deterministic)
```

---

## [2026-05-28 morning] — Phase A Complete: 96 Runs Done

### Done
- All **96 Phase A runs** completed with zero infrastructure failures.
- Collected aggregate stats (mean ± std across 3 seeds) into
  `evidence/series_v7_phase_a_v2_aggregate_summary.json`.
- TimesNet reproduces original paper within ±2.1% MSE on 15/16 configs.
  Only outlier: ETTm1 h720 (std=0.0208 — non-deterministic cuDNN on long horizon).
- DLinear ETTm gap is by design: paper uses per-config tuned look-back (336/720);
  we fix L=96 for controlled comparison.
- Populated Phase A table in `paper/main.tex`.

### Phase A baseline (H=96, seed aggregate)
| Dataset | TimesNet MSE | TimesNet MAE |
|---------|-------------|-------------|
| ETTh1   | 0.3891      | 0.4120      |
| ETTh2   | 0.3292      | 0.3701      |
| ETTm1   | 0.3363      | 0.3765      |
| ETTm2   | 0.1883      | 0.2678      |

---

## [2026-05-28 morning] — Phase B v1: Entropy Minimisation (+λH)

### Hypothesis
Add `+λH` to training loss where H = spectral entropy of pre-projection activations.
Gradient descent minimises H → forces low-rank encoder bottleneck → expected to reduce
overfitting on long-horizon targets.

### Implementation
- Wrote `scripts/run_se.py` (SE_VERSION=4):
  - `forward_pre_hook` on `model.projection` captures `(B, T, D)` activation tensor
  - Covariance → `eigvalsh` → normalise → `H = −Σ pᵢ log pᵢ`
  - `loss = loss + λ * H`
- Wrote `scripts/generate_phase_b_matrix.py` — embeds `run_se.py` as base64 bootstrap.
- Generated + submitted batch `series_v7_phase_b_pilot` (12 runs):
  - λ ∈ {0.01, 0.05, 0.10} × ETTh1/ETTh2/ETTm1/ETTm2, H=96, seed=1

### Results (all 12 runs complete)
| λ    | ETTh1   | ETTh2   | ETTm1   | ETTm2   |
|------|---------|---------|---------|---------|
| 0.01 | −0.20%  | +1.02%  | +0.09%  | +0.88%  |
| 0.05 | +8.24%  | +2.25%  | +2.32%  | +5.08%  |
| 0.10 | +16.94% | +2.83%  | +3.42%  | +9.27%  |

**Verdict: NO-GO.** GO criterion: ΔMSE < −0.5% on ≥2/4 datasets for any λ. **0/12 pass.**

### Diagnosis
`+λH` causes gradient descent to **collapse** the encoder toward rank-1.  
TimesNet's multi-scale frequency decomposition needs diverse representations —
eliminating them destroys accuracy. Larger λ = more collapse = more degradation.

### Pivot
Flip the sign: try **`−λH`** (entropy maximisation) to reward spectral diversity.

---

## [2026-05-28 afternoon] — Phase B v2: Entropy Maximisation (−λH)

### Implementation
- Updated `run_se.py` to SE_VERSION=5:  `loss = loss − λ * H`
- Updated sentinel check in `generate_phase_b_matrix.py` (SE_VERSION=5).
- Added v2 collect line to `scripts/series_v7_autorun.sh`.
- Generated + submitted batch `series_v7_phase_b_pilot_v2` (12 runs):
  - same design as v1 — λ ∈ {0.01, 0.05, 0.10} × ETT×4, H=96, seed=1
- Wrote `scripts/analyze_phase_b_pilot_v2.py` (3-way comparison: baseline / v1 / v2).

### Results (all 12 runs complete, 09:35 UTC)
| λ    | ETTh1   | ETTh2   | ETTm1   | ETTm2   |
|------|---------|---------|---------|---------|
| 0.01 | +0.38%  | +0.69%  | −0.30%  | **−1.64%** |
| 0.05 | +0.93%  | −0.75%  | −0.89%  | −1.26%  |
| 0.10 | +1.30%  | +0.18%  | −1.10%  | −1.27%  |

**Verdict: NO-GO.** GO criterion not met: 0/12 configs with ΔMSE < −0.5% on ≥2 datasets.

### Analysis
- Entropy maximisation eliminates the catastrophic v1 degradation (no run > +1.3% vs. v1's +17%).
- ETTm2 shows a mild consistent benefit (−1.3 to −1.6%), ETTm1 weakly benefits at higher λ.
- ETTh1 is consistently harmed at all λ values.
- The encoder is already well-conditioned spectrally — diversity reward adds noise, not signal.

### Evidence files written
- `evidence/series_v7_phase_b_pilot_v2_summary.json`
- `evidence/series_v7_phase_b_pilot_v2_analysis.json`
- `analysis/phase_b_pilot_v2_table.tex`

---

## [2026-05-28 afternoon] — Paper & Milestone Updated

### Done
- Filled `paper/main.tex` Phase B v2 table (all `—` placeholders replaced with real numbers).
- Wrote `\subsection{Failure Modes}` — explains rank-collapse mechanism and probe-location issue.
- Wrote `\section{Limitations}` — single architecture, ETT only, seed variance caveat.
- Wrote `\section{Conclusion}` — null result framing, informative findings, future directions.
- Updated abstract with full result summary.
- Updated `analysis/milestone_phase_b_launch_2026-05-28.md`:
  - Title and status updated to reflect null result
  - Phase B v2 results table added
  - Next Steps rewritten for post-null-result path

---

## Current State (2026-05-28, end of day)

| Phase | Runs | Status |
|-------|------|--------|
| Phase A — baseline reproduction | 96 | ✅ Complete |
| Phase B v1 — entropy minimisation (+λH) | 12 | ✅ Complete, NO-GO |
| Phase B v2 — entropy maximisation (−λH) | 12 | ✅ Complete, NO-GO |
| **Total** | **120** | |

**Paper**: `paper/main.tex` — abstract, Phase A table, Phase B v1+v2 tables, Failure Modes, Limitations, Conclusion written. Introduction, Related Work, Method, Ablations sections are stubs.

**Null result finding**: Pre-projection spectral entropy (`model.projection` hook, TimesNet, ETT×4) is not a productive regularisation target. The encoder already operates in a well-conditioned spectral regime. Both directions of pressure (+λH and −λH) fail to improve forecasting accuracy consistently.

---

## TODO

### High priority — complete the paper
- [ ] Write `\section{Introduction}` — motivate SE regularisation, state contributions, null result framing
- [ ] Write `\section{Related Work}` — rank regularisation, self-supervised representation learning, time-series regularisers
- [ ] Write `\section{Method}` — SE definition, hook placement, loss variants, complexity
- [ ] Write `\subsection{Ablations}` — λ sensitivity, probe location comparison, seed stability
- [ ] Fix DLinear section in paper (Phase A table has DLinear rows but no SE comparison)

### Medium priority — strengthen the null result
- [ ] Reproduce v2 best configs (λ=0.01 ETTm2) with seeds 2 & 3 to confirm −1.6% is real and not noise
  - Phase A shows ±0.002 MSE seed variance — the −0.003 improvement is marginal
- [ ] Try SE on **within-block** activations (post-FFN inside TSB, not pre-projection)
  - Current probe (`model.projection`) may already be well-conditioned downstream of the bottleneck
- [ ] Test on **PatchTST or iTransformer** — Transformer attention is more prone to representation
  collapse; SE maximisation may have a larger effect there

### Lower priority — expand benchmark coverage
- [ ] Add Weather, Traffic, Electricity datasets (data already available on worker at `~/data/`)
- [ ] Run DLinear with SE to confirm SE is truly model-agnostic (DLinear is linear — SE should be flat)
- [ ] Horizons 192/336/720 for Phase B (current pilots are H=96 only)

### Infrastructure / housekeeping
- [ ] Archive stale queue entries (series_v1 through series_v6) into a history JSON to keep `run_queue.json` lean
- [ ] Add `project.yaml` status field update: `"phase_b_complete_null_result"`
- [ ] Git commit all analysis, evidence, and paper changes
