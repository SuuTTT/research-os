# Research Ideas — Structural Entropy & Time-Series Regularisation

**Status legend**: 🟢 High priority · 🟡 Medium · 🔵 Exploratory · ⚫ Long-shot  
**Effort legend**: S = <1 day · M = 1–3 days · L = 1–2 weeks

---

## Part 1 — Fixes to the Current Approach

These directly address the diagnosed failure modes.
Run these before claiming the final null result is definitive.

### 1.1 🟢 Within-block probe location [Effort: M]
**What**: Move the SE hook from `model.projection` to activations
*inside* TimesBlock — specifically after the period-analysis FFN or
after the 2D-convolution stack.

**Why promising**: The pre-projection layer may already be well-conditioned
(it's downstream of all temporal processing). Internal FFN activations are
more likely to exhibit rank-collapse because each block independently
processes one periodicity scale.

**How**:
```python
# Replace hook on model.projection with:
for i, block in enumerate(model.model.layer):
    block.conv.register_forward_hook(...)  # after 2D conv
```
**Expected signal**: If any block's activations show low effective rank,
entropy-max at that location could help.

---

### 1.2 🟢 Multi-seed confirmation of ETTm2 −1.6% result [Effort: S]
**What**: Re-run v2 λ=0.01 on ETTm2 with seeds 2 and 3.

**Why**: Phase A std is ±0.002 MSE; the observed improvement is −0.003.
If seeds 2/3 also show improvement (even smaller), the pattern is real.

**How**: Add 2 runs to queue, same config, seeds 2 & 3.

**GO criterion**: Mean Δ MSE < −0.002 across 3 seeds → worth investigating
at other horizons.

---

### 1.3 🟢 Fine-grained λ sweep around λ=0.001–0.01 [Effort: M]
**What**: Try λ ∈ {0.0001, 0.0003, 0.001, 0.003, 0.005} for entropy-max.

**Why**: λ=0.01 is the sweet spot where SE doesn't overwhelm MSE.
There may be an even smaller λ where ETTh1 is no longer hurt but
ETTm gains persist.

**Expected signal**: A λ ≈ 0.002 could show consistent −0.5% across 3+
datasets. If so, → GO for full 144-run sweep.

---

### 1.4 🟡 Entropy maximisation on PatchTST / iTransformer [Effort: L]
**What**: Apply `-λH` to PatchTST or iTransformer with the same 12-run design.

**Why strongly promising**: These architectures use self-attention, which is
known to suffer from representation collapse (uniform attention maps, low
effective rank in key/query projections). SE maximisation is much more likely
to have a measurable effect here.

**Probe locations**:
- PatchTST: activations after the final encoder block (before the flatten + linear head)
- iTransformer: activations after inverted self-attention (before projection)

**Predicted outcome**: 3–8% MSE improvement at λ=0.01–0.05 on ETTh1/ETTh2.

---

### 1.5 🟡 Adaptive λ schedule [Effort: M]
**What**: Start with λ=0 for the first 3 epochs (warm-up), then linearly
ramp up to λ_max over epochs 3–7.

**Why**: Early training has large MSE gradients that dwarf the SE signal;
the SE penalty disrupts gradient flow when it's least useful. Ramping λ
after convergence begins avoids this.

**How**:
```python
λ_eff = λ * min(1.0, max(0.0, (epoch - warmup) / ramp_epochs))
```

---

## Part 2 — New Probe Ideas (Structural Entropy Variants)

Different ways to define / compute the entropy signal.

### 2.1 🟢 Per-head attention entropy in Transformers [Effort: M]
**What**: In PatchTST / iTransformer, compute H of the attention weight
distribution per head (not covariance entropy). Penalise heads that
concentrate attention on a single token.

**Why**: Attention collapse (uniform or single-token attention) is a
known pathology. Entropy of attention distributions is a natural, cheap
signal.

**Formula**: For attention weights A ∈ ℝ^(B×H×N×N), compute H per head
and average: `H_attn = mean_h(−Σ_j A_{ij} log A_{ij})`.

---

### 2.2 🟡 Temporal singular value spectrum (SVD of hidden states) [Effort: M]
**What**: Instead of covariance over feature dimensions, compute SVD of the
time-axis: reshape to (B×D, T) and take SVD. Entropy of singular values
measures how many temporal "modes" the encoder uses.

**Why**: TimesNet explicitly decomposes periods; the temporal SVD could
reveal whether certain periodicity modes dominate and others are unused.

---

### 2.3 🟡 Channel-wise rank (per-channel covariance) [Effort: S]
**What**: Treat each of the D feature channels independently and compute
the rank of their Gram matrix over the batch. Sum or average over channels.

**Why**: Fine-grained signal; if only a subset of channels collapse, a
per-channel penalty avoids hurting already-diverse channels.

---

### 2.4 🔵 Effective rank as a training diagnostic (no regularisation) [Effort: S]
**What**: Track H(X) during baseline training (no λ term) across all layers.
Log rank trajectory as the model converges.

**Why**: Before regularising, understand what's happening. If rank is
naturally increasing during training → maxing it further won't help. If
rank is collapsing in early epochs → targeted warmup SE could prevent it.

**Deliverable**: Figure showing effective rank vs. epoch for each TimesBlock
and the projection layer.

---

## Part 3 — Beyond Spectral Entropy (New Regularisation Ideas)

### 3.1 🟢 Decorrelation regularisation (DeCov-style) [Effort: M]
**What**: Penalise the off-diagonal squared entries of the feature
covariance: `L_cov = ||C − diag(C)||_F^2`.

**Why**: This is the direct supervised analogue of VICReg's covariance
term. Unlike SE, it does not require eigendecomposition (cheaper),
and it acts on individual channel pairs rather than the global spectrum.

**Prior work**: DeCov (Cogswell et al., 2016) showed this reduces
overfitting in supervised CNNs.

---

### 3.2 🟢 Mixup in representation space [Effort: M]
**What**: After the encoder, interpolate between two representations in
the batch before the projection head. Train on interpolated predictions.

**Why**: Manifold mixup in latent space regularises the representation
geometry without any covariance computation. Simple, cheap, known to work
in other forecasting tasks.

**Expected**: 0.5–2% consistent improvement across datasets at α=0.2.

---

### 3.3 🟡 Temporal contrastive auxiliary loss [Effort: L]
**What**: Add a contrastive term that pulls together representations of
nearby time windows and pushes apart distant ones (similar to TNC or TS2Vec).

**Why**: The MSE loss provides no signal about temporal consistency of
representations — only output accuracy. A contrastive auxiliary loss
directly optimises representation quality as a side objective.

---

### 3.4 🟡 Frequency-domain regularisation [Effort: M]
**What**: Compute the FFT of the forecast output and add a penalty on
high-frequency components (energy in top-K frequencies beyond a cutoff).

**Why**: TimesNet operates in the period domain; penalising high-frequency
output oscillations directly regularises over-fitting to noise.

**Note**: This regularises the output, not the representation — different
from SE but potentially more directly effective.

---

### 3.5 🔵 Wasserstein distance between train/val representation distributions [Effort: L]
**What**: Compute the Sinkhorn approximation of W2 between the distribution
of encoder activations on train vs. validation batches. Add as a penalty.

**Why**: Minimising the train/val gap in representation space directly
targets the generalisation mechanism. If the gap is large, the model is
memorising training patterns at the representation level.

---

## Part 4 — Architecture-Level Ideas

### 4.1 🟡 Rank-constrained projection layer [Effort: M]
**What**: Replace the dense projection `model.projection` with a
low-rank factorised layer: `W = U V^T` where U ∈ ℝ^(D×r), V ∈ ℝ^(out×r),
r < D. Sweep r ∈ {4, 8, 16, 32}.

**Why**: This directly enforces a rank constraint at the projection
without any auxiliary loss. If low rank is beneficial, it should show up
as improved test MSE for small r.

**Predicted**: Slight improvement at r=8–16 for ETTm; degradation at r=4.

---

### 4.2 🟡 SE regularisation on iTransformer's channel attention [Effort: M]
**What**: iTransformer inverts the transformer: attention is over channels
(D tokens), time is processed by FFN. Apply SE-max to the channel attention
output (the "value" projection activations).

**Why**: In iTransformer, channels attending to a single dominant channel
(low-rank attention) is a known failure mode. SE of the attended
representation is a natural proxy.

---

### 4.3 🔵 Neural ODE encoder with SE regularisation [Effort: L]
**What**: Replace TimesNet's period-decomposition stack with a Neural ODE
that integrates the hidden state over time. Apply SE-max to the ODE state
trajectory.

**Why**: Neural ODEs produce smooth, continuous hidden state trajectories;
SE of the trajectory covariance measures how many dynamical modes are used.

---

## Part 5 — Your Own Ideas (Placeholder)

> **Add your ideas below. Notes on format:**  
> - Start with a brief `What` / `Why` / `How` even if rough  
> - Tag priority (🟢🟡🔵⚫) and effort (S/M/L)  
> - A promising-but-half-formed idea is fine — capture it now

### 5.1 [Your idea title] [Effort: ?]
**What**:  
**Why**:  
**How**:  

---

### 5.2 [Your idea title] [Effort: ?]
**What**:  
**Why**:  
**How**:  

---

### 5.3 [Your idea title] [Effort: ?]
**What**:  
**Why**:  
**How**:  

---

## Prioritised execution order

| # | Idea | Effort | Expected signal | Dependency |
|---|------|--------|----------------|------------|
| 1 | 1.2 Multi-seed ETTm2 confirmation | S | Confirms -1.6% is real | None |
| 2 | 1.3 Fine λ sweep (0.001–0.005) | M | Find a safe λ that helps 3+ datasets | 1 |
| 3 | 2.4 Rank diagnostic (no λ) | S | Understand what's actually collapsing | None |
| 4 | 1.1 Within-block probe | M | Expose a real bottleneck | 3 |
| 5 | 3.1 DeCov-style decorrelation | M | Cheaper, more targeted than SE | 3 |
| 6 | 1.4 SE on PatchTST/iTransformer | L | Highest impact if collapse confirmed | 3–4 |
| 7 | 3.2 Mixup in representation space | M | Easy win, architecture-agnostic | None |
| 8 | 4.1 Rank-constrained projection | M | Direct rank test | 3 |
