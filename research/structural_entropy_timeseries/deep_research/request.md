# Deep Research Prompt
# Version: 2  (updated 2026-05-28 after Phase A baseline completion)

## Research Hypothesis

Structural entropy (SE) — a measure of the complexity of learned internal
representations — can serve as a model-agnostic regulariser for long-term
time-series forecasting. Adding an SE penalty to the training loss forces the
model to compress its representations, which we hypothesise reduces overfitting
and improves test generalisation across architectures and datasets.

We have completed Phase A (clean reproduction of TimesNet and DLinear on ETT,
3 seeds). Phase B is SE regularisation. Before designing Phase B, we need a
current-state-of-the-field picture.

---

## Primary Research Questions

1. **SOTA landscape (2023–2026)**: Which supervised long-term forecasting models
   lead on standard benchmarks as of early 2026? What are their key innovations
   relative to TimesNet (2023)?

2. **Official benchmark protocol as of 2024-25 top venues**: What exact
   experimental protocol do NeurIPS 2024, ICLR 2025, ICML 2024 papers use?
   Specifically:
   - Which datasets (names, sizes, download URLs)?
   - Fixed vs. tuned look-back window L?
   - Multivariate (M) vs. univariate (S) evaluation?
   - Normalisation method (global vs. instance norm)?
   - Number of seeds and how variance is reported?
   - Is there a canonical framework (TSLib? GluonTS? Darts? other)?
   - What metrics beyond MSE and MAE are now standard?

3. **Protocol fairness critiques**: What are the currently accepted critiques of
   ETT-focused benchmarks? Specifically:
   - Saturation: are ETT differences within noise floors?
   - Look-back window tuning as hidden hyperparameter — how do top papers handle it?
   - Decoder teacher-forcing leakage (ground-truth future as decoder input at
     train time but zeros at test time) — which model classes are affected and
     how do top papers disclose or mitigate it?
   - Test-set overlap from sliding windows — does any top paper address this?
   - Normalised vs. denormalised metric reporting inconsistencies?
   - Any known cases of cherry-picked seeds or dataset-specific tuning presented
     as general results?

4. **Structural entropy / representation complexity in forecasting**: Has SE or
   any equivalent measure (representation entropy, intrinsic dimensionality,
   spectral complexity, effective rank) been used as a training objective or
   analysis tool in time-series forecasting? Cite papers if yes.
   - If no direct prior work: what is the closest analogue (e.g., information
     bottleneck approaches in sequence models, compression-based regularisers
     in vision)?

5. **SE regularisation design questions**:
   - Which layer(s) of TimesNet / PatchTST / iTransformer are natural candidates
     for measuring SE? (e.g., the 2D-conv feature maps in TimesNet, the patch
     embeddings in PatchTST, the variate-attention outputs in iTransformer)
   - What is the computational cost of computing SE over a batch relative to
     the forward pass?
   - Are there existing differentiable entropy regularisers that could serve as
     proxies if SE computation is expensive? (e.g., nuclear norm, Frobenius norm
     of correlation matrix, contrastive losses)

6. **Target venue and publishability assessment**:
   - What would a Phase B result need to show to be publishable at ICLR 2026
     or NeurIPS 2026?
   - Minimum number of datasets, baselines, ablation conditions?
   - Is a workshop paper (e.g., NeurIPS 2025 Time Series workshop) a more
     realistic first target?

---

## Return Format

Return a structured report with these sections:

### S1. SOTA Table (2023–2026)
Table with columns:
  Paper title | Year | Venue | Model name | Method category |
  ETTh1-96 MSE | ETTm1-96 MSE | Weather-96 MSE | Traffic-96 MSE |
  Datasets used | L (look-back) | Code URL | License

Include at minimum: DLinear, TimesNet, PatchTST, iTransformer, TimeMixer,
FITS, Crossformer, TiDE, TimesFM (zero-shot), Moirai (zero-shot), Chronos.

### S2. Official Benchmark Protocol (2024-25 standard)
Exact protocol used by the dominant framework(s). Quote directly from paper
appendices or code where possible. Flag any inconsistencies between papers.

### S3. Critique Catalogue
For each known critique, state:
  - Name of critique
  - Severity (blocks publishability / weakens claims / cosmetic)
  - Which models/papers it affects
  - How it is typically addressed in top papers
  - Whether our current setup (fixed L=96, ETT×4, 3 seeds) is affected

### S4. SE Prior Work
Literature review of entropy/complexity regularisation in time-series or
sequence models. If no direct work, provide the closest analogues with full
citations.

### S5. Phase B Design Recommendation
Concrete recommendation for Phase B experimental design:
  - Which 2-3 backbone models to apply SE to (justify)
  - Which 6-8 datasets (justify — must include at least 2 non-ETT)
  - Which λ values for the SE penalty to sweep
  - What look-back window L to use
  - What ablation conditions are minimally necessary for a fair claim
  - What a "positive result" looks like (e.g., SE reduces test MSE by X% on
    at least Y out of Z configurations)
  - What a "negative result" tells us (is a negative result publishable?)

### S6. Publishability Roadmap
Timeline and checklist for a workshop or full paper submission.

### S7. BibTeX
All cited papers in BibTeX format.

---

## Rules

- Separate confirmed facts (with citations) from inference (clearly labelled).
- Prefer paper appendices and official repos over blog posts.
- Flag incompatible benchmark settings explicitly (e.g., "Paper X reports L=336
  but Paper Y reports L=96 — not directly comparable").
- If a dataset or method is paywalled or unavailable, say so.
- Be sceptical: if a claim sounds too good (e.g., +10% improvement with one
  trick), look for whether the baseline was weakened or the dataset was
  cherry-picked.
- Include direct URLs to arXiv, GitHub, and dataset sources.
- Do not summarise from secondary sources if the primary paper is accessible.

