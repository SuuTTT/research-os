# Time Series Forecasting: A Brutally Honest Newcomer's Guide

*Written for people who can code but have never touched time series. Assumes you've seen a neural network before. Pulls no punches.*

---

## Table of Contents

1. [What is a time series?](#1-what-is-a-time-series)
2. [The forecasting task](#2-the-forecasting-task)
3. [Datasets: the ETT family and why everyone uses them](#3-datasets-the-ett-family-and-why-everyone-uses-them)
4. [Data splits: the single most important thing to get right](#4-data-splits-the-single-most-important-thing-to-get-right)
5. [Normalization](#5-normalization)
6. [Metrics: MSE and MAE](#6-metrics-mse-and-mae)
7. [Models: from simple to complex](#7-models-from-simple-to-complex)
8. [The TSLib protocol (what this project uses)](#8-the-tslib-protocol-what-this-project-uses)
9. [Silly questions with real answers](#9-silly-questions-with-real-answers)
10. [Lessons learned the hard way](#10-lessons-learned-the-hard-way)
11. [Critics and known pathologies in the field](#11-critics-and-known-pathologies-in-the-field)

---

## 1. What is a time series?

A **time series** is a sequence of values indexed by time:

```
t=0   t=1   t=2   t=3  ...  t=T
 x₀    x₁    x₂    x₃        xₜ
```

"Values" can be anything: electricity load, stock price, temperature, CPU usage.

**Univariate**: one variable over time. `[0.3, 0.5, 0.4, 0.7, ...]`

**Multivariate**: multiple variables observed at the same timestamps.
```
time    OT     HUFL   HULL  ...
0:00    8.2    3.1    2.7
0:30    8.5    3.2    2.8
1:00    8.1    3.0    2.6
```
The ETT datasets have 7 variables (oil temperature + 6 load features).

**Key insight**: unlike tabular data, the ORDER of rows matters fundamentally.
Row 100 is meaningless without knowing it came after row 99.

---

## 2. The forecasting task

**Long-term multivariate forecasting** (what this project does):

- Given the last `L` timesteps (the **look-back window**),
  predict the next `H` timesteps (the **horizon**).
- All variables are predicted simultaneously.

```
|<---- look-back L=96 ---->|<-- horizon H=720 -->|
 x_{t-95} ... x_{t-1}  x_t  x_{t+1} ... x_{t+H}
 [      INPUT / ENCODER    ] [    TARGET / PREDICT  ]
```

Common horizons: 96, 192, 336, 720 (for hourly ETT data, 720 = 30 days).

**Why L=96?** It's one standard choice. Some papers tune L per model per dataset
(see Critique §11.2). We fix L=96 everywhere for fair comparison.

---

## 3. Datasets: the ETT family and why everyone uses them

**ETT** = Electricity Transformer Temperature.
Collected from two Chinese counties, 2016–2018, at 15-min and hourly granularity.

| Dataset | Granularity | Train size | Val size | Test size | Variables |
|---------|-------------|-----------|---------|---------|-----------|
| ETTh1   | Hourly      | 8,545     | 2,881   | 2,881   | 7         |
| ETTh2   | Hourly      | 8,545     | 2,881   | 2,881   | 7         |
| ETTm1   | 15-min      | 34,465    | 11,521  | 11,521  | 7         |
| ETTm2   | 15-min      | 34,465    | 11,521  | 11,521  | 7         |

The split is always 70% / 10% / 20% **in chronological order** (never shuffled).

**Why does everyone use ETT?**
It's small enough to run quickly, publicly available, and every paper benchmarks
on it, so numbers are comparable. That said, it's also over-benchmarked to the
point of near-meaninglessness for state-of-the-art claims — see §11.

---

## 4. Data splits: the single most important thing to get right

### The cardinal rule: NEVER shuffle time series data

In standard ML you randomly split your data:
```python
# WRONG for time series
from sklearn.model_selection import train_test_split
X_train, X_test = train_test_split(X, test_size=0.2)  # ← DO NOT DO THIS
```

In time series you always split temporally:
```
|<-------- train --------->|<-- val -->|<-- test -->|
  t=0                  t=T1         t=T2          t=T3
```

**Why?** Because the future must not influence the past during training.
If you shuffle, your model trains on data from after the test period and learns
"future" patterns — its test score will look great but is completely fake.

### The sliding window

A single long series becomes many training examples via sliding windows:

```
Example 1:  x[0:96]   → x[96:96+H]
Example 2:  x[1:97]   → x[97:97+H]
...
Example n:  x[n:n+96] → x[n+96:n+96+H]
```

This is done within the train split only. The test split stays as one contiguous
block evaluated left-to-right.

---

## 5. Normalization

**What**: subtract mean, divide by std. Applied per-channel (per variable).

**Critical detail**: fit the scaler on the TRAINING SET ONLY, then apply it to
val and test. If you fit on the full dataset, you leak statistics from the future
into the past.

```python
# CORRECT
scaler = StandardScaler().fit(X_train)
X_train_norm = scaler.transform(X_train)
X_val_norm   = scaler.transform(X_val)    # ← use train scaler
X_test_norm  = scaler.transform(X_test)   # ← use train scaler

# WRONG — leaks future stats into training
scaler = StandardScaler().fit(X_all)      # ← don't do this
```

TSLib normalizes per-segment (each window), not globally. This is called
**instance normalization** and is the current standard approach.

---

## 6. Metrics: MSE and MAE

**MSE** (Mean Squared Error):
$$\text{MSE} = \frac{1}{N \cdot H \cdot C} \sum_{i,h,c} (\hat{y}_{i,h,c} - y_{i,h,c})^2$$

**MAE** (Mean Absolute Error):
$$\text{MAE} = \frac{1}{N \cdot H \cdot C} \sum_{i,h,c} |\hat{y}_{i,h,c} - y_{i,h,c}|$$

Where $N$ = number of test windows, $H$ = horizon, $C$ = number of channels.

**Key difference**:
- MSE penalizes large errors quadratically. One big spike doubles the loss.
- MAE treats all errors linearly. More robust to outliers.
- For electricity data with occasional demand spikes, MAE can be more meaningful.
- Most papers report both. **Lower is better for both.**

**Pitfall**: because MSE is squared, raw MSE values are NOT in the same units as
the original data. A MSE of 0.4 on normalized data tells you little about real
forecast accuracy without knowing the original scale.

---

## 7. Models: from simple to complex

### DLinear (2023)

```
Input (L × C) → per-channel linear layer → Output (H × C)
```

Literally one linear layer per channel. No attention, no convolution, no
recurrence. Trains in seconds. Often beats complex Transformer variants.

**Why it matters**: if a linear model beats a 12-layer Transformer, something is
wrong with the Transformer — not the linear model.

Paper: *"Are Transformers Effective for Time Series Forecasting?"* (Zeng et al., 2023).
Answer: sometimes no.

### TimesNet (2023)

Converts the 1D time series into a 2D structure by reshaping based on detected
periodicity, then applies 2D convolutions.

Key hyperparameters for ETT (from the paper):
- `d_model=16`, `d_ff=32`, `top_k=5`, `e_layers=2`, `d_layers=1`, `factor=3`

**Common mistake**: using `d_model=32` or forgetting `factor=3`. Both produce
noticeably worse results (we did this in Phase A v1 — see lesson §10.2).

### PatchTST (2023)

Splits the time series into patches (like ViT for images), then applies
a Transformer on patches instead of individual timesteps. Reduces sequence length,
improves long-range modeling.

### iTransformer (2024)

Inverts the attention: instead of attending across time, attends across variates.
Counterintuitive but works on multi-variate forecasting. Controversial — see §11.4.

### LSTM / Temporal Convolutional Networks

Classic approaches. Still competitive on short horizons but generally outperformed
on long horizons (H ≥ 336) by linear and patch-based models.

---

## 8. The TSLib protocol (what this project uses)

[Time-Series-Library](https://github.com/thuml/Time-Series-Library) is a unified
benchmark framework maintained by Tsinghua. It implements 30+ models with a single
training script.

**Our fixed protocol** (Phase A baseline):

```
seq_len   = 96     # look-back window L
label_len = 48     # decoder input overlap (Transformer encoder-decoder only)
pred_len  = H      # 96 / 192 / 336 / 720
features  = M      # multivariate (all 7 channels)
train_epochs = 10
patience     = 3   # early stopping
batch_size   = 32
itr          = 1   # one run per config
```

### What is label_len?

This is a common source of confusion. In Transformer-based models (Informer,
Autoformer, etc.), the **decoder** gets a "start token": the last `label_len`
steps of the encoder input are given to the decoder as context before it begins
predicting. It is NOT target leakage — these are all past values.

```
Encoder input:  [x_{t-95}, ..., x_{t-47}, x_{t-46}, ..., x_t]
                 |<-- first 48 -->|  |<-- last 48 = label_len -->|
Decoder input:  [x_{t-47}, ..., x_t, 0, 0, ..., 0]
                 |<-- label_len -->| |<--  pred_len zeros  -->|
Decoder target: [x_{t-47}, ..., x_t, x_{t+1}, ..., x_{t+H}]
```

For purely encoder models (DLinear, TimesNet, PatchTST), `label_len` is ignored.

---

## 9. Silly questions with real answers

**Q: Why don't we just predict the mean? That's the simplest baseline.**

A: You should always report a "naive" baseline (e.g. last observed value,
seasonal naive, mean). Many Transformer papers fail to beat naive baselines on
some datasets. We don't report naive here but it's a legitimate sanity check.

**Q: If DLinear beats Transformers, why bother with complex models?**

A: DLinear is competitive on ETT because ETT has strong linear trends and
seasonal patterns. On more chaotic series (financial data, irregular events),
linear models fail. Also, DLinear can't do multivariate cross-channel attention,
which matters when variable interactions are complex.

**Q: What's the difference between `features=M` and `features=S`?**

A: `M` = multivariate: all 7 channels as input, all 7 predicted.
`S` = univariate: each channel trained and evaluated separately.
`MS` = multivariate input, single channel output.
Most benchmark papers use `M`. Some older papers use `S`, which is an easier task.
Comparing `M` and `S` numbers directly is wrong.

**Q: Why does ETTm have lower MSE than ETTh even though it's longer?**

A: ETTm is 15-minute granularity with very strong diurnal periodicity. The
short-term patterns are highly regular, so prediction is easier. ETTh is hourly
and exhibits more irregular variation.

**Q: My model's training loss goes down but test MSE is flat. What's wrong?**

A: Classic overfitting, or a normalization leak (scaler fitted on full data),
or your sliding window has overlap between train and test splits (the window
at the boundary might include test timesteps in the input).

**Q: How many seeds should I run?**

A: At minimum 3 for stochastic models. DLinear has no random initialization so
one seed is enough. TimesNet with `d_model=16` is very stable — most configs
show std < 0.001 across seeds, except some long-horizon ETTm cases (std ≈ 0.02).

**Q: My GPU runs out of memory on ETTm2 h720. What do I do?**

A: Reduce `batch_size` (try 16 or 8). The model itself is small — TimesNet with
`d_model=16` is only ~5MB. The memory issue is usually the data batch (sequence
length × batch × features).

**Q: Why does `label_len` matter if it's just past data?**

A: It doesn't affect forecasting correctness, but it affects decoder attention:
giving the decoder context helps autoregressive decoding for Transformer-based
models. For non-autoregressive models (like DLinear, TimesNet), it's irrelevant.

---

## 10. Lessons learned the hard way

### 10.1 Never leave debug shortcuts in production configs

In Phase A v1, we had `--train_epochs 1 --patience 1` left in from debugging.
The model trained for one epoch and early-stopped immediately. Results looked
valid (no errors, metrics reported) but were completely useless.

DLinear ETTm2 h720: **0.680 (ours) vs 0.397 (paper)** — a 71% gap.
Lesson: always print `train_epochs` in your run log header.

### 10.2 Architecture hyperparameters matter more than you think

TimesNet `d_model=32` vs `d_model=16` (paper):
- ETTm2 h720: 0.4417 vs 0.4225 — 4.5% difference from one number.
- `factor` (number of periods to detect) default is 1; paper uses 3.

Always cross-check every hyperparameter against the original paper's Table or
appendix. The model code default != paper best config.

### 10.3 pip "downgrade" of numpy doesn't clean up compiled extensions

When you downgrade numpy (e.g., 2.x → 1.26), pip replaces the Python files but
leaves stale `.so` compiled extensions in `numpy/_core/`. Scipy and torch then
import the new Python but the old binary, causing cryptic ABI crashes at runtime.

Fix: `find /path/to/numpy/_core/ -name '*.so' -delete` before reinstall.

### 10.4 TSLib saves huge prediction arrays to disk

After each run, TSLib saves `pred.npy` and `true.npy` to `./results/`. For
ETTm2 with H=720 this is ~230MB per file × 2 = ~460MB per run. Over 64 runs
this fills a 32GB disk. The metric is already printed to stdout — delete
`results/` after each run.

### 10.5 Disk full mid-run causes silent partial results

If `/tmp` runs out of space, PyTorch's multiprocessing can fail with
`OSError: [Errno 28] No space left on device: '/tmp/pymp-XXXXX'`.
The run fails mid-epoch. The log may have some epoch lines but no final metric.
The collector sees no metric → marks the run as succeeded with `mse=nan`.
Always monitor disk and clear pip cache before long batch runs.

### 10.6 Vast.ai workers: `BatchMode=yes` breaks dispatch

When `BatchMode=yes` is set in the SSH client config, any interactive prompt
(host key confirmation, password) causes an immediate exit. On ephemeral workers
with rotating host keys, this silently drops the connection.
Use `StrictHostKeyChecking=no` and no `BatchMode`.

---

## 11. Critics and known pathologies in the field

### 11.1 Leaking future information into training (label leakage)

**The problem**: some implementations use the ground-truth future values as part
of the decoder input during training, but then at inference time must use their
own predictions. This creates a **train/inference discrepancy**.

In TSLib's Transformer decoder:
```
Training:    decoder input = [label_len past values] + [H ground-truth future]
Inference:   decoder input = [label_len past values] + [H zeros]
```
The model learns to rely on the ground-truth future during training. At test time
it gets zeros — a completely different distribution. For purely non-autoregressive
models (DLinear, TimesNet, PatchTST) this doesn't apply since they don't use a
decoder in this way.

**More subtle leak**: if you normalize using statistics computed over the full
dataset (train + val + test combined), the model indirectly "knows" future
statistics during training. Fit your scaler on training data only.

**Even more subtle**: if your validation set is used for early stopping AND for
final model selection (not unusual), you've implicitly optimized on val — report
val as a tuning artifact, not a real evaluation number. Test set should be
untouched until the final evaluation.

### 11.2 Look-back window tuning as hidden hyperparameter

Several benchmark papers (including the original DLinear paper) tune `seq_len`
per model per dataset. DLinear on ETTm1 with `seq_len=336` beats `seq_len=96`
by a large margin. If Model A uses its best `seq_len` and Model B uses a fixed
`seq_len=96`, the comparison is unfair.

The honest approach: either fix `seq_len` for all models (our approach, L=96),
or tune it identically for all models using a held-out validation set.

### 11.3 Overfitting on small test sets

ETTh test = 2,881 timesteps. With H=96 and step=1, you get ~2,785 overlapping
test windows. These windows share 95 timesteps with their neighbors — they are
NOT independent. Standard error estimates assuming i.i.d. samples are invalid.
A model that memorizes the test distribution could show much better MSE without
generalizing.

Some researchers address this by reporting confidence intervals via block
bootstrapping. Most do not.

### 11.4 iTransformer: attending across variates, not time

The iTransformer (Liu et al., 2024) feeds each timestep's vector of all variates
to attention, then attends across variates instead of time. On some benchmarks it
achieves SOTA.

The critique: this effectively ignores temporal ordering within the attention
mechanism. The "temporal" modeling is entirely offloaded to the FFN. Whether
this is a clever insight or a coincidental match to ETT's variate-correlated
structure is debated. Results on real-world non-ETT benchmarks are less consistent.

### 11.5 The "ETT is too easy" problem

ETT has become the MNIST of time series forecasting: every paper benchmarks on
it, scores are highly tuned, and it no longer discriminates between models.
A linear model achieves near-paper-quality results because ETT has strong
linear structure.

If you want to claim a genuine improvement, you need: (a) diverse datasets
(weather, traffic, exchange rates, healthcare), (b) statistical significance
tests, and (c) evaluation on unseen data distributions.

### 11.6 Reporting MSE of normalized predictions

TSLib evaluates on normalized (zero-mean, unit-variance) data. An MSE of 0.4
sounds good but you cannot compare it to MSE in the original units without
knowing the variance of the target variable. Always report whether metrics are
on normalized or denormalized values, and be consistent.

### 11.7 "We beat Transformers with a linear model" is a 2023 claim

The DLinear paper was a watershed moment in 2023, showing linear models could
beat Transformers. This has since been addressed: PatchTST (patches + Transformer)
and iTransformer both beat DLinear on most benchmarks. The narrative has moved on.
Citing "linear models beat Transformers" as a current finding without qualification
is now outdated.

### 11.8 Reproducibility via public checkpoints and seeds

Many papers do not release: (a) exact random seeds used, (b) trained checkpoints,
(c) full hyperparameter search logs. "Reproduction" then requires guessing configs
until numbers match. Our Phase A v1 took hours to reproduce TimesNet correctly
because `d_model` and `factor` were not clearly stated in the paper text
(they were in the appendix code release).

**Best practice**: always pin: git SHA, Python version, all library versions,
all random seeds, all hyperparameters. See our `project.yaml` for the template.

---

## Quick Reference Checklist

When reading a time series forecasting paper, ask:

- [ ] What is the look-back window `seq_len`? Fixed or tuned per model?
- [ ] Is the evaluation on normalized or denormalized values?
- [ ] Is `features=M` (multivariate) or `features=S` (univariate per channel)?
- [ ] Are train/val/test splits strictly temporal (no shuffle)?
- [ ] Is the scaler fitted only on training data?
- [ ] Are results averaged across multiple seeds?
- [ ] Are error bars / standard deviations reported?
- [ ] Is the test set used only for final evaluation, not for model selection?
- [ ] Do Transformer models use ground-truth decoder input during training?
- [ ] Is there a naive baseline (last-value, seasonal naive) to compare against?

If the answer to any of these is "unclear" or "no", treat the numbers with caution.

---

*Last updated: 2026-05-28. Part of the structural entropy timeseries project.*
