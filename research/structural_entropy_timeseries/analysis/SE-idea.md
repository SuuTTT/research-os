# NBEATSx-DC: Reproducible English Method Specification for Algorithm Engineers

> This document is a cleaned, implementation-oriented English rewrite of the provided Chinese draft. It is not a literal translation. The goal is to make the method reproducible, auditable, and easy to convert into a conference-paper method section.

## 1. Problem Statement

We study multivariate long-horizon time-series forecasting for ship-motion data under complex sea conditions. Compared with common benchmark datasets such as electric transformer temperature (ETT), ship-motion signals usually have weaker visible periodicity, multiple coupled periodic scales, and stronger nonlinear interactions between vessel-state variables and environmental variables. These properties make direct use of all exogenous variables in a standard forecasting model both noisy and computationally inefficient.

Given a multivariate time series

\[
\mathbf{X}_{1:T} = [\mathbf{x}_1, \dots, \mathbf{x}_T] \in \mathbb{R}^{T \times (n+1)},
\]

where one variable is the forecasting target \(y_t\) and the remaining \(n\) variables are exogenous variables \(\mathbf{z}_t \in \mathbb{R}^{n}\), the goal is to predict the future target sequence

\[
\hat{\mathbf{y}}_{T+1:T+H}
\]

from a lookback window of length \(L\), while selecting only exogenous variables that are genuinely useful for the target forecast.

The proposed method, **NBEATSx-DC**, improves NBEATSx by adding two preprocessing modules:

1. **Dual-stage spectral decomposition**, which separates intrinsic target frequencies from externally driven frequencies.
2. **Community-driven feature selection**, which builds a graph over variable-frequency components and uses structural-entropy-based community detection to select relevant exogenous variables.

The final selected features and the residualized target are passed into an NBEATSx forecasting backbone.

---

## 2. Core Motivation

A standard NBEATSx model directly consumes all exogenous variables. This is simple but problematic for ship-motion prediction:

- Many exogenous variables are weakly related or unrelated to the target.
- Some variables share superficial periodicity with the target but do not truly drive it.
- The target contains intrinsic oscillatory components caused by the vessel's own dynamics.
- External variables such as waves, wind, water speed, current direction, and heading may drive additional periodic responses.

Therefore, before forecasting, the method first removes the target's dominant intrinsic frequencies, then analyzes the remaining target dynamics and exogenous variables in the frequency domain. This reduces false correlations and improves both interpretability and efficiency.

---

## 3. Notation

| Symbol | Meaning |
|---|---|
| \(T\) | Total time-series length |
| \(L\) | Lookback/input length |
| \(H\) | Forecast horizon |
| \(n\) | Number of exogenous variables |
| \(y_t\) | Target variable at time \(t\) |
| \(z_t^{(i)}\) | Exogenous variable \(i\) at time \(t\) |
| \(K_1\) | Number of intrinsic target frequencies extracted in stage 1 |
| \(K_2\) | Number of driving frequencies extracted per variable in stage 2 |
| \(\theta\) | Graph sparsification threshold ratio |
| \(\beta\) | Variable-level secondary selection ratio |
| \(G=(V,E,W)\) | Weighted variable-frequency association graph |
| \(v_{i,k}\) | Node corresponding to variable \(i\)'s \(k\)-th selected frequency component |
| \(E_{i,k}\) | Normalized spectral energy of node \(v_{i,k}\) |
| \(\mathrm{TMI}(\cdot,\cdot)\) | Temporal mutual information |
| \(\mathrm{DTW}(\cdot,\cdot)\) | Dynamic time warping alignment |

Variable index \(i=0\) denotes the target variable. Indices \(i=1,\dots,n\) denote exogenous variables.

---

## 4. Overall Pipeline

For each data segment or training split, NBEATSx-DC runs the following pipeline:

1. **Segment the time series** into local windows or scenario-specific chunks if the dominant frequencies vary strongly across time.
2. **Stage-1 spectral decomposition** on the target variable only.
3. **Remove the target's top-\(K_1\) intrinsic frequency components** to obtain a residual target signal.
4. **Stage-2 spectral decomposition** on the residual target and all exogenous variables.
5. **Create variable-frequency nodes**, where each selected frequency component of each variable becomes one graph node.
6. **Align frequency-component series** using DTW when their periods or component lengths differ.
7. **Compute edge weights** using temporal mutual information multiplied by spectral energy terms.
8. **Sparsify the graph** by thresholding weak edges.
9. **Run structural-entropy-based community detection** on the graph.
10. **Select exogenous variables** whose frequency components fall into the same communities as target frequency components.
11. **Train NBEATSx** using the residualized target and selected exogenous variables.
12. **Evaluate** using standard forecasting metrics such as MSE and MAE.

---

## 5. Stage 1: Intrinsic Target-Frequency Removal

### 5.1 Rationale

The target signal may contain intrinsic oscillations determined by the physical properties of the vessel, such as hull structure, inertia, and natural motion behavior. These intrinsic components can dominate the frequency spectrum but may not reflect the effect of external variables.

If we build a cross-variable association graph before removing these intrinsic frequencies, the graph may contain false associations. For example, two variables may both show a daily or periodic pattern, but this does not necessarily mean one is useful for forecasting the other.

### 5.2 Implementation

Given a target segment \(\mathbf{y} \in \mathbb{R}^{L_s}\), compute the real-valued FFT:

\[
\mathbf{Y} = \mathrm{rFFT}(\mathbf{y}).
\]

Compute the amplitude spectrum:

\[
A_k = |Y_k|.
\]

Recommended reproducibility choice:

- Exclude the DC component \(k=0\) from frequency selection, unless the implementation explicitly treats the mean as a removable component.
- Select the top \(K_1\) nonzero-frequency indices by amplitude.

Let \(\mathcal{F}_{\mathrm{in}}\) be the selected intrinsic-frequency index set. Reconstruct the intrinsic component by keeping only these frequency bins:

\[
\tilde{Y}_k =
\begin{cases}
Y_k, & k \in \mathcal{F}_{\mathrm{in}}, \\
0, & \text{otherwise}.
\end{cases}
\]

Then:

\[
\mathbf{y}_{\mathrm{in}} = \mathrm{irFFT}(\tilde{\mathbf{Y}}),
\]

and the residual target is:

\[
\mathbf{y}_{\mathrm{res}} = \mathbf{y} - \mathbf{y}_{\mathrm{in}}.
\]

This residual target is used in stage 2 and later as the target-side input to NBEATSx.

### 5.3 Practical Notes

Use one consistent FFT normalization convention across all experiments. In NumPy/PyTorch, the default FFT normalization is usually fine as long as it is used consistently.

If the series has a strong trend, apply normalization or detrending before FFT, or leave trend modeling to the NBEATSx trend block and exclude the DC component from spectral selection.

---

## 6. Stage 2: Driving-Frequency Extraction

### 6.1 Rationale

After intrinsic target components are removed, the residual target may still contain weak periodic signals driven by external conditions. Stage 2 extracts candidate driving-frequency components from:

- the residual target \(\mathbf{y}_{\mathrm{res}}\), and
- each exogenous variable \(\mathbf{z}^{(i)}\).

Each selected frequency component becomes a variable-frequency node.

### 6.2 Implementation

For every variable \(i \in \{0,1,\dots,n\}\), define:

\[
\mathbf{s}^{(0)} = \mathbf{y}_{\mathrm{res}}, \quad
\mathbf{s}^{(i)} = \mathbf{z}^{(i)}, \ i=1,\dots,n.
\]

Compute:

\[
\mathbf{S}^{(i)} = \mathrm{rFFT}(\mathbf{s}^{(i)}),
\]

and select the top \(K_2\) nonzero frequencies by amplitude. For each selected frequency \(k\), reconstruct the single-frequency component:

\[
\mathbf{c}_{i,k} = \mathrm{irFFT}(\tilde{\mathbf{S}}^{(i,k)}),
\]

where \(\tilde{\mathbf{S}}^{(i,k)}\) keeps only bin \(k\) and zeros out all other bins.

Define the normalized spectral energy of this component as:

\[
E_{i,k} = \frac{|S^{(i)}_k|^2}{\sum_{q \in \mathcal{F}^{(i)}_{K_2}} |S^{(i)}_q|^2 + \epsilon}.
\]

Here \(\epsilon\) is a small constant such as \(10^{-8}\) to avoid division by zero.

The graph node is:

\[
v_{i,k} = (i, k, \mathbf{c}_{i,k}, E_{i,k}, P_k),
\]

where \(P_k\) is the approximate period corresponding to frequency bin \(k\):

\[
P_k = \frac{L_s}{k}.
\]

---

## 7. Variable-Frequency Association Graph

### 7.1 Node Definition

The node set is:

\[
V = \{v_{i,k} \mid i = 0,1,\dots,n; \ k \in \mathcal{F}^{(i)}_{K_2}\}.
\]

The total number of nodes is approximately:

\[
|V| = (n+1)K_2.
\]

### 7.2 Edge Weight Definition

Edges are only computed between frequency components from different variables. Components from the same variable are not connected, because the goal is to model cross-variable coupling rather than intra-variable decomposition.

For two nodes \(v_{i,a}\) and \(v_{j,b}\), if \(i=j\), set:

\[
w_{(i,a),(j,b)} = 0.
\]

Otherwise:

1. Align the two component series using DTW:

\[
(\bar{\mathbf{c}}_{i,a}, \bar{\mathbf{c}}_{j,b}) = \mathrm{DTWAlign}(\mathbf{c}_{i,a}, \mathbf{c}_{j,b}).
\]

2. Compute temporal mutual information:

\[
I_{(i,a),(j,b)} = \mathrm{TMI}(\bar{\mathbf{c}}_{i,a}, \bar{\mathbf{c}}_{j,b}; \tau),
\]

where \(\tau\) is an optional lag parameter. A reproducible default is to evaluate \(\tau \in \{0,1,\dots,\tau_{\max}\}\) and use the maximum value.

3. Weight TMI by spectral energy:

\[
w_{(i,a),(j,b)} = E_{i,a} E_{j,b} I_{(i,a),(j,b)}.
\]

This definition rewards pairs that are both statistically coupled and spectrally important.

### 7.3 Graph Sparsification

Let \(W\) be the dense weighted adjacency matrix. Apply thresholding:

\[
w_{pq} =
\begin{cases}
w_{pq}, & w_{pq} \ge \theta \cdot \max(W), \\
0, & \text{otherwise}.
\end{cases}
\]

Recommended default from the original draft:

\[
\theta = 0.25.
\]

This step removes weak edges and reduces the computational cost of community detection.

---

## 8. Structural-Entropy-Based Community Feature Selection

### 8.1 Goal

The graph is designed so that useful exogenous frequency components are likely to be close to, or in the same community as, target frequency components. Therefore, feature selection is transformed into a graph community detection problem.

### 8.2 Structural Entropy Objective

Given a weighted graph \(G=(V,E,W)\) and an encoding tree \(T\), structural entropy measures the average uncertainty of encoding a random walk on the graph under the hierarchical partition defined by \(T\).

A common form is:

\[
H_T(G) = -\sum_{\alpha \in T, \alpha \ne \lambda}
\frac{g_\alpha}{\mathrm{vol}(G)}
\log_2 \frac{\mathrm{vol}(\alpha)}{\mathrm{vol}(\alpha^-)}.
\]

where:

- \(\lambda\) is the root of the encoding tree,
- \(\alpha^-\) is the parent of node \(\alpha\) in the encoding tree,
- \(g_\alpha\) is the cut weight between partition \(\alpha\) and its complement,
- \(\mathrm{vol}(\alpha)\) is the weighted degree volume of partition \(\alpha\),
- \(\mathrm{vol}(G)\) is the total graph volume.

The graph structural entropy is:

\[
H(G) = \min_T H_T(G).
\]

If the tree height is limited to \(K\), then the \(K\)-dimensional structural entropy is:

\[
H_K(G) = \min_{T: h(T) \le K} H_T(G).
\]

Minimizing structural entropy encourages compact communities with strong internal connections and weak external connections.

### 8.3 Community Extraction

The original draft describes a greedy encoding-tree optimization procedure based on merge and combine operators. For reproducibility, an implementer can use either:

1. an official or existing structural entropy minimization implementation, if available; or
2. a deterministic community detection replacement, such as Louvain or Leiden, clearly reported as an approximation.

For a conference paper, the exact community algorithm must be fixed and reported. If structural entropy is claimed as the main contribution, do not silently replace it with modularity-based Louvain without stating the approximation.

### 8.4 Target-Associated Community Selection

Let \(V_y\) be the set of nodes corresponding to target residual frequency components:

\[
V_y = \{v_{0,k} \mid k \in \mathcal{F}^{(0)}_{K_2}\}.
\]

Let the community partition be:

\[
\mathcal{C}=\{C_1,C_2,\dots,C_m\}.
\]

Define target-associated communities:

\[
\mathcal{C}_{\mathrm{target}} = \{C_r \in \mathcal{C} \mid C_r \cap V_y \ne \emptyset\}.
\]

Then retain all exogenous frequency nodes that appear in these communities:

\[
V_{\mathrm{valid}} = \{v_{i,k} \mid i \ne 0, v_{i,k} \in \bigcup_{C \in \mathcal{C}_{\mathrm{target}}} C\}.
\]

Map selected frequency nodes back to variable-level features:

\[
\mathcal{Z}_{\mathrm{selected}} = \{i \mid \exists k, v_{i,k} \in V_{\mathrm{valid}}\}.
\]

### 8.5 Optional Secondary Variable Ranking

For each selected exogenous variable \(i\), compute an importance score:

\[
\mathrm{Score}(i)=\sum_{v_{i,k}\in V_{\mathrm{valid}}} E_{i,k}.
\]

Sort variables by \(\mathrm{Score}(i)\) in descending order. Keep the top \(\lceil \beta |\mathcal{Z}_{\mathrm{selected}}| \rceil\) variables.

Recommended default from the original draft:

\[
\beta = 0.9.
\]

This secondary ranking removes the weakest selected variables while preserving most of the relevant information.

---

## 9. Forecasting Backbone: NBEATSx

After feature selection, train NBEATSx using:

- residualized target history \(\mathbf{y}_{\mathrm{res}}\), and
- selected exogenous variables \(\mathbf{Z}_{\mathrm{selected}}\).

The NBEATSx backbone can preserve its original block design:

1. **Trend block** for slowly varying components.
2. **Seasonality block** for periodic components.
3. **Exogenous block** for selected external variables.

The method does not require modifying the internal NBEATSx block architecture. The improvement is mainly introduced through input-side spectral decomposition and graph-based exogenous feature selection.

A reproducible implementation should report:

- lookback length \(L\),
- forecast horizon \(H\),
- stack types and number of blocks,
- hidden dimension,
- number of layers per block,
- optimizer,
- learning rate,
- batch size,
- number of epochs,
- early stopping rule,
- random seeds,
- whether the final forecast predicts \(y\) directly or predicts the residual and then adds back removed components.

Recommended forecasting target handling:

- During training, predict the original target \(y\) if the removed intrinsic component can be reconstructed for the forecast horizon.
- Otherwise, predict \(y_{\mathrm{res}}\) and evaluate against a residualized target only as an ablation.
- For a fair main experiment, the final output should be mapped back to the original target scale before computing MSE and MAE.

---

## 10. End-to-End Pseudocode

```python
class NBEATSxDC:
    def __init__(self, K1=50, K2=3, theta=0.25, beta=0.9,
                 tau_max=0, community_method="structural_entropy"):
        self.K1 = K1
        self.K2 = K2
        self.theta = theta
        self.beta = beta
        self.tau_max = tau_max
        self.community_method = community_method

    def fit(self, X_train, y_index=0):
        # X_train: [T, n+1]
        y = X_train[:, y_index]
        Z = remove_column(X_train, y_index)

        # Stage 1: remove intrinsic target frequencies
        y_res, intrinsic_info = remove_top_fft_components(y, K=self.K1)

        # Stage 2: extract driving frequency components
        variables = [y_res] + [Z[:, i] for i in range(Z.shape[1])]
        nodes = []
        for i, series in enumerate(variables):
            comps = extract_top_fft_components(series, K=self.K2)
            for comp in comps:
                nodes.append(make_node(variable_id=i, component=comp))

        # Build weighted graph
        W = build_variable_frequency_graph(
            nodes,
            use_dtw=True,
            tmi_tau_max=self.tau_max,
        )
        W = sparsify(W, theta=self.theta)

        # Community detection
        communities = detect_communities(W, method=self.community_method)

        # Select variables associated with target communities
        selected_vars = select_target_associated_variables(
            nodes=nodes,
            communities=communities,
            beta=self.beta,
        )

        # Train NBEATSx with residual target and selected exogenous variables
        X_model = build_nbeatsx_inputs(y_res, Z[:, selected_vars])
        self.model = train_nbeatsx(X_model)
        self.selected_vars = selected_vars
        self.intrinsic_info = intrinsic_info
        return self

    def predict(self, X_context):
        y_context = X_context[:, 0]
        Z_context = X_context[:, 1:]
        y_res_context = apply_stage1_residualization(y_context, self.intrinsic_info)
        X_model = build_nbeatsx_inputs(y_res_context, Z_context[:, self.selected_vars])
        y_hat_res = self.model.predict(X_model)
        y_hat = reconstruct_original_scale(y_hat_res, self.intrinsic_info)
        return y_hat
```

---

## 11. Reproducible Implementation Details

### 11.1 Data Format

Use a single table or array:

```text
timestamp, target, exog_1, exog_2, ..., exog_n
```

Requirements:

- timestamps must be uniformly sampled or resampled before training;
- missing values must be imputed before FFT;
- all variables should be normalized using training-set statistics only;
- normalization parameters must be reused for validation and test splits.

### 11.2 Data Splits

Use chronological splits only. Do not shuffle time indices.

Recommended split:

```text
train: 70%
validation: 10%
test: 20%
```

For high-frequency ship-motion data, also report whether splits are made:

- globally across the entire series, or
- segment-wise by sea state / voyage / day / scenario.

If frequency patterns vary strongly within a day, segment-wise training is recommended. However, the segmentation rule must be deterministic and reported.

### 11.3 Window Construction

For each sample:

\[
\mathbf{X}_{t-L+1:t} \rightarrow y_{t+1:t+H}.
\]

Use a fixed stride, e.g. stride \(=1\) for full training or larger stride for faster experiments.

Report all horizons, e.g.:

```text
H ∈ {96, 192, 336, 720}
```

or domain-specific horizons for ship-motion prediction.

### 11.4 Metrics

Use at least:

\[
\mathrm{MSE} = \frac{1}{N}\sum_i (y_i - \hat{y}_i)^2,
\]

\[
\mathrm{MAE} = \frac{1}{N}\sum_i |y_i - \hat{y}_i|.
\]

Optionally include:

- RMSE,
- MAPE only if the target is safely nonzero,
- inference latency,
- graph construction time,
- number of selected variables.

### 11.5 Seeds

Run at least 3 seeds. Prefer 5 seeds for conference submission.

Report:

```text
mean ± standard deviation
```

for all main results.

---

## 12. Default Hyperparameters

| Hyperparameter | Meaning | Suggested values |
|---|---|---|
| \(K_1\) | Stage-1 intrinsic frequencies | 10, 20, 50 |
| \(K_2\) | Stage-2 driving frequencies per variable | 1, 2, 3, 4, 5 |
| \(\theta\) | Graph sparsification ratio | 0.25 default; tune in {0.1, 0.25, 0.5} |
| \(\beta\) | Secondary variable retention ratio | 0.9 default |
| \(\tau_{\max}\) | Maximum TMI lag | 0, 3, 5, 10 |
| Tree height | Structural entropy tree height | 2 or 3 for practical search |
| Lookback \(L\) | Input context length | 2H to 7H, following NBEATS practice |

Important: \(K_2\) has a much stronger effect on graph construction time than \(K_1\), because the graph size scales with \((n+1)K_2\).

---

## 13. Complexity Analysis

The graph contains:

\[
|V| = (n+1)K_2
\]

nodes.

A dense graph over cross-variable frequency pairs requires approximately:

\[
O(n^2K_2^2)
\]

pairwise comparisons.

If DTW is used between two component series of length \(\ell_a\) and \(\ell_b\), its cost is:

\[
O(\ell_a \ell_b).
\]

Therefore, graph construction is the dominant cost:

\[
T_{\mathrm{graph}} \approx
\sum_{i<j}\sum_{a=1}^{K_2}\sum_{b=1}^{K_2}
O(\ell_{i,a}\ell_{j,b})
+
O(\mathrm{TMI}).
\]

If all aligned component lengths are approximately \(\ell\), then:

\[
T_{\mathrm{graph}} = O(n^2K_2^2\ell^2).
\]

The original draft reports that increasing \(K_2\) from 1 to 5 causes total runtime to grow superlinearly, with graph construction becoming the main bottleneck. This is expected because \(K_2\) increases both the number of nodes and the number of pairwise frequency-component comparisons.

---

## 14. Required Ablation Studies

For a conference paper, the following ablations are essential.

### 14.1 Backbone Ablation

| Variant | Purpose |
|---|---|
| NBEATSx with all exogenous variables | Tests whether feature selection helps |
| NBEATSx with no exogenous variables | Tests value of external variables |
| NBEATSx-DC full model | Main method |

### 14.2 Decomposition Ablation

| Variant | Purpose |
|---|---|
| Remove stage 1 | Tests whether intrinsic-frequency removal reduces false associations |
| Remove stage 2 | Tests whether driving-frequency graph is necessary |
| FFT top-K only without graph | Tests whether spectral filtering alone is enough |

### 14.3 Feature-Selection Ablation

| Variant | Purpose |
|---|---|
| Pearson correlation selection | Linear baseline |
| Mutual information without DTW | Tests value of temporal alignment |
| TMI + DTW without energy weighting | Tests value of spectral energy |
| Random variable selection with same selected count | Controls for dimensionality reduction |
| Louvain or Leiden instead of structural entropy | Tests value of structural entropy objective |

### 14.4 Hyperparameter Sensitivity

Report sensitivity for:

- \(K_1\),
- \(K_2\),
- \(\theta\),
- \(\beta\),
- structural entropy tree height,
- number of selected variables.

---

## 15. Suggested Experimental Table Format

### Main Forecasting Results

| Dataset | Horizon | Metric | NBEATSx | NBEATSx + FFT | NBEATSx + Selection | NBEATSx-DC |
|---|---:|---|---:|---:|---:|---:|
| ShipMotion | H1 | MSE | | | | |
| ShipMotion | H1 | MAE | | | | |
| ShipMotion | H2 | MSE | | | | |
| ShipMotion | H2 | MAE | | | | |

### Runtime Results

| \(K_2\) | FFT time | Graph time | Selection time | Training time | Inference time | Total time |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | | | | | | |
| 2 | | | | | | |
| 3 | | | | | | |
| 4 | | | | | | |
| 5 | | | | | | |

### Selected Variable Analysis

| Segment | Selected variables | Number selected | Dominant target periods | Notes |
|---|---|---:|---|---|
| Segment 1 | | | | |
| Segment 2 | | | | |

---

## 16. Conference-Paper Method Section Draft

### 16.1 Method Overview

We propose NBEATSx-DC, a frequency-aware and community-driven extension of NBEATSx for multivariate ship-motion forecasting. Unlike standard NBEATSx, which directly incorporates all exogenous variables, NBEATSx-DC first decomposes the target and exogenous variables in the frequency domain, then constructs a variable-frequency association graph to identify exogenous variables that are dynamically coupled with the target. The selected variables and the residualized target signal are finally passed to an NBEATSx forecasting backbone.

### 16.2 Dual-Stage Spectral Decomposition

The first stage removes dominant intrinsic frequencies from the target variable. These frequencies correspond to vessel-specific natural oscillations and may not reflect external environmental drivers. We compute the Fourier spectrum of the target sequence and reconstruct the top-\(K_1\) amplitude components. The residual signal is obtained by subtracting the reconstructed intrinsic component from the original target.

The second stage extracts candidate driving frequencies from both the residual target and all exogenous variables. For each variable, we compute the Fourier spectrum and retain the top-\(K_2\) nonzero frequency components. Each retained component is represented as a variable-frequency node with its reconstructed temporal component and normalized spectral energy.

### 16.3 Variable-Frequency Association Graph

To model cross-variable and cross-scale dependencies, we build a weighted graph whose nodes are variable-frequency components. For two nodes from different variables, we first align their reconstructed component series using dynamic time warping. We then compute temporal mutual information over the aligned series and multiply it by the corresponding normalized spectral energies. This produces an edge weight that jointly measures nonlinear temporal coupling and spectral importance. Weak edges are removed by thresholding relative to the maximum edge weight.

### 16.4 Community-Driven Feature Selection

We apply structural-entropy-based community detection to the variable-frequency graph. Communities containing at least one target-frequency node are treated as target-associated communities. Exogenous variables with frequency nodes in these communities are selected as useful predictors. Optionally, variables are further ranked by the sum of normalized spectral energy of their selected nodes, and only the top \(\beta\) fraction is retained.

### 16.5 Forecasting

The selected exogenous variables and the residualized target history are used as inputs to the NBEATSx backbone. The backbone contains trend, seasonality, and exogenous blocks. Since the proposed modules operate before the backbone, the architecture of NBEATSx remains unchanged. This design makes the method easy to integrate into existing NBEATSx implementations.

---

## 17. Minimal Reproduction Plan

### Step 1: Implement FFT Component Extraction

Implement:

```python
extract_top_fft_components(series, K, exclude_dc=True)
```

Return:

- frequency indices,
- amplitudes,
- normalized energies,
- reconstructed single-frequency components.

### Step 2: Implement Stage-1 Residualization

Implement:

```python
remove_top_fft_components(target, K1)
```

Return:

- residual target,
- removed intrinsic component,
- selected intrinsic frequencies.

### Step 3: Implement Graph Construction

Implement:

```python
build_variable_frequency_graph(nodes, theta, use_dtw=True, tmi_tau_max=0)
```

Start with a simple version:

- no lag, \(\tau=0\),
- histogram-based mutual information,
- exact DTW or fastdtw,
- thresholded dense adjacency matrix.

### Step 4: Implement Community Selection

Start with Louvain or Leiden to debug the pipeline. Then replace with structural-entropy minimization when available.

### Step 5: Integrate with NBEATSx

Use any stable NBEATSx implementation. Keep the backbone unchanged at first. Only change the input variables.

### Step 6: Run Sanity Experiments

Run:

1. NBEATSx with all variables.
2. NBEATSx with selected variables only.
3. NBEATSx-DC without stage 1.
4. Full NBEATSx-DC.

### Step 7: Add Runtime Profiling

Measure:

- FFT time,
- DTW/TMI graph time,
- community detection time,
- training time,
- inference time.

---

## 18. Common Reproduction Pitfalls

1. **Using future data in FFT decomposition**  
   Compute decomposition only within the training segment or lookback context. Do not compute spectral components using the entire train+test series.

2. **Leaking normalization statistics**  
   Fit scalers only on the training set.

3. **Incorrect FFT inverse reconstruction**  
   For real signals, use rFFT/irFFT consistently. Do not lose conjugate symmetry by mixing complex FFT APIs incorrectly.

4. **Selecting the DC component as a frequency**  
   The DC component corresponds to the mean, not a periodic component. Exclude it unless explicitly intended.

5. **Unstable mutual information estimation**  
   Histogram bin count or kernel estimator settings can change results. Fix them and report them.

6. **Changing selected variables across validation/test unfairly**  
   Feature selection should be fit on training data and then applied consistently to validation/test.

7. **Over-claiming causality**  
   TMI and graph communities indicate statistical association, not causal influence. Use terms such as coupled, associated, or predictive rather than causal unless causal validation is performed.

8. **Ignoring graph construction cost**  
   DTW and TMI over all frequency pairs can dominate runtime. Always report runtime and selected variable count.

9. **Unclear final target reconstruction**  
   State whether evaluation is performed on the original target or residual target. Main results should usually be on the original target scale.

---

## 19. What to Strengthen Before Submission

The original draft contains a promising method, but several parts should be strengthened before a conference submission:

1. **Clarify whether stage-1 removed intrinsic components are added back during forecasting.**  
   This is essential for fair evaluation on the original target.

2. **Specify the structural entropy optimizer.**  
   Give exact pseudocode, stopping conditions, and complexity. If using an existing implementation, cite it and freeze the version.

3. **Make TMI estimation deterministic.**  
   Define binning, lag range, estimator type, and normalization.

4. **Avoid causal language.**  
   The method identifies predictive associations, not verified physical causality.

5. **Report selected variable statistics.**  
   Show that the method does not simply select most variables.

6. **Add strong baselines.**  
   Compare against simple feature selection, full exogenous input, no exogenous input, and modern time-series baselines.

7. **Include significance testing.**  
   Use multiple seeds and report mean/std. For small improvements, include paired statistical tests over segments or horizons.

---

## 20. Paper Contribution Claims

A safe and defensible contribution statement is:

1. We propose a dual-stage spectral decomposition module that separates target-intrinsic frequencies from potential externally driven frequency components.
2. We introduce a variable-frequency association graph that combines DTW-aligned temporal mutual information with spectral energy weighting.
3. We formulate exogenous feature selection as a community detection problem and use structural entropy minimization to identify target-associated variable communities.
4. We integrate the selected variables into NBEATSx without modifying its internal forecasting blocks, making the method modular and easy to reproduce.
5. We provide systematic ablations on decomposition stages, graph construction choices, community detection, and runtime-performance trade-offs.

---

## 21. Suggested Repository Structure

```text
nbeatsx_dc/
  README.md
  requirements.txt
  configs/
    shipmotion.yaml
    etth2.yaml
  data/
    README.md
  src/
    decomposition.py
    graph_builder.py
    mutual_information.py
    dtw_align.py
    community.py
    feature_selection.py
    nbeatsx_wrapper.py
    train.py
    evaluate.py
  scripts/
    run_main.sh
    run_ablation.sh
    run_sensitivity.sh
  results/
    tables/
    figures/
  tests/
    test_fft_reconstruction.py
    test_graph_builder.py
    test_feature_selection.py
```

Minimum unit tests:

- FFT reconstruction returns a signal with the same length.
- Removing zero components leaves the original signal unchanged.
- Graph adjacency is symmetric.
- Same-variable edges are zero.
- Selected variables are a subset of exogenous variables.
- No validation/test values are used during training feature selection.

---

## 22. Minimal Command-Line Interface

```bash
python src/train.py \
  --data data/ship_motion.csv \
  --target roll_angle \
  --horizon 96 \
  --lookback 336 \
  --model nbeatsx_dc \
  --K1 50 \
  --K2 3 \
  --theta 0.25 \
  --beta 0.9 \
  --community structural_entropy \
  --seed 1
```

Ablation example:

```bash
python src/train.py \
  --data data/ship_motion.csv \
  --target roll_angle \
  --horizon 96 \
  --lookback 336 \
  --model nbeatsx_dc \
  --disable_stage1 \
  --K2 3 \
  --seed 1
```

---

## 23. Final Reproducibility Checklist

Before claiming improvement, verify the following:

- [ ] Chronological data split is fixed and reported.
- [ ] Normalization is fit only on the training set.
- [ ] FFT decomposition does not use future test data.
- [ ] \(K_1\), \(K_2\), \(\theta\), \(\beta\), and tree height are reported.
- [ ] TMI estimator and DTW implementation are fixed.
- [ ] Community detection algorithm is deterministic or seeded.
- [ ] Number of selected variables is reported.
- [ ] Main metrics are computed on the original target scale.
- [ ] At least 3 random seeds are used.
- [ ] Runtime is reported, especially graph construction time.
- [ ] Ablations show which module causes the gain.
- [ ] Code commit hash and environment are recorded.

---

## 24. One-Paragraph Summary

NBEATSx-DC is a modular extension of NBEATSx for multivariate ship-motion forecasting. It first removes dominant intrinsic frequency components from the target, then extracts candidate driving frequencies from the residual target and all exogenous variables. These frequency components form a weighted variable-frequency graph, where edges combine DTW-aligned temporal mutual information and spectral energy. Structural-entropy-based community detection identifies exogenous variables that share strong communities with target frequency components. The selected variables are then fed into an unchanged NBEATSx backbone, reducing noisy inputs while preserving predictive cross-variable dynamics.
