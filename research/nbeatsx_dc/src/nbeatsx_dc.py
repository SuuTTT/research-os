"""
nbeatsx_dc.py
Main NBEATSx-DC pipeline class.

Orchestrates:
  1. Stage-1 intrinsic frequency removal
  2. Stage-2 driving frequency extraction → graph nodes
  3. Graph construction (DTW + TMI edge weights)
  4. Community detection (structural entropy or Louvain)
  5. Feature selection (community-based)
  6. Backbone training on selected features

The backbone defaults to SlidingWindowRidge for fast prototyping.
Swap backbone= for any class with .fit(train) / .evaluate(test) interface.
"""

import numpy as np
from typing import Optional, Literal

from .decomposition import remove_intrinsic_frequencies, build_all_nodes
from .graph_builder import build_graph
from .community import detect_communities
from .feature_selection import select_features, selection_summary
from .backbone import SlidingWindowRidge


class NBEATSxDC:
    def __init__(
        self,
        K1: int = 20,
        K2: int = 3,
        theta: float = 0.50,
        theta_mode: str = "percentile",  # 'max' | 'percentile' | 'none'
        beta: float = 0.9,
        tau_max: int = 0,
        use_dtw: bool = False,          # True = DTW-align components (slower)
        community_method: Literal["structural_entropy", "louvain"] = "structural_entropy",
        lookback: int = 96,
        horizon: int = 96,
        backbone=None,
        verbose: bool = True,
    ):
        self.K1 = K1
        self.K2 = K2
        self.theta = theta
        self.theta_mode = theta_mode
        self.beta = beta
        self.tau_max = tau_max
        self.use_dtw = use_dtw
        self.community_method = community_method
        self.lookback = lookback
        self.horizon = horizon
        self.verbose = verbose

        # Backbone: default to RidgeCV sliding-window
        self.backbone = backbone or SlidingWindowRidge(lookback=lookback, horizon=horizon)

        # Fitted state
        self.selected_var_ids_: Optional[list] = None
        self.intrinsic_: Optional[np.ndarray] = None
        self.selection_summary_: Optional[dict] = None
        self.nodes_: Optional[list] = None
        self.partition_: Optional[dict] = None

    def _log(self, msg: str):
        if self.verbose:
            print(f"[NBEATSx-DC] {msg}")

    def fit(
        self,
        target: np.ndarray,    # (T,)
        exog: np.ndarray,      # (T, n_exog)
    ) -> "NBEATSxDC":
        """
        Run the full pipeline on training data.

        Parameters
        ----------
        target : 1-D array, the forecasting target
        exog   : 2-D array, exogenous variables (columns)
        """
        T, n_exog = len(target), exog.shape[1]

        # ---- Stage 1: remove intrinsic target frequencies ----
        self._log(f"Stage 1: removing top-{self.K1} intrinsic frequencies from target")
        residual, intrinsic, intrinsic_idx = remove_intrinsic_frequencies(
            target, K1=self.K1
        )
        self.intrinsic_ = intrinsic

        # ---- Stage 2: extract driving frequencies ----
        self._log(f"Stage 2: extracting top-{self.K2} freq components per variable "
                  f"({n_exog + 1} variables → {(n_exog + 1) * self.K2} nodes)")
        nodes = build_all_nodes(residual, exog, K2=self.K2)
        self.nodes_ = nodes

        # ---- Build variable-frequency graph ----
        self._log(f"Building graph (theta={self.theta}, mode={self.theta_mode}, "
                  f"DTW={self.use_dtw}, tau_max={self.tau_max})")
        G = build_graph(nodes, theta=self.theta, theta_mode=self.theta_mode,
                        tau_max=self.tau_max, use_dtw=self.use_dtw)
        self._log(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges "
                  f"(after sparsification)")

        # ---- Community detection ----
        self._log(f"Community detection: method={self.community_method!r}")
        partition = detect_communities(G, method=self.community_method)
        self.partition_ = partition
        n_comms = len(set(partition.values())) if partition else 0
        self._log(f"Found {n_comms} communities")

        # ---- Feature selection ----
        selected_ids = select_features(nodes, partition, beta=self.beta)
        self.selected_var_ids_ = selected_ids
        summary = selection_summary(nodes, selected_ids, n_exog)
        self.selection_summary_ = summary
        self._log(
            f"Selected {summary['n_selected']}/{n_exog} exogenous variables "
            f"({100 * summary['selection_ratio']:.0f}%): "
            f"var_ids={selected_ids}"
        )

        # ---- Train backbone ----
        # selected_ids are 1-based exog column indices
        exog_selected = self._select_exog(exog, selected_ids)
        # Stack target + selected exog: (T, 1 + n_selected)
        train_data = np.column_stack([target, exog_selected])
        self._log(f"Training backbone on {train_data.shape[1]} features "
                  f"(target + {exog_selected.shape[1]} exog)")
        self.backbone.fit(train_data)
        self._log("Done.")
        return self

    def evaluate(
        self,
        target: np.ndarray,  # (T_test,)
        exog: np.ndarray,    # (T_test, n_exog)
    ) -> dict:
        """Evaluate on held-out test set."""
        exog_selected = self._select_exog(exog, self.selected_var_ids_)
        test_data = np.column_stack([target, exog_selected])
        return self.backbone.evaluate(test_data)

    def _select_exog(self, exog: np.ndarray, ids: list) -> np.ndarray:
        """ids are 1-based variable indices → 0-based exog columns."""
        if not ids:
            return np.zeros((len(exog), 0))
        cols = [i - 1 for i in ids]  # convert 1-based to 0-based
        return exog[:, cols]
