"""
graph_builder.py
Build the variable-frequency association graph.

Nodes  : FreqComponent objects (one per (variable, frequency) pair)
Edges  : only between nodes of *different* variables
Weights: E_i * E_j * TMI(c_i, c_j)   (optionally with DTW alignment)
"""

import numpy as np
import networkx as nx
from typing import List, Optional
from itertools import combinations

from .decomposition import FreqComponent
from .mutual_info import temporal_mi


def _dtw_align(a: np.ndarray, b: np.ndarray) -> tuple:
    """
    Align two arrays with fastdtw (falls back to identity if lengths match).
    Returns (a_aligned, b_aligned).
    """
    if len(a) == len(b):
        return a, b
    try:
        from fastdtw import fastdtw
        from scipy.spatial.distance import euclidean
        _, path = fastdtw(a, b, dist=euclidean)
        idx_a = [p[0] for p in path]
        idx_b = [p[1] for p in path]
        return a[idx_a], b[idx_b]
    except ImportError:
        # Trim to shorter length as cheap fallback
        n = min(len(a), len(b))
        return a[:n], b[:n]


def build_graph(
    nodes: List[FreqComponent],
    theta: float = 0.25,
    theta_mode: str = "percentile",
    tau_max: int = 0,
    use_dtw: bool = True,
    bins: int = 20,
) -> nx.Graph:
    """
    Build a weighted undirected graph over frequency-component nodes.

    Parameters
    ----------
    nodes       : list of FreqComponent from decomposition.build_all_nodes
    theta       : sparsification parameter
    theta_mode  : 'max'        → keep edges ≥ theta * max(W)  [original spec]
                  'percentile' → keep top (1-theta) fraction of edges
                                 e.g. theta=0.5 keeps top-50% by weight
                  'none'       → keep all non-zero edges
    tau_max     : max lag for TMI
    use_dtw     : whether to DTW-align components before MI computation
    bins        : histogram bins for MI estimation

    Returns
    -------
    G : nx.Graph where nodes are integer indices into *nodes* list
        with attributes: var_id, freq_idx, energy, period
    """
    G = nx.Graph()

    # Add nodes
    for idx, node in enumerate(nodes):
        G.add_node(idx,
                   var_id=node.var_id,
                   freq_idx=node.freq_idx,
                   energy=node.energy,
                   period=node.period)

    # Compute all cross-variable edge weights
    edges = []
    for i, j in combinations(range(len(nodes)), 2):
        if nodes[i].var_id == nodes[j].var_id:
            continue  # no intra-variable edges
        ci, cj = nodes[i].component, nodes[j].component
        if use_dtw:
            ci, cj = _dtw_align(ci, cj)
        mi = temporal_mi(ci, cj, tau_max=tau_max, bins=bins)
        w = nodes[i].energy * nodes[j].energy * mi
        edges.append((i, j, w))

    if not edges:
        return G

    weights = [w for _, _, w in edges]

    # Compute threshold
    if theta_mode == "max":
        threshold = theta * max(weights)
    elif theta_mode == "percentile":
        # keep top (1-theta) fraction of edges by weight
        threshold = float(np.percentile(weights, 100 * theta))
    elif theta_mode == "none":
        threshold = 0.0
    else:
        raise ValueError(f"Unknown theta_mode: {theta_mode!r}")

    for i, j, w in edges:
        if w >= threshold:
            G.add_edge(i, j, weight=w)

    return G
