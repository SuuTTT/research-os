"""
feature_selection.py
Map graph community detection results back to selected exogenous variables.
"""

import math
import networkx as nx
from typing import Dict, List
from .decomposition import FreqComponent


def select_features(
    nodes: List[FreqComponent],
    partition: Dict[int, int],   # node_index → community_id
    beta: float = 0.9,
) -> List[int]:
    """
    Select exogenous variable indices whose frequency nodes share a
    community with at least one target (var_id=0) node.

    Parameters
    ----------
    nodes     : list of FreqComponent (same order as graph nodes)
    partition : {node_index: community_id}
    beta      : top-beta fraction of selected vars to keep (secondary ranking)

    Returns
    -------
    selected_var_ids : sorted list of exogenous variable ids (1-based)
    """
    # Communities that contain at least one target node (var_id == 0)
    target_comms = set(
        partition[i]
        for i, node in enumerate(nodes)
        if node.var_id == 0 and i in partition
    )

    if not target_comms:
        # If no target community found, return all exog vars (safe fallback)
        all_exog = sorted(set(n.var_id for n in nodes if n.var_id != 0))
        return all_exog

    # Exogenous nodes in target-associated communities
    selected: Dict[int, float] = {}  # var_id → cumulative energy score
    for i, node in enumerate(nodes):
        if node.var_id == 0:
            continue
        if i in partition and partition[i] in target_comms:
            selected[node.var_id] = selected.get(node.var_id, 0.0) + node.energy

    if not selected:
        return []

    # Secondary ranking: keep top-beta fraction by energy score
    ranked = sorted(selected.items(), key=lambda x: x[1], reverse=True)
    keep_n = max(1, math.ceil(beta * len(ranked)))
    selected_ids = [var_id for var_id, _ in ranked[:keep_n]]
    return sorted(selected_ids)


def selection_summary(
    nodes: List[FreqComponent],
    selected_ids: List[int],
    n_exog: int,
) -> dict:
    """Return a human-readable summary of the feature selection result."""
    scores: Dict[int, float] = {}
    for node in nodes:
        if node.var_id != 0:
            scores[node.var_id] = scores.get(node.var_id, 0.0) + node.energy

    return {
        "n_exog_total": n_exog,
        "n_selected": len(selected_ids),
        "selected_var_ids": selected_ids,
        "selection_ratio": len(selected_ids) / n_exog if n_exog > 0 else 0.0,
        "energy_scores": {
            vid: round(scores.get(vid, 0.0), 4)
            for vid in selected_ids
        },
    }
