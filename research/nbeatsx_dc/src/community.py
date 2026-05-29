"""
community.py
Community detection wrapper supporting Louvain and structural entropy.
"""

import networkx as nx
from typing import Dict, Literal


def detect_communities(
    G: nx.Graph,
    method: Literal["structural_entropy", "louvain"] = "structural_entropy",
    seed: int = 42,
) -> Dict[int, int]:
    """
    Detect communities in *G*.

    Parameters
    ----------
    G      : weighted undirected networkx Graph
    method : 'structural_entropy' (default) or 'louvain'
    seed   : random seed for Louvain

    Returns
    -------
    partition : dict {node_id: community_id}
    """
    if len(G.nodes()) == 0:
        return {}

    if method == "structural_entropy":
        from .structural_entropy import minimize_structural_entropy
        return minimize_structural_entropy(G)

    elif method == "louvain":
        try:
            import community as community_louvain
            partition = community_louvain.best_partition(G, weight="weight",
                                                         random_state=seed)
            return partition
        except ImportError:
            raise ImportError(
                "python-louvain not installed. "
                "Run: pip install python-louvain"
            )
    else:
        raise ValueError(f"Unknown community method: {method!r}. "
                         "Use 'structural_entropy' or 'louvain'.")
