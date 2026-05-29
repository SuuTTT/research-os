"""
structural_entropy.py  (v2 — efficient incremental implementation)
Greedy 2D structural entropy minimisation for community detection.

Reference:
  Li & Pan (2016) "Structural Information and Dynamical Complexity of Networks"
  IEEE Trans. Information Theory 62(6): 3290-3339.

Key optimisation: per-community stats are cached and only recomputed for
the two merged communities — O(k*E) per pass instead of O(k^2*V).
"""

import math
from typing import Dict, Set
import networkx as nx

_EPS = 1e-12


def minimize_structural_entropy(G: nx.Graph) -> Dict[int, int]:
    """
    Greedy 2D structural entropy minimisation.

    Returns
    -------
    partition : dict {node_id: community_id}
    """
    nodes_list = list(G.nodes())
    if not nodes_list:
        return {}

    vol_G = sum(G.degree(v, weight="weight") for v in nodes_list)
    if vol_G == 0:
        return {v: i for i, v in enumerate(nodes_list)}

    # Initialise: each node is its own singleton community
    node2comm: Dict = {v: v for v in nodes_list}
    comm_members: Dict[int, set] = {v: {v} for v in nodes_list}
    comm_vol: Dict[int, float] = {}
    comm_g: Dict[int, float] = {}
    comm_leaf: Dict[int, float] = {}

    for v in nodes_list:
        dv = float(G.degree(v, weight="weight"))
        comm_vol[v] = dv
        comm_g[v] = dv      # singleton: all edges leave the community
        comm_leaf[v] = 0.0  # -(dv/vol_G)*log2(dv/dv) = 0

    def _cross_weight(ca, cb):
        sa = comm_members[ca]
        sb = comm_members[cb]
        if len(sa) > len(sb):
            sa, sb = sb, sa
        w = 0.0
        for u in sa:
            for nbr, data in G[u].items():
                if nbr in sb:
                    w += data.get("weight", 1.0)
        return w

    def _delta(ca, cb):
        vol_a, vol_b = comm_vol[ca], comm_vol[cb]
        vol_ab = vol_a + vol_b
        if vol_ab <= 0:
            return 0.0
        g_a, g_b = comm_g[ca], comm_g[cb]
        w_ab = _cross_weight(ca, cb)
        g_ab = g_a + g_b - 2.0 * w_ab

        def h1(g, vol):
            if g <= 0 or vol <= 0:
                return 0.0
            return -(g / vol_G) * math.log2(vol / vol_G + _EPS)

        h1_d = h1(g_ab, vol_ab) - h1(g_a, vol_a) - h1(g_b, vol_b)

        leaf_ab = sum(
            -(G.degree(v, weight="weight") / vol_G) *
            math.log2(G.degree(v, weight="weight") / vol_ab + _EPS)
            for v in comm_members[ca] | comm_members[cb]
            if G.degree(v, weight="weight") > 0
        )
        leaf_d = leaf_ab - comm_leaf[ca] - comm_leaf[cb]
        return h1_d + leaf_d

    def _recompute(ca):
        members = comm_members[ca]
        vol_c = sum(G.degree(v, weight="weight") for v in members)
        g_c = sum(
            data.get("weight", 1.0)
            for v in members
            for nbr, data in G[v].items()
            if nbr not in members
        )
        leaf_c = 0.0
        for v in members:
            dv = G.degree(v, weight="weight")
            if dv > 0 and vol_c > 0:
                leaf_c -= (dv / vol_G) * math.log2(dv / vol_c + _EPS)
        comm_vol[ca] = vol_c
        comm_g[ca] = g_c
        comm_leaf[ca] = leaf_c

    # Greedy merge loop
    improved = True
    while improved:
        improved = False
        best_delta = -_EPS
        best_pair = None

        seen: set = set()
        for u, v in G.edges():
            ca, cb = node2comm[u], node2comm[v]
            if ca == cb:
                continue
            pair = (min(ca, cb), max(ca, cb))
            if pair in seen:
                continue
            seen.add(pair)
            d = _delta(ca, cb)
            if d < best_delta:
                best_delta = d
                best_pair = (ca, cb)

        if best_pair is not None:
            ca, cb = best_pair
            for v in comm_members[cb]:
                node2comm[v] = ca
            comm_members[ca] |= comm_members[cb]
            del comm_members[cb], comm_vol[cb], comm_g[cb], comm_leaf[cb]
            _recompute(ca)
            improved = True

    unique = sorted(set(node2comm.values()))
    remap = {old: new for new, old in enumerate(unique)}
    return {v: remap[c] for v, c in node2comm.items()}
