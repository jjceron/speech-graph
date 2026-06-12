"""Graph metrics computed directly from a NetworkX DiGraph (SRL-based)."""

from __future__ import annotations

import networkx as nx
import numpy as np

METRICS = [
    "wc", "nodes", "edges", "re", "pe", "l1", "l2", "l3",
    "lcc", "lsc", "atd", "density", "diameter", "asp", "cc",
]


def _empty() -> dict[str, float]:
    return {m: 0.0 for m in METRICS} | {
        "diameter": float("nan"), "asp": float("nan"), "cc": float("nan"),
    }


def compute_srl_metrics_weighted(G: nx.DiGraph, window_size: int = 3) -> dict[str, float]:
    """Weighted variant of :func:`compute_srl_metrics`.

    Same as the original except:
      - ``edges`` = sum of all edge weights (instead of unique edge count).
      - ``density`` = total edge weight / (n * (n-1)) for directed complete
        reference (instead of unique undirected non-self pairs / nC2).

    Column names are identical to the original; use a different output file
    (e.g. ``*_weight.txt``) to distinguish.
    """
    result = compute_srl_metrics(G, window_size=window_size)
    node_count = G.number_of_nodes()
    edge_sum = int(sum(d.get("weight", 1) for _, _, d in G.edges(data=True)))

    result["edges"] = edge_sum
    if node_count > 1:
        result["density"] = float(edge_sum / (node_count * (node_count - 1)))
    else:
        result["density"] = 0.0
    return result


def compute_srl_metrics(G: nx.DiGraph, window_size: int = 3) -> dict[str, float]:
    """Compute the 15 SpeechGraph metrics from an SRL-based DiGraph.

    Args:
        G: A NetworkX DiGraph with ``weight`` edge attributes.
        window_size: Number of sentences in the window (used for ``wc``).

    Returns:
        Dict with 15 metric values, matching ``METRICS``.
    """
    node_count = G.number_of_nodes()
    edge_unique = G.number_of_edges()
    edge_sum = int(sum(d.get("weight", 1) for _, _, d in G.edges(data=True)))

    if node_count == 0:
        return _empty()

    wc = window_size

    # Build adjacency matrix for matrix-based metrics
    nodes_list = list(G.nodes())
    n = len(nodes_list)
    idx = {node: i for i, node in enumerate(nodes_list)}
    adj = np.zeros((n, n), dtype=np.float64)
    for u, v, d in G.edges(data=True):
        adj[idx[u], idx[v]] = d.get("weight", 1)

    # Self-loops
    self_loop_count = 0
    for u, v in G.edges():
        if u == v:
            w = G.edges[u, v].get("weight", 1)
            self_loop_count += w

    # Repeated edges: edges with weight > 1
    repeated = int(
        sum(max(0, d.get("weight", 1) - 1) for _, _, d in G.edges(data=True))
    )

    # Parallel (bidirectional) edges
    parallel = 0
    visited: set[frozenset] = set()
    for u, v in G.edges():
        if u == v:
            continue
        pair = frozenset((u, v))
        if pair in visited:
            continue
        visited.add(pair)
        w_uv = G.get_edge_data(u, v, default={"weight": 0}).get("weight", 0)
        w_vu = G.get_edge_data(v, u, default={"weight": 0}).get("weight", 0)
        if w_uv > 0 and w_vu > 0:
            parallel += min(w_uv, w_vu)

    # Matrix-based metrics
    no_self = adj.copy()
    np.fill_diagonal(no_self, 0)

    l1 = int(self_loop_count)
    pe = int(np.trace(no_self @ no_self) / 2) if n else 0
    l2 = parallel
    l3 = int(np.trace(no_self @ no_self @ no_self) / 3) if n else 0

    # Undirected component analysis
    undirected = G.to_undirected()
    lcc = int(
        max((len(c) for c in nx.connected_components(undirected)), default=0)
    )
    lsc = int(
        max(
            (len(c) for c in nx.strongly_connected_components(G)), default=0
        )
    )
    atd = float(edge_sum / node_count) if node_count else 0.0

    # Density: unique undirected non-self pairs / all possible pairs
    undirected_edges: set[frozenset] = set()
    for u, v in G.edges():
        if u != v:
            undirected_edges.add(frozenset((u, v)))
    unique_pairs = len(undirected_edges)
    density = (
        float(unique_pairs / (node_count * (node_count - 1) / 2))
        if node_count > 1
        else 0.0
    )

    # Path-based metrics (diameter, asp, cc)
    diameter = float("nan")
    asp = float("nan")
    cc_val = float("nan")
    if node_count > 1 and undirected.number_of_edges() > 0:
        largest = max(nx.connected_components(undirected), key=len)
        component = undirected.subgraph(largest).copy()
        if component.number_of_nodes() > 1:
            diameter = float(nx.diameter(component))
            asp = float(nx.average_shortest_path_length(component))
            cc_val = float(nx.average_clustering(component))

    return {
        "wc": wc,
        "nodes": int(node_count),
        "edges": int(edge_unique),
        "re": int(repeated),
        "pe": int(pe),
        "l1": int(l1),
        "l2": int(l2),
        "l3": int(l3),
        "lcc": int(lcc),
        "lsc": int(lsc),
        "atd": float(atd),
        "density": float(density),
        "diameter": diameter,
        "asp": asp,
        "cc": cc_val,
    }
