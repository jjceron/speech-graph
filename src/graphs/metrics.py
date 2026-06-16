"""Graph metric computation for speech transcripts."""

from __future__ import annotations

from typing import Sequence

import networkx as nx
import numpy as np

from .builder import adjacency_matrix, edge_counts, split_by_boundaries

METRICS = [
    "wc", "nodes", "edges", "re", "pe", "l1", "l2", "l3",
    "lcc", "lsc", "atd", "density", "diameter", "asp", "cc",
]


def _as_segments(tokens_or_segments: Sequence[str] | Sequence[Sequence[str]]) -> list[list[str]]:
    if not tokens_or_segments:
        return []
    first = tokens_or_segments[0]
    if isinstance(first, str):
        return [list(tokens_or_segments)]
    return [list(seg) for seg in tokens_or_segments if len(seg) > 0]


def _empty() -> dict[str, float]:
    return {m: 0.0 for m in METRICS} | {"diameter": float("nan"), "asp": float("nan"), "cc": float("nan")}


def compute_metrics(
    tokens_or_segments: Sequence[str] | Sequence[Sequence[str]],
    segment_boundaries: list[bool] | None = None,
) -> dict[str, float]:
    """Compute all graph metrics for a token sequence.

    Args:
        tokens_or_segments: Flat list of tokens or list of segments.
        segment_boundaries: Optional boundary flags for splitting flat tokens.

    Returns:
        Dict with 15 metric values.
    """
    if segment_boundaries is not None:
        tokens = list(tokens_or_segments) if not tokens_or_segments or isinstance(tokens_or_segments[0], str) else [t for seg in tokens_or_segments for t in seg]
        segments = split_by_boundaries(tokens, segment_boundaries)
    else:
        segments = _as_segments(tokens_or_segments)

    wc = sum(len(seg) for seg in segments)
    if wc == 0:
        return _empty()

    tokens = [t for seg in segments for t in seg]
    nodes = list(dict.fromkeys(tokens))
    node_count = len(nodes)
    ec = edge_counts(segments)
    edge_total = int(sum(ec.values()))
    repeated_edges = int(sum(count - 1 for count in ec.values() if count > 1))

    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(ec.keys())
    undirected = graph.to_undirected()

    lcc = int(max((len(c) for c in nx.connected_components(undirected)), default=0)) if node_count else 0
    lsc = int(max((len(c) for c in nx.strongly_connected_components(graph)), default=0)) if node_count else 0
    atd = float(2.0 * edge_total / node_count) if node_count else 0.0

    unique_pairs = {frozenset((s, t)) for s, t in ec if s != t}
    unique_undirected_non_self = len(unique_pairs)
    density = float(unique_undirected_non_self / (node_count * (node_count - 1) / 2)) if node_count > 1 else 0.0

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

    adj = adjacency_matrix(ec, nodes)
    l1 = int(np.trace(adj)) if node_count else 0
    pe = int(edge_total - l1 - unique_undirected_non_self) if node_count else 0

    adj_dir = (adj > 0).astype(int)
    np.fill_diagonal(adj_dir, 0)
    adj_dir2 = adj_dir @ adj_dir
    l2 = int(np.trace(adj_dir2) // 2) if node_count else 0
    l3 = int(np.trace(adj_dir2 @ adj_dir) // 3) if node_count else 0

    return {
        "wc": int(wc),
        "nodes": int(node_count),
        "edges": int(edge_total),
        "re": int(repeated_edges),
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
