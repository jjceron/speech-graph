from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import networkx as nx
import numpy as np

CANONICAL_METRICS = [
    "wc", "nodes", "edges", "re", "pe", "l1", "l2", "l3", "lcc", "lsc",
    "atd", "density", "diameter", "asp", "cc",
]


def _as_segments(tokens_or_segments: Sequence[str] | Sequence[Sequence[str]]) -> list[list[str]]:
    if not tokens_or_segments:
        return []
    first = tokens_or_segments[0]  # type: ignore[index]
    if isinstance(first, str):
        return [list(tokens_or_segments)]  # type: ignore[arg-type]
    return [list(segment) for segment in tokens_or_segments if len(segment) > 0]  # type: ignore[arg-type]


def _edge_counts(segments: list[list[str]]) -> Counter[tuple[str, str]]:
    counts: Counter[tuple[str, str]] = Counter()
    for segment in segments:
        counts.update(zip(segment[:-1], segment[1:]))
    return counts


def _adjacency(edge_counts: Counter[tuple[str, str]], nodes: list[str]) -> np.ndarray:
    index = {node: i for i, node in enumerate(nodes)}
    matrix = np.zeros((len(nodes), len(nodes)), dtype=int)
    for (source, target), count in edge_counts.items():
        matrix[index[source], index[target]] = count
    return matrix


def _parallel_edges(edge_counts: Counter[tuple[str, str]]) -> int:
    pairs = set()
    total = 0
    for source, target in edge_counts:
        if source == target:
            continue
        key = tuple(sorted((source, target)))
        if key in pairs:
            continue
        pairs.add(key)
        a, b = key
        total += min(edge_counts.get((a, b), 0), edge_counts.get((b, a), 0))
    return int(total)


def _empty() -> dict[str, float]:
    return {
        "wc": 0, "nodes": 0, "edges": 0, "re": 0, "pe": 0,
        "l1": 0, "l2": 0, "l3": 0, "lcc": 0, "lsc": 0,
        "atd": 0.0, "density": 0.0, "diameter": np.nan, "asp": np.nan, "cc": np.nan,
    }


def compute_metrics(tokens_or_segments: Sequence[str] | Sequence[Sequence[str]]) -> dict[str, float]:
    """Compute directed speech-graph attributes without crossing break boundaries."""
    segments = _as_segments(tokens_or_segments)
    wc = sum(len(segment) for segment in segments)
    if wc == 0:
        return _empty()

    tokens = [token for segment in segments for token in segment]
    nodes = list(dict.fromkeys(tokens))
    node_count = len(nodes)
    edge_counts = _edge_counts(segments)
    edge_total = int(sum(edge_counts.values()))
    repeated_edges = int(sum(count - 1 for count in edge_counts.values() if count > 1))
    pe = _parallel_edges(edge_counts)

    dg = nx.DiGraph()
    dg.add_nodes_from(nodes)
    dg.add_edges_from(edge_counts.keys())
    ug = dg.to_undirected()

    lcc = int(max((len(c) for c in nx.connected_components(ug)), default=0)) if node_count else 0
    lsc = int(max((len(c) for c in nx.strongly_connected_components(dg)), default=0)) if node_count else 0
    non_self_unique = sum(1 for source, target in edge_counts if source != target)
    density = float(non_self_unique / (node_count * (node_count - 1))) if node_count > 1 else 0.0
    atd = float(np.mean([dg.in_degree(node) + dg.out_degree(node) for node in nodes])) if node_count else 0.0

    diameter = np.nan
    asp = np.nan
    cc = np.nan
    if node_count > 1 and ug.number_of_edges() > 0:
        largest = max(nx.connected_components(ug), key=len)
        sub = ug.subgraph(largest).copy()
        if sub.number_of_nodes() > 1:
            diameter = float(nx.diameter(sub))
            asp = float(nx.average_shortest_path_length(sub))
            cc = float(nx.average_clustering(sub))

    adj = _adjacency(edge_counts, nodes)
    no_self = adj.copy()
    if node_count:
        np.fill_diagonal(no_self, 0)
    l1 = int(np.trace(adj)) if node_count else 0
    l2 = int(np.trace(no_self @ no_self) / 2) if node_count else 0
    l3 = int(np.trace(no_self @ no_self @ no_self) / 3) if node_count else 0

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
        "diameter": float(diameter) if np.isfinite(diameter) else np.nan,
        "asp": float(asp) if np.isfinite(asp) else np.nan,
        "cc": float(cc) if np.isfinite(cc) else np.nan,
    }


def compute_metrics_from_segments(segments: Sequence[Sequence[str]]) -> dict[str, float]:
    return compute_metrics(segments)
