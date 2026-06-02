from __future__ import annotations

from collections import Counter
from typing import Sequence

import networkx as nx
import numpy as np

MODEL_METRICS = [
    "wc", "nodes", "edges", "re", "pe", "l1", "l2", "l3",
    "lcc", "lsc", "atd", "density", "diameter", "asp", "cc",
]

OUTPUT_METRICS = MODEL_METRICS + ["parallel_edges"]


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
        matrix[index[source], index[target]] += int(count)
    return matrix


def _parallel_edges(edge_counts: Counter[tuple[str, str]]) -> int:
    total = 0
    visited: set[frozenset[str]] = set()
    for source, target in edge_counts:
        if source == target:
            continue
        pair = frozenset((source, target))
        if pair in visited:
            continue
        total += min(edge_counts.get((source, target), 0), edge_counts.get((target, source), 0))
        visited.add(pair)
    return int(total)


def _empty() -> dict[str, float]:
    return {metric: 0.0 for metric in OUTPUT_METRICS} | {"diameter": float("nan"), "asp": float("nan"), "cc": float("nan")}


def compute_metrics(tokens_or_segments: Sequence[str] | Sequence[Sequence[str]]) -> dict[str, float]:
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
    parallel_edges = _parallel_edges(edge_counts)

    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edge_counts.keys())
    undirected = graph.to_undirected()

    lcc = int(max((len(c) for c in nx.connected_components(undirected)), default=0)) if node_count else 0
    lsc = int(max((len(c) for c in nx.strongly_connected_components(graph)), default=0)) if node_count else 0
    atd = float(np.mean([graph.in_degree(node) + graph.out_degree(node) for node in nodes])) if node_count else 0.0

    non_self_edges = sum(1 for source, target in edge_counts if source != target)
    density = float(non_self_edges / (node_count * (node_count - 1))) if node_count > 1 else 0.0

    diameter = float("nan")
    asp = float("nan")
    cc = float("nan")
    if node_count > 1 and undirected.number_of_edges() > 0:
        largest = max(nx.connected_components(undirected), key=len)
        component = undirected.subgraph(largest).copy()
        if component.number_of_nodes() > 1:
            diameter = float(nx.diameter(component))
            asp = float(nx.average_shortest_path_length(component))
            cc = float(nx.average_clustering(component))

    adjacency = _adjacency(edge_counts, nodes)
    no_self = adjacency.copy()
    if node_count:
        np.fill_diagonal(no_self, 0)
    l1 = int(np.trace(adjacency)) if node_count else 0
    l2 = int(np.trace(no_self @ no_self) / 2) if node_count else 0
    l3 = int(np.trace(no_self @ no_self @ no_self) / 3) if node_count else 0

    return {
        "wc": int(wc),
        "nodes": int(node_count),
        "edges": int(edge_total),
        "re": int(repeated_edges),
        "pe": int(parallel_edges),
        "parallel_edges": int(parallel_edges),
        "l1": int(l1),
        "l2": int(l2),
        "l3": int(l3),
        "lcc": int(lcc),
        "lsc": int(lsc),
        "atd": float(atd),
        "density": float(density),
        "diameter": diameter,
        "asp": asp,
        "cc": cc,
    }


def compute_metrics_from_segments(segments: Sequence[Sequence[str]]) -> dict[str, float]:
    return compute_metrics(segments)
