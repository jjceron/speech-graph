"""Graph building utilities: edge counting, adjacency matrix, DiGraph construction."""

from __future__ import annotations

from collections import Counter

import networkx as nx
import numpy as np


def edge_counts(segments: list[list[str]]) -> Counter[tuple[str, str]]:
    """Count directed edges between consecutive tokens in each segment."""
    counts: Counter[tuple[str, str]] = Counter()
    for segment in segments:
        counts.update(zip(segment[:-1], segment[1:]))
    return counts


def adjacency_matrix(edge_counts: Counter[tuple[str, str]], nodes: list[str]) -> np.ndarray:
    """Build a square adjacency matrix from edge counts."""
    index = {node: i for i, node in enumerate(nodes)}
    matrix = np.zeros((len(nodes), len(nodes)), dtype=int)
    for (source, target), count in edge_counts.items():
        matrix[index[source], index[target]] += int(count)
    return matrix


def parallel_edges(edge_counts: Counter[tuple[str, str]]) -> int:
    """Count parallel edges (pairs with edges in both directions)."""
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


def build_graph(segments: list[list[str]]) -> nx.DiGraph:
    """Build a directed graph from token segments."""
    counts = edge_counts(segments)
    graph = nx.DiGraph()
    for (source, target), weight in counts.items():
        graph.add_edge(source, target, weight=weight)
    return graph


def split_by_boundaries(tokens: list[str], boundaries: list[bool]) -> list[list[str]]:
    """Split a flat token list into segments using boundary flags.

    ``boundaries[i]`` is True when a segment break occurs before token *i*.
    """
    segments: list[list[str]] = []
    current: list[str] = []
    for i, token in enumerate(tokens): ###
        # if i > 0 and boundaries[i] != boundaries[i - 1]:
        if i > 0 and boundaries[i]:
            # if current:
            segments.append(current)
            current = []
        current.append(token) ### 
    if current:
        segments.append(current)
    return segments
