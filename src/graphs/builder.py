"""Graph building utilities: edge counting, adjacency matrix, DiGraph construction."""

from __future__ import annotations

from collections import Counter

import networkx as nx
import numpy as np


def edge_counts(segments: list[list[str]]) -> Counter[tuple[str, str]]:
    """Count directed token transitions across all segments.

    For each segment, consecutive token pairs are treated as directed
    edges. The function returns the frequency of each edge across all
    segments.

    Examples:
        >>> edge_counts([["a", "b", "c"]])
        Counter({
            ("a", "b"): 1,
            ("b", "c"): 1
        })

        >>> edge_counts([
        ...     ["a", "b", "c"],
        ...     ["a", "b", "d"]
        ... ])
        Counter({
            ("a", "b"): 2,
            ("b", "c"): 1,
            ("b", "d"): 1
        })

    Args:
        segments: List of token segments. Each segment is processed
            independently, and edges are created only between consecutive
            tokens within the same segment.

    Returns:
        A Counter mapping each directed edge ``(source_token,
        target_token)`` to its occurrence count.

    Notes:
        Segments containing fewer than two tokens do not contribute any
        edges. --- Nodo aislado no tiene edges
    """
    counts: Counter[tuple[str, str]] = Counter()
    for segment in segments:
        ## Si segment = ["a", "b", "c", "d"]
        ## segment[:-1] = ["a", "b", "c"]
        ## segment[1:] = ["b", "c", "d"]
        ## Entonces zip(segment[:-1], segment[1:]) --> Counter({("a", "b"), ("b", "c"), ("c", "d")})
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


def split_by_boundaries(tokens: list[str],
                        boundaries: list[bool]) -> list[list[str]]:
    """Split a sequence of tokens into segments based on boundary markers.

    A new segment starts whenever ``boundaries[i]`` is ``True`` for a token
    position ``i > 0``. The token at position ``i`` becomes the first element
    of the new segment.

    Example:
        >>> split_by_boundaries(
        ...     ["a", "b", "c", "d"],
        ...     [False, False, True, False]
        ... )
        [['a', 'b'], ['c', 'd']]

    Args:
        tokens: Ordered list of tokens to be segmented.
        boundaries: Boolean flags indicating segment boundaries. A value of
            ``True`` at position ``i`` indicates that a new segment begins
            before ``tokens[i]``.

    Returns:
        A list of token segments. Each segment is represented as a list of
        strings.

    Notes:
        - ``tokens`` and ``boundaries`` are expected to have the same length.
        - A boundary at index ``0`` is ignored because there is no preceding
          segment to split.
    """
    segments: list[list[str]] = []
    current: list[str] = []
    for i, token in enumerate(tokens):
        if i > 0 and boundaries[i]:
            segments.append(current)
            current = []
        current.append(token)
    if current:
        segments.append(current)
    return segments

