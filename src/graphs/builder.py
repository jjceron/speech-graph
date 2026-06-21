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


def adjacency_matrix(
    edge_counts: Counter[tuple[str, str]],
    nodes: list[str],
) -> np.ndarray:
    """Build a directed adjacency matrix from edge frequency counts.

    Creates a square adjacency matrix where rows represent source nodes
    and columns represent target nodes. Each cell ``(i, j)`` contains
    the total number of occurrences of the directed edge
    ``nodes[i] -> nodes[j]``.

    Example:
        >>> nodes = ["a", "b", "c"]
        >>> edge_counts = Counter({
        ...     ("a", "b"): 2,
        ...     ("b", "c"): 1,
        ...     ("c", "a"): 1,
        ... })
        >>> adjacency_matrix(edge_counts, nodes)
        array([
            [0, 2, 0],
            [0, 0, 1],
            [1, 0, 0]
        ])

    Args:
        edge_counts:
            Mapping from directed edge ``(source, target)`` to its
            occurrence count.
        nodes:
            Ordered list of node labels defining the row and column
            ordering of the resulting matrix.

    Returns:
        A square NumPy array of shape ``(n, n)``, where ``n`` is the
        number of nodes. Matrix entries contain edge frequencies rather
        than binary connectivity values.

    Notes:
        - Self-loops are stored on the matrix diagonal.
        - Multiple occurrences of the same edge are accumulated in the
          corresponding matrix cell.
        - The matrix represents a directed graph and is therefore not
          necessarily symmetric.
    """
    index = {node: i for i, node in enumerate(nodes)}
    matrix = np.zeros((len(nodes), len(nodes)), dtype=int)
    for (source, target), count in edge_counts.items():
        matrix[index[source], index[target]] += int(count)
    return matrix

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

