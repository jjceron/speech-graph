"""Graph metric computation for speech transcripts.

Matches Java SpeechGraphs.jar logic exactly.
"""

from __future__ import annotations

from typing import Sequence

import networkx as nx
import numpy as np

from .builder import adjacency_matrix, edge_counts, split_by_boundaries

METRICS = [
    "wc", "nodes", "edges", "re", "pe", "l1", "l2", "l3",
    "lcc", "lsc", "atd", "density", "diameter", "asp", "cc",
    ]


def _as_segments(
    tokens_or_segments: Sequence[str] | Sequence[Sequence[str]]
) -> list[list[str]]:
    """Normalize tokens or token segments into a list of segments.

    Accepts either a flat sequence of tokens or an already segmented
    sequence of token groups and returns a normalized representation
    as ``list[list[str]]``.

    Examples:
        >>> _as_segments(["a", "b", "c", "d"])
        [['a', 'b', 'c', 'd']]

        >>> _as_segments([["a", "b"], ["c", "d"]])
        [['a', 'b'], ['c', 'd']]

        >>> _as_segments([])
        []

    Args:
        tokens_or_segments:
            Either:
            - A flat sequence of tokens (``Sequence[str]``), or
            - A sequence of token segments
              (``Sequence[Sequence[str]]``).

    Returns:
        A normalized list of token segments.

        - If the input is a flat token sequence, the result contains a
          single segment with all tokens.
        - If the input is already segmented, each non-empty segment is
          converted to a list and preserved.
        - Empty inputs return an empty list.

    Notes:
        Empty segments in a segmented input are discarded.
    """
    if not tokens_or_segments:
        return []
    first = tokens_or_segments[0]
    if isinstance(first, str):
        return [list(tokens_or_segments)]
    return [list(seg) for seg in tokens_or_segments if len(seg) > 0]


def _empty() -> dict[str, float]:
    return {m: 0.0 for m in METRICS} | {"diameter": float("nan"), "asp": float("nan"), "cc": float("nan")}


def compute_metrics_from_counts(
    nodes: list[str],
    ec: Counter,
    wc: int = 0,
) -> dict[str, float]:
    """Compute all metrics from node list and edge count dict directly.

    Args:
        nodes: Ordered list of unique node labels.
        ec: Edge count dict (source, target) -> count.
        wc: Total word count (for the ``wc`` output key).

    Returns:
        Dict with all 15 metric keys.
    """
    node_count = len(nodes)
    if node_count == 0:
        return _empty() | {"wc": int(wc)}

    edge_total = int(sum(ec.values())) ### Se toman todos los edges (se utiliza el grafo dirigido)
    
    ## Si se tiene ec = Counter({
    #                               ("a", "b"): 2,
    #                               ("b", "c"): 1,
    #                               ("b", "d"): 1
    #                               ("b", "a"): 1
    #                           })
    ## Entonces repeated_edges = 1 porque SOLO se deben tomar edges repetidos respeyando dirección
    repeated_edges = int(sum(count - 1 for count in ec.values() if count > 1))

    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(ec.keys()) ## Se construye el grafo sin que sea Multigrafo --> Solo se cuenta un edge cúan cuando pueden existir varios entre pares
    lsc = int(max((len(c) for c in nx.strongly_connected_components(graph)), default=0))

    # Remove self-loops, keep one edge per undirected pair
    unique_pairs = {frozenset((s, t)) for s, t in ec if s != t}
    unique_undirected_non_self = len(unique_pairs)
    density = float(unique_undirected_non_self / (node_count * (node_count - 1) / 2)) if node_count > 1 else 0.0

    # Undirected graph from post-removal edges
    und = nx.Graph()
    und.add_nodes_from(nodes)
    for pair in unique_pairs:
        und.add_edge(*pair)

    # LCC: Undirected graph
    if und.number_of_edges() > 0:
        components = list(nx.connected_components(und))
        lcc = int(max(len(c) for c in components))
        largest = max(components, key=len)
    else:
        lcc = 0
        largest = set()

    # Diameter, ASP: Java's DistanceStatistics / UnweightedShortestPath on LCC
    diameter = float("nan")
    asp = float("nan")
    if len(largest) > 1:
        component = und.subgraph(largest).copy()
        if component.number_of_nodes() > 1:
            diameter = float(nx.diameter(component))
            asp = float(nx.average_shortest_path_length(component))

    # CC: Java's Metrics.clusteringCoefficients on ALL vertices (und2)
    cc_val = float(nx.average_clustering(und)) if und.number_of_edges() > 0 else float("nan")

    adj = adjacency_matrix(ec, nodes)
    l1 = int(np.trace(adj))
    pe = int(edge_total - l1 - unique_undirected_non_self)
    atd = float(np.mean(adj.sum(axis=0)+adj.sum(axis=1)))

    adj_dir = (adj > 0).astype(int)
    np.fill_diagonal(adj_dir, 0)
    adj_dir2 = adj_dir @ adj_dir
    l2 = int(np.trace(adj_dir2) // 2)
    l3 = int(np.trace(adj_dir2 @ adj_dir) // 3)

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


def compute_metrics(
    tokens_or_segments: Sequence[str] | Sequence[Sequence[str]],
    segment_boundaries: list[bool] | None = None,
) -> dict[str, float]:
    ## Primera opción de input
    #     ["a", "b", "c", "d"],
    #     [False, False, True, False]
    ## Segunda opción de input
    # [
    # ["a", "b"],
    # ["c", "d"]
    # ]
    # No se necesita segment_boundaries

    if segment_boundaries is not None:
        # Volver flatten list. Si es ["a", "b", "c", "d"] se queda igual si entrada es [["a", "b"], ["c", "d"]] entonces pasa a ["a", "b", "c", "d"]
        tokens = list(tokens_or_segments) if not tokens_or_segments or isinstance(tokens_or_segments[0], str) else [t for seg in tokens_or_segments for t in seg]

        segments = split_by_boundaries(tokens, segment_boundaries) ## Entra flatten list y vuelve [['a', 'b'], ['c', 'd']]
    else:
        segments = _as_segments(tokens_or_segments) ## Entra [["a", "b"], ["c", "d"]] | ["a", "b","c", "d"]. Return  [['a', 'b'], ['c', 'd']] | [["a", "b","c", "d"]]
    

    wc = sum(len(seg) for seg in segments)
    if wc == 0:
        return _empty()

    tokens = [t for seg in segments for t in seg]
    nodes = list(dict.fromkeys(tokens)) ## Generar dict sin contar repeticiones y respetando orden de aparción. Si tengo tokens = ["a", "b", "c", "a", "d", "b"] entonces genera {"a": None, "b": None, "c": None, "d": None}
    ec = edge_counts(segments) ## Genera Counts con edges entre pares de nodos. Frecuencia >=1 (multigrafo)
    # Si se tienen segments [["a", "b", "c"], ["a", "b", "c"], ["a", "b", "d"]]
    # Counter({
    #         ("a", "b"): 2,
    #         ("b", "c"): 1,
    #         ("b", "d"): 1
    #     })
    return compute_metrics_from_counts(nodes, ec, wc=wc)
