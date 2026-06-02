from __future__ import annotations

from collections import Counter
from typing import Dict, List

import networkx as nx
import numpy as np


def adjacency_counts(tokens: List[str]) -> tuple[np.ndarray, List[str]]:
    if not tokens:
        return np.zeros((0, 0), dtype=int), []

    node_list = list(dict.fromkeys(tokens))
    index = {node: i for i, node in enumerate(node_list)}
    matrix = np.zeros((len(node_list), len(node_list)), dtype=int)

    for src, dst in zip(tokens[:-1], tokens[1:]):
        matrix[index[src], index[dst]] += 1

    return matrix, node_list


def empty_metrics() -> Dict[str, float]:
    return {
        "token_count": 0,
        "nodes": 0,
        "edges": 0,
        "unique_edges": 0,
        "repeated_edges": 0,
        "repeated_edges_ratio": 0.0,
        "lcc": 0,
        "lsc": 0,
        "lcc_ratio": 0.0,
        "lsc_ratio": 0.0,
        "density": 0.0,
        "atd": 0.0,
        "diameter": float("nan"),
        "asp": float("nan"),
        "clustering": float("nan"),
        "l1": 0.0,
        "l2": 0.0,
        "l3": 0.0,
    }


def compute_metrics(tokens: List[str]) -> Dict[str, float]:
    token_count = len(tokens)
    if token_count == 0:
        return empty_metrics()

    edge_pairs = list(zip(tokens[:-1], tokens[1:]))
    edge_counts = Counter(edge_pairs)
    edges = len(edge_pairs)
    unique_edges = len(edge_counts)
    repeated_edges = sum(count - 1 for count in edge_counts.values() if count > 1)

    adj, nodes = adjacency_counts(tokens)
    node_count = len(nodes)

    dg = nx.DiGraph()
    dg.add_nodes_from(nodes)
    dg.add_edges_from(edge_counts.keys())

    ug = dg.to_undirected()
    edges_und = ug.number_of_edges()

    if node_count > 0:
        lcc = max((len(c) for c in nx.connected_components(ug)), default=0)
        lsc = max((len(c) for c in nx.strongly_connected_components(dg)), default=0)
    else:
        lcc = 0
        lsc = 0

    lcc_ratio = lcc / node_count if node_count else 0.0 # Revisar
    lsc_ratio = lsc / node_count if node_count else 0.0 # Revisar

    density = (2.0 * edges_und) / (node_count * (node_count - 1)) if node_count > 1 else 0.0
    atd = (2.0 * edges_und) / node_count if node_count else 0.0 # Revisar

    diameter = float("nan")
    asp = float("nan")
    clustering = float("nan")

    if node_count > 1 and edges_und > 0:
        largest_nodes = max(nx.connected_components(ug), key=len)
        sub = ug.subgraph(largest_nodes)
        if sub.number_of_nodes() > 1:
            diameter = float(nx.diameter(sub))
            asp = float(nx.average_shortest_path_length(sub)) # Revisar
            clustering = float(nx.average_clustering(sub)) # Revisar

    l1 = float(np.trace(adj)) if node_count else 0.0
    l2 = float(np.trace(adj @ adj) / 2.0) if node_count else 0.0
    l3 = float(np.trace(adj @ adj @ adj) / 3.0) if node_count else 0.0

    return {
        "token_count": token_count,
        "nodes": node_count,
        "edges": edges,
        "unique_edges": unique_edges,
        "repeated_edges": repeated_edges,
        # "repeated_edges_ratio": repeated_edges / edges if edges else 0.0,
        "lcc": lcc,
        "lsc": lsc,
        "lcc_ratio": lcc_ratio,
        "lsc_ratio": lsc_ratio,
        "density": density,
        "atd": atd,
        "diameter": diameter,
        "asp": asp,
        "clustering": clustering,
        "l1": l1,
        "l2": l2,
        "l3": l3,
    }
