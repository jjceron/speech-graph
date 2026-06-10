"""Build NetworkX graphs from aggregated SRL relations."""

from __future__ import annotations

from collections import defaultdict

import networkx as nx


def aggregate_relations(
    sentence_results: list[dict],
    rel_key: str = "semantic_relations",
) -> dict[str, int]:
    """Sum relation counts across a list of sentence results.

    Args:
        sentence_results: List of per-sentence SRL result dicts.
        rel_key: Which relation type to aggregate
                  ("ap_relations", "pa_relations", "semantic_relations").

    Returns:
        Dict mapping "source--target" -> total count.
    """
    combined: dict[str, int] = defaultdict(int)
    for entry in sentence_results:
        for k, v in entry.get(rel_key, {}).items():
            combined[k] += v
    return dict(combined)


def build_srl_graph(relations: dict[str, int]) -> nx.DiGraph:
    """Build a directed weighted graph from relation dictionaries.

    Args:
        relations: Dict mapping "source--target" -> weight.

    Returns:
        NetworkX DiGraph with edge attribute ``weight``.
    """
    G = nx.DiGraph()
    for edge_key, weight in relations.items():
        source, target = edge_key.split("--", 1)
        G.add_edge(source, target, weight=weight)
    return G
