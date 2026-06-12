"""Random SRL graph generation with multiple null models.

Available models:
    - shuffle: Shuffle target nodes of semantic relations while preserving
      the multiset of sources and targets.
    - erdos_renyi: Erdős–Rényi model preserving the number of nodes and
      total edge count but not the degree sequence.
"""

from __future__ import annotations

import math
import random

from src.graphs.srl_graphs import build_srl_graph
from src.graphs.srl_metrics import compute_srl_metrics, compute_srl_metrics_weighted
from src.analysis.random_graph import compute_z_scores


def shuffle_srl_relations(
    relations: dict[str, int],
    seed: int | None = None,
) -> dict[str, int]:
    """Shuffle target nodes of SRL relations, preserving source multiset.

    Takes a dict of ``"source--target" -> weight``, expands to individual
    (source, target) pairs, permutes the targets independently, and
    re-aggregates by summing weights for duplicate source--target pairs.

    Args:
        relations: Original relation counts (``"A--B" -> count``).
        seed: Random seed for reproducibility.

    Returns:
        New relation dict with same total weight but permuted targets.
    """
    if not relations:
        return {}

    rng = random.Random(seed)
    sources: list[str] = []
    targets: list[str] = []
    for key, weight in relations.items():
        source, target = key.split("--", 1)
        for _ in range(weight):
            sources.append(source)
            targets.append(target)

    rng.shuffle(targets)

    new_relations: dict[str, int] = {}
    for s, t in zip(sources, targets):
        key = f"{s}--{t}"
        new_relations[key] = new_relations.get(key, 0) + 1
    return new_relations


def generate_random_srl_graphs(
    relations: dict[str, int],
    window_size: int,
    n_random: int = 100,
    seed: int = 42,
) -> list[dict[str, float]]:
    """Generate random SRL graphs by shuffling relation targets.

    Args:
        relations: Original aggregated relation counts.
        window_size: Number of sentences in the window (used for ``wc``).
        n_random: Number of random graphs to generate.
        seed: Random seed.

    Returns:
        List of metric dicts, one per random graph.
    """
    if not relations:
        return _empty_graphs(window_size, n_random, metrics_func=compute_srl_metrics_weighted)

    results: list[dict[str, float]] = []
    for i in range(n_random):
        shuffled = shuffle_srl_relations(relations, seed=seed + i)
        G = build_srl_graph(shuffled)
        m = compute_srl_metrics_weighted(G, window_size=window_size)
        results.append(m)
    return results


def _extract_er_parameters(
    relations: dict[str, int],
) -> tuple[list[str], int]:
    """Extract node list and total edge count from a relation dict."""
    nodes_set: set[str] = set()
    total_edges = 0
    for key, weight in relations.items():
        source, target = key.split("--", 1)
        nodes_set.add(source)
        nodes_set.add(target)
        total_edges += weight
    return sorted(nodes_set), total_edges


def _erdos_renyi_trial(
    nodes: list[str],
    n_edges: int,
    rng: random.Random,
) -> dict[str, int]:
    """Generate one Erdős–Rényi graph trial.

    Uniformly samples ``n_edges`` directed pairs (source, target) from
    the set of all N² possible pairs without replacement.

    Args:
        nodes: List of node labels.
        n_edges: Number of directed edges to place.
        rng: Seeded random generator.

    Returns:
        Relation dict with unit weights, one entry per sampled pair.
    """
    n = len(nodes)
    if n == 0 or n_edges == 0:
        return {}

    # Build index of all N² possible directed pairs
    possible = [(i, j) for i in range(n) for j in range(n)]

    # Clamp n_edges to the maximum possible
    k = min(n_edges, len(possible))

    sampled = rng.sample(possible, k)

    relations: dict[str, int] = {}
    for i, j in sampled:
        key = f"{nodes[i]}--{nodes[j]}"
        relations[key] = relations.get(key, 0) + 1
    return relations


def generate_random_er_graphs(
    relations: dict[str, int],
    window_size: int,
    n_random: int = 100,
    seed: int = 42,
) -> list[dict[str, float]]:
    """Generate random SRL graphs using the Erdős–Rényi model.

    Each random graph preserves the number of nodes and total edge count
    of the original ``relations``, but edges are placed uniformly at
    random among all N² possible directed pairs (including self-loops),
    sampled without replacement.

    This differs from :func:`generate_random_srl_graphs` in that it does
    **not** preserve the multiset of source/target frequencies.

    Args:
        relations: Original aggregated relation counts.
        window_size: Number of sentences in the window (used for ``wc``).
        n_random: Number of random graphs to generate.
        seed: Random seed.

    Returns:
        List of metric dicts, one per random graph.
    """
    nodes, n_edges = _extract_er_parameters(relations)

    if not nodes or n_edges == 0:
        return _empty_graphs(window_size, n_random, metrics_func=compute_srl_metrics)

    results: list[dict[str, float]] = []
    for i in range(n_random):
        rng = random.Random(seed + i)
        trial_rels = _erdos_renyi_trial(nodes, n_edges, rng)
        G = build_srl_graph(trial_rels)
        m = compute_srl_metrics(G, window_size=window_size)
        results.append(m)
    return results


def _empty_graphs(
    window_size: int,
    n_random: int,
    metrics_func=compute_srl_metrics,
) -> list[dict[str, float]]:
    """Return ``n_random`` empty metric dicts.

    Args:
        metrics_func: Metric function to use (default: ``compute_srl_metrics``).
    """
    empty = metrics_func(build_srl_graph({}), window_size=window_size)
    return [dict(empty) for _ in range(n_random)]
