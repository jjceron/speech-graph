"""Comparison of original vs random graphs with z-scores."""

from __future__ import annotations

from src.graphs.metrics import compute_metrics, METRICS
from .random_graph import generate_random_graphs, compute_z_scores


def compare_original_vs_random(
    tokens: list[str],
    boundaries: list[bool],
    n_random: int = 100,
    seed: int = 42,
    metrics: list[str] | None = None,
) -> dict:
    """Compare original graph metrics against random baseline.

    Args:
        tokens: Flat list of tokens.
        boundaries: Boundary flags.
        n_random: Number of random graphs.
        seed: Random seed.
        metrics: Which metrics to compare (default: all).

    Returns:
        Dict with 'original', 'random_list', 'random_mean', 'z_scores', 'changes'.
    """
    if metrics is None:
        metrics = METRICS[:]

    original = compute_metrics(tokens, segment_boundaries=boundaries)
    original_subset = {k: original[k] for k in metrics if k in original}

    random_list = generate_random_graphs(tokens, boundaries, n_random, seed, metrics)

    random_mean: dict[str, float] = {}
    random_std: dict[str, float] = {}
    for m in metrics:
        values = [r[m] for r in random_list if m in r]
        if values:
            mean = sum(values) / len(values)
            std = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5
            random_mean[m] = mean
            random_std[m] = std
        else:
            random_mean[m] = float("nan")
            random_std[m] = float("nan")

    z_scores = compute_z_scores(original_subset, random_list)

    changes: dict[str, bool] = {}
    for m in metrics:
        if m in original_subset and m in random_mean:
            changes[m] = abs(original_subset[m] - random_mean[m]) > 0.01

    return {
        "original": original_subset,
        "random_mean": random_mean,
        "random_std": random_std,
        "random_list": random_list,
        "z_scores": z_scores,
        "changes": changes,
    }
