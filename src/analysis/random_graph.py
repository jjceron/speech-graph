"""Random graph generation by shuffling tokens within segments."""

from __future__ import annotations

import random

from src.graphs.metrics import compute_metrics, METRICS


def shuffle_within_segments(
    tokens: list[str],
    boundaries: list[bool],
    seed: int | None = None,
) -> list[str]:
    """Shuffle tokens within each segment, preserving boundary structure.

    Args:
        tokens: Flat list of tokens.
        boundaries: Boundary flags (True = break before this token).
        seed: Random seed for reproducibility.

    Returns:
        Shuffled token list with same segment structure.
    """
    rng = random.Random(seed)
    segments: list[list[str]] = []
    current: list[str] = []
    for i, token in enumerate(tokens):
        if i > 0 and boundaries[i]:
            if current:
                segments.append(current)
            current = []
        current.append(token)
    if current:
        segments.append(current)

    for seg in segments:
        rng.shuffle(seg)
    return [t for seg in segments for t in seg]


def generate_random_graphs(
    tokens: list[str],
    boundaries: list[bool],
    n_random: int = 100,
    seed: int = 42,
    metrics: list[str] | None = None,
) -> list[dict[str, float]]:
    """Generate n random graphs by shuffling within segments.

    Args:
        tokens: Flat list of tokens.
        boundaries: Boundary flags.
        n_random: Number of random graphs to generate.
        seed: Random seed.
        metrics: Which metrics to compute (default: all).

    Returns:
        List of metric dicts, one per random graph.
    """
    if metrics is None:
        metrics = METRICS[:]
    results: list[dict[str, float]] = []
    for i in range(n_random):
        shuffled = shuffle_within_segments(tokens, boundaries, seed=seed + i)
        m = compute_metrics(shuffled, segment_boundaries=boundaries)
        results.append({k: m[k] for k in metrics if k in m})
    return results


def compute_z_scores(
    original: dict[str, float],
    random_list: list[dict[str, float]],
) -> dict[str, float]:
    """Compute z-scores for each metric.

    Args:
        original: Original metric values.
        random_list: List of random metric dicts.

    Returns:
        Dict of z_{metric} values.
    """
    if not random_list:
        return {}
    metrics = list(random_list[0].keys())
    z: dict[str, float] = {}
    for m in metrics:
        values = [r[m] for r in random_list if m in r and _is_finite(r[m])]
        base = original.get(m, float("nan"))
        if not values or not _is_finite(base):
            z[f"z_{m}"] = float("nan")
            continue
        mean = sum(values) / len(values)
        std = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5
        z_score = 0.0 if std < 1e-10 else (base - mean) / std
        z[f"z_{m}"] = max(-10.0, min(10.0, z_score))
    return z


def _is_finite(v: float) -> bool:
    try:
        import math
        return math.isfinite(v)
    except (TypeError, ValueError):
        return False
