from __future__ import annotations

from collections import Counter
from typing import Iterable

from src.preprocessing.annotations import canonical_label, extract_double_bracket_labels, safe_label_name


def extract_discourse_labels(tokens: Iterable[str]) -> list[str]:
    """Extract canonical labels enclosed in double brackets from tokenized transcript text."""
    return extract_double_bracket_labels(tokens)


def label_features(tokens: Iterable[str]) -> dict[str, float]:
    """Count and normalize [[...]] labels for a transcript, activity or segment.

    The denominator for ratios is the total number of tokens, so labels are measured
    as part of the discourse stream used to build the word graph. Timestamped and
    typo variants are collapsed to protocol categories.
    """
    token_list = list(tokens)
    n_tokens = len(token_list)
    labels = [canonical_label(label) for label in extract_discourse_labels(token_list)]
    counts = Counter(safe_label_name(label) for label in labels)

    features: dict[str, float] = {
        "label_total_count": int(len(labels)),
        "label_total_ratio": (len(labels) / n_tokens) if n_tokens else 0.0,
        "label_unique_count": int(len(counts)),
    }
    for label, count in sorted(counts.items()):
        features[f"label_count_{label}"] = int(count)
        features[f"label_ratio_{label}"] = (count / n_tokens) if n_tokens else 0.0
    return features
