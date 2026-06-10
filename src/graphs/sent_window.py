"""Sentence-level sliding windows for SRL results."""

from __future__ import annotations

from collections.abc import Generator


def sentence_windows(
    sentences: list,
    window_size: int = 3,
    step: int = 1,
) -> Generator[tuple[int, int, int, list], None, None]:
    """Iterate over overlapping windows of consecutive sentences.

    Args:
        sentences: List of sentence-level SRL results.
        window_size: Number of sentences per window (default 3).
        step: Step size between windows (default 1).

    Yields:
        (window_id, start_idx, end_idx, window_sentences)
    """
    n = len(sentences)
    if n < window_size:
        return

    for start in range(0, n - window_size + 1, step):
        end = start + window_size
        window_id = start // step
        yield window_id, start, end, sentences[start:end]
