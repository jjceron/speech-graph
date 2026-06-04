"""Sliding window iteration with segment boundary support."""

from __future__ import annotations

from collections.abc import Iterator


def sliding_windows(
    tokens: list[str],
    window_size: int,
    step: int,
    allow_short: bool = False,
    segment_boundaries: list[bool] | None = None,
) -> Iterator[tuple[list[str], int, int] | tuple[list[str], int, int, list[bool]]]:
    """Yield sliding windows over a token list.

    Args:
        tokens: Flat list of tokens.
        window_size: Number of tokens per window.
        step: Number of tokens to advance between windows.
        allow_short: If True, yield a short window when tokens < window_size.
        segment_boundaries: Optional boundary flags for the full token list.

    Yields:
        (window_tokens, start, end) or (window_tokens, start, end, window_boundaries)
    """
    if window_size <= 0:
        raise ValueError("window_size must be greater than zero")
    if step <= 0:
        raise ValueError("step must be greater than zero")
    n = len(tokens)
    if n == 0:
        return
    if n < window_size:
        if allow_short:
            if segment_boundaries is not None:
                yield tokens, 0, n, _window_boundaries(segment_boundaries, 0, n)
            else:
                yield tokens, 0, n
        return
    start = 0
    while start + window_size <= n:
        end = start + window_size
        if segment_boundaries is not None:
            yield tokens[start:end], start, end, _window_boundaries(segment_boundaries, start, end)
        else:
            yield tokens[start:end], start, end
        start += step


def _window_boundaries(segment_boundaries: list[bool], start: int, end: int) -> list[bool]:
    """Return boundary flags for tokens within a window.

    ``boundaries[i]`` is True when there is a segment break before the i-th
    token of the window (i.e. between position ``start + i - 1`` and
    ``start + i`` in the full token list).
    """
    boundaries: list[bool] = [False]
    for pos in range(start + 1, end):
        boundaries.append(segment_boundaries[pos] != segment_boundaries[pos - 1])
    return boundaries
