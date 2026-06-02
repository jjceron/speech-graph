from __future__ import annotations

from typing import Iterable, Iterator, List, Tuple


def sliding_windows(
    tokens: List[str],
    window_size: int,
    step: int,
    allow_short: bool = True,
) -> Iterator[Tuple[List[str], int, int]]:
    if window_size <= 0:
        raise ValueError("window_size must be greater than zero")
    if step <= 0:
        raise ValueError("step must be greater than zero")

    n = len(tokens)
    if n == 0:
        return

    if n < window_size:
        if allow_short:
            yield tokens, 0, n
        return

    start = 0
    while start + window_size <= n:
        end = start + window_size
        yield tokens[start:end], start, end
        start += step
