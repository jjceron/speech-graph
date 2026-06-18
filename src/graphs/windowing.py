"""Sliding window iteration with segment boundary support."""

from __future__ import annotations

from collections.abc import Iterator



def sliding_windows(
    tokens: list[str],
    window_size: int,
    step: int,
    allow_short: bool = False, ### Contabilizar ventanas que tienen menos de window_size
    segment_boundaries: list[bool] | None = None,
) -> Iterator[tuple[list[str], int, int] | tuple[list[str], int, int, list[bool]]]:
    """
    Generate sliding windows over a sequence of tokens.

    This generator iterates through a flat token sequence and yields
    consecutive windows of fixed size. Windows can overlap depending on
    the value of ``step``. Optionally, the function can also return
    information about segment transitions occurring within each window.

    Processing behavior:
        1. Validates that ``window_size`` and ``step`` are greater than zero.
        2. Returns immediately if the token list is empty.
        3. If the number of tokens is smaller than ``window_size``:
            - Returns a single short window when ``allow_short=True``.
            - Returns no windows when ``allow_short=False``.
        4. Iteratively generates windows of length ``window_size``,
           advancing by ``step`` tokens each time.
        5. If ``segment_boundaries`` is provided, computes window-level
           segment transition flags using ``_window_boundaries``.

    Args:
        tokens:
            Flat list of tokens to process.
        window_size:
            Number of tokens contained in each window.
        step:
            Number of tokens by which the window advances after each
            iteration.
        allow_short:
            If True and the token sequence is shorter than
            ``window_size``, yields a single short window containing all
            available tokens. If False, no window is generated.
        segment_boundaries:
            Optional segment membership information aligned with the full
            token sequence. When provided, each yielded window includes a
            corresponding list of boundary flags indicating where segment
            transitions occur within the window.

    Yields:
        If ``segment_boundaries`` is None:

            (
                window_tokens,
                start,
                end
            )

            where:
                - window_tokens: tokens contained in the window
                - start: start index (inclusive) in the full token list
                - end: end index (exclusive) in the full token list

        If ``segment_boundaries`` is provided:

            (
                window_tokens,
                start,
                end,
                window_boundaries
            )

            where:
                - window_boundaries[i] is True when a segment transition
                  occurs immediately before the i-th token of the window

    Raises:
        ValueError:
            If ``window_size <= 0``.
        ValueError:
            If ``step <= 0``.

    Examples:
        >>> list(sliding_windows(
        ...     ["a", "b", "c", "d", "e"],
        ...     window_size=3,
        ...     step=1
        ... ))
        [
            (["a", "b", "c"], 0, 3),
            (["b", "c", "d"], 1, 4),
            (["c", "d", "e"], 2, 5)
        ]

        >>> list(sliding_windows(
        ...     ["a", "b"],
        ...     window_size=5,
        ...     step=1,
        ...     allow_short=True
        ... ))
        [
            (["a", "b"], 0, 2)
        ]

        >>> list(sliding_windows(
        ...     ["a", "b", "c", "d"],
        ...     window_size=4,
        ...     step=1,
        ...     segment_boundaries=[0, 0, 1, 1]
        ... ))
        [
            (
                ["a", "b", "c", "d"],
                0,
                4,
                [False, False, True, False]
            )
        ]
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
                yield tokens, 0, n, _window_boundaries(segment_boundaries, 0, n) ### Compute segment transition flags within a token window.
            else:
                yield tokens, 0, n ### (["hola", "mundo"], 0, 2)
        return
    start = 0
    while start + window_size <= n:
        end = start + window_size
        if segment_boundaries is not None:
            yield tokens[start:end], start, end, _window_boundaries(segment_boundaries, start, end)
            ### _window_boundaries Compute segment transition flags within a token window.
            # segment_boundaries = [0, 0, 1, 1]
            # start = 0
            # end = 4
            # Result:
            # [False, False, True, False]
            ### (["hola", "mundo"], 0, 2, [False,False])

        else:
            yield tokens[start:end], start, end
        start += step


def _window_boundaries(segment_boundaries: list[bool], start: int, end: int) -> list[bool]:
    """
    Compute segment transition flags within a token window.

    This function identifies where segment boundaries (i.e., changes in
    discourse segments) occur inside a sliding window of tokens. It compares
    consecutive values in the global `segment_boundaries` list and marks a
    transition whenever the segment assignment changes between adjacent tokens.

    The returned list has the same length as the window (`end - start`) and
    indicates whether a segment break occurs before each token in the window:

        - boundaries[0] is always False (no previous token in the window)
        - boundaries[i] is True if there is a segment change between
          tokens at positions (start + i - 1) and (start + i)

    Args:
        segment_boundaries: A list of boolean or segment identifiers aligned
            with the full token sequence, indicating segment membership per token.
        start: Start index of the window (inclusive).
        end: End index of the window (exclusive).

    Returns:
        A list of boolean values of length (end - start), where each value
        indicates whether a segment boundary occurs before the corresponding
        token in the window.

    Example:
        segment_boundaries = [0, 0, 1, 1]
        start = 0
        end = 4

        Result:
        [False, False, True, False]
    """
    boundaries: list[bool] = [False]
    for pos in range(start + 1, end):
        boundaries.append(segment_boundaries[pos] != segment_boundaries[pos - 1])
    return boundaries
