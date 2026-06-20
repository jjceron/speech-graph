"""Tokenization and segmentation of normalized text."""

from __future__ import annotations

import re

from .annotations import BREAK_TOKEN, normalize_annotations_text

_GRAPH_PLACEHOLDER = "___GRAPH_TOKEN___"
_BREAK_PLACEHOLDER = "___BREAK_PLACEHOLDER___"
_WORD_RE = re.compile(
    r"\[\[EE\]\]|[0-9A-Za-z\u00C0-\u024F]+(?:-[0-9A-Za-z\u00C0-\u024F]+)?",
    flags=re.UNICODE,
)


def clean_text(
    text: str,
    lowercase: bool = True,
    normalize: bool = True,
) -> str:
    """Clean raw text for tokenization.

    Steps:
        1. Normalize annotations (optional)
        2. Remove spk_N: labels
        3. Protect [[EE]] and BREAK_TOKEN with placeholders
        4. Remove parenthesized content, ^, <, >
        5. Lowercase (optional)
        6. Strip non-alphanumeric characters
        7. Restore placeholders
    """
    value = str(text or "")
    value = re.sub(r"<\s*(.*?)\s*>", r"\n\1\n", value)
    value = normalize_annotations_text(value) if normalize else value
    value = re.sub(r"\bspk_?\d*\s*:\s*", " ", value, flags=re.IGNORECASE)
    value = value.replace("[[EE]]", f" {_GRAPH_PLACEHOLDER} ")
    value = value.replace(BREAK_TOKEN, f" {_BREAK_PLACEHOLDER} ")
    value = re.sub(r"\(([^)]*)\)", r"\1", value)
    value = value.replace("^", "")
    value = value.replace("<", " ").replace(">", " ")
    value = value.replace("'", "")
    if lowercase:
        value = value.lower()
    value = re.sub(r"[^0-9A-Za-z\u00C0-\u024F_]+", " ", value, flags=re.UNICODE)
    value = re.sub(r"\s+", " ", value).strip()
    value = value.replace(_GRAPH_PLACEHOLDER.lower(), "[[EE]]")
    value = value.replace(_GRAPH_PLACEHOLDER, "[[EE]]")
    value = value.replace(_BREAK_PLACEHOLDER.lower(), BREAK_TOKEN)
    value = value.replace(_BREAK_PLACEHOLDER, BREAK_TOKEN)
    return value


def tokenize_segments(
    text: str,
    lowercase: bool = True,
    normalize: bool = True,
    return_segment_map: bool = False,
) -> list[list[str]] | tuple[list[list[str]], list[int]]:
    """Tokenize text into segments split by BREAK_TOKEN.

    Returns:
        If return_segment_map=False: list of segments (each a list of tokens)
        If return_segment_map=True:  (segments, segment_ids) where segment_ids[i]
            is the segment index for token i.
    """
    cleaned = clean_text(text, lowercase=lowercase, normalize=normalize)
    if not cleaned:
        return ([], []) if return_segment_map else []

    segments: list[list[str]] = []
    for part in cleaned.split(BREAK_TOKEN):
        part = part.strip()
        if len(part) <= 1:
            continue
        tokens = [m.group(0) for m in _WORD_RE.finditer(part)]
        if tokens:
            segments.append(tokens)

    if not return_segment_map:
        return segments

    segment_ids: list[int] = []
    for seg_idx, segment in enumerate(segments):
        segment_ids.extend([seg_idx] * len(segment))
    return segments, segment_ids


def tokenize(
    text: str,
    lowercase: bool = True,
    normalize: bool = True,
) -> list[str]:
    """Flatten all segments into a single list of tokens."""
    return [
        token
        for segment in tokenize_segments(text, lowercase, normalize)
        for token in segment
    ]
