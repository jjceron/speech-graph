"""Tokenization and segmentation of normalized text."""

from __future__ import annotations

import re
from collections.abc import Collection
from functools import lru_cache

import stanza

from .annotations import BREAK_TOKEN, normalize_annotations_text

_GRAPH_PLACEHOLDER = "___GRAPH_TOKEN___"
_BREAK_PLACEHOLDER = "___BREAK_PLACEHOLDER___"
_WORD_RE = re.compile(
    r"\[\[EE\]\]|[0-9A-Za-z\u00C0-\u024F]+(?:-[0-9A-Za-z\u00C0-\u024F]+)?",
    flags=re.UNICODE,
)

_POS_KEEP_DEFAULT = frozenset({"NOUN", "PRON", "VERB", "ADJ", "ADV"})


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


def clean_text_all(
    text: str,
    lowercase: bool = True,
    normalize: bool = True,
) -> str:
    """Clean raw text for tokenization, additionally removing [[EE]].

    Same as clean_text but [[EE]] is removed instead of preserved.
    """
    value = normalize_annotations_text(text) if normalize else str(text or "")
    value = re.sub(r"\bspk_?\d*\s*:\s*", " ", value, flags=re.IGNORECASE)
    value = re.sub(r"\[\[EE\]\]", " ", value)
    value = value.replace(BREAK_TOKEN, f" {_BREAK_PLACEHOLDER} ")
    value = re.sub(r"\(([^)]*)\)", r"\1", value)
    value = value.replace("^", "")
    value = value.replace("<", " ").replace(">", " ")
    value = value.replace("'", "")
    if lowercase:
        value = value.lower()
    value = re.sub(r"[^0-9A-Za-z\u00C0-\u024F_]+", " ", value, flags=re.UNICODE)
    value = re.sub(r"\s+", " ", value).strip()
    value = value.replace(_BREAK_PLACEHOLDER.lower(), BREAK_TOKEN)
    value = value.replace(_BREAK_PLACEHOLDER, BREAK_TOKEN)
    return value


@lru_cache(maxsize=2)
def _get_stanza_pipeline(lang: str = "es") -> stanza.Pipeline:
    return stanza.Pipeline(
        lang=lang, processors="tokenize,pos", verbose=False, use_gpu=False,
    )


def filter_pos(
    tokens: list[str],
    lang: str = "es",
    keep_tags: Collection[str] | None = None,
) -> list[str]:
    """Filter tokens keeping only those with desired POS tags using stanza."""
    if keep_tags is None:
        keep_tags = _POS_KEEP_DEFAULT
    if not tokens:
        return tokens

    nlp = _get_stanza_pipeline(lang)
    text = " ".join(tokens)
    doc = nlp(text)

    stanza_words: list[tuple[str, str]] = []
    for sent in doc.sentences:
        for word in sent.words:
            stanza_words.append((word.text, word.upos))

    if len(stanza_words) != len(tokens):
        return tokens

    return [
        token
        for token, (_, upos) in zip(tokens, stanza_words)
        if upos in keep_tags
    ]


def tokenize_segments(
    text: str,
    lowercase: bool = True,
    normalize: bool = True,
    return_segment_map: bool = False,
    clean_func: str = "clean_text",
    pos_filter: bool = False,
    pos_lang: str = "es",
) -> list[list[str]] | tuple[list[list[str]], list[int]]:
    """Tokenize text into segments split by BREAK_TOKEN.

    Args:
        text: Raw transcript text.
        lowercase: Lowercase tokens.
        normalize: Apply annotation normalization.
        return_segment_map: Also return segment index per token.
        clean_func: Cleaning function - "clean_text" or "clean_text_all".
        pos_filter: Apply POS tag filtering via stanza.
        pos_lang: Language for stanza POS tagging.

    Returns:
        If return_segment_map=False: list of segments (each a list of tokens)
        If return_segment_map=True:  (segments, segment_map)
    """
    if clean_func == "clean_text_all":
        cleaned = clean_text_all(text, lowercase=lowercase, normalize=normalize)
    else:
        cleaned = clean_text(text, lowercase=lowercase, normalize=normalize)
    if not cleaned:
        return ([], []) if return_segment_map else []

    segments: list[list[str]] = []
    segment_map: list[int] = []
    for part in cleaned.split(BREAK_TOKEN):
        part = part.strip()
        if len(part) <= 1:
            continue
        tokens = [m.group(0) for m in _WORD_RE.finditer(part)]
        if not tokens:
            continue
        if pos_filter:
            tokens = filter_pos(tokens, lang=pos_lang)
        if tokens:
            seg_idx = len(segments)
            segments.append(tokens)
            if return_segment_map:
                segment_map.extend([seg_idx] * len(tokens))

    if not return_segment_map:
        return segments

    return segments, segment_map


def tokenize(
    text: str,
    lowercase: bool = True,
    normalize: bool = True,
    clean_func: str = "clean_text",
    pos_filter: bool = False,
    pos_lang: str = "es",
) -> list[str]:
    """Flatten all segments into a single list of tokens."""
    return [
        token
        for segment in tokenize_segments(text, lowercase, normalize, clean_func=clean_func, pos_filter=pos_filter, pos_lang=pos_lang)
        for token in segment
    ]
