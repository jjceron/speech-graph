"""Tokenization and segmentation of normalized text."""

from __future__ import annotations

import re
from collections.abc import Collection
from functools import lru_cache

import stanza

from .annotations import BREAK_TOKEN, normalize_annotations_text

_POS_KEEP_DEFAULT = frozenset({"NOUN", "PRON", "VERB", "ADJ", "ADV"})
_WORD_RE = re.compile(
    r"\[\[EE\]\]|[0-9A-Za-z\u00C0-\u024F]+(?:-[0-9A-Za-z\u00C0-\u024F]+)?",
    flags=re.UNICODE,
    ) ### Extrae palabras (incluyendo acentuadas y compuestas con guion) y preserva el token especial [[EE]] como una unidad indivisible
_GRAPH_PLACEHOLDER = "___GRAPH_TOKEN___"
_BREAK_PLACEHOLDER = "___BREAK_PLACEHOLDER___"


def clean_text(
    text: str,
    lowercase: bool = True,
    normalize: bool = True,
) -> str:
    """
    Clean and normalize transcript text for downstream NLP processing.

    This function performs a multi-stage cleaning pipeline designed to
    transform raw transcript text containing speaker labels, annotation
    markup, and special tokens into a standardized, model-ready format.

    The cleaning process includes:
    - Optional normalization of annotation markup via ``normalize_annotations_text``
    - Removal of speaker labels (e.g., ``spk_1:``, ``spk_2:``)
    - Protection of important tokens (``[[EE]]`` and ``BREAK_TOKEN``) using
      temporary placeholders
    - Removal of parenthesized content while preserving inner text
    - Removal of specific punctuation and special characters (e.g., ^, <, >, ')
    - Optional conversion to lowercase
    - Removal of non-alphanumeric characters (keeping Unicode letters)
    - Whitespace normalization
    - Restoration of protected placeholders back to original tokens

    Token handling:
        - ``[[EE]]`` is preserved using a placeholder during cleaning
        - ``BREAK_TOKEN`` is preserved and restored after processing

    Args:
        text: Raw input transcript text containing annotations, speaker tags,
            and possibly noisy formatting.
        lowercase: If True, converts the final output to lowercase.
        normalize: If True, applies annotation normalization before cleaning.

    Returns:
        A cleaned and normalized string suitable for NLP tasks such as
        tokenization, embedding, or classification.

    Example:
        clean_text("spk_1: Hola [[EE]] (ruido)")

        -> "hola [[EE]] ruido"
    """

    value = str(text or "")
    value = re.sub(r"<\s*(.*?)\s*>", r"\n\1\n", value)
    value = normalize_annotations_text(value) if normalize else value ### Normalize annotations (optional)
    value = re.sub(r"\bspk_?\d*\s*:\s*", " ", value, flags=re.IGNORECASE) ### Remove spk_N: labels
    value = value.replace("[[EE]]", f" {_GRAPH_PLACEHOLDER} ") ### Protect [[EE]] with placeholder
    value = value.replace(BREAK_TOKEN, f" {_BREAK_PLACEHOLDER} ") ### Protect BREAK_TOKEN with placeholder
    ### Remove parenthesized content, ^, <, >, '
    value = re.sub(r"\(([^)]*)\)", r"\1", value)
    value = value.replace("^", "")
    value = value.replace("<", " ").replace(">", " ")
    value = value.replace("'", "")
    if lowercase: ### Lowercase (optional)
        value = value.lower()
    value = re.sub(r"[^0-9A-Za-z\u00C0-\u024F_]+", " ", value, flags=re.UNICODE) ### Strip non-alphanumeric characters
    value = re.sub(r"\s+", " ", value).strip()
    ### Restore placeholders
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
    """
    Clean and normalize with an agrresive approach transcript text for downstream NLP processing.

    This function performs a multi-stage cleaning pipeline designed to
    transform raw transcript text containing speaker labels, annotation
    markup, and special tokens into a standardized, model-ready format.

    The cleaning process includes:
    - Optional normalization of annotation markup via ``normalize_annotations_text``
    - Removal of speaker labels (e.g., ``spk_1:``, ``spk_2:``)
    - Delete [[EE]]
    - Protection of important tokens (``BREAK_TOKEN``) using
      temporary placeholders
    - Removal of parenthesized content while preserving inner text
    - Removal of specific punctuation and special characters (e.g., ^, <, >, ')
    - Optional conversion to lowercase
    - Removal of non-alphanumeric characters (keeping Unicode letters)
    - Whitespace normalization
    - Restoration of protected placeholders back to original tokens

    Token handling:
        - ``BREAK_TOKEN`` is preserved and restored after processing

    Args:
        text: Raw input transcript text containing annotations, speaker tags,
            and possibly noisy formatting.
        lowercase: If True, converts the final output to lowercase.
        normalize: If True, applies annotation normalization before cleaning.

    Returns:
        A cleaned and normalized string suitable for NLP tasks such as
        tokenization, embedding, or classification.

    Example:
        clean_text("spk_1: Hola [[EE]] (ruido)")

        -> "hola ruido"
    """

    value = normalize_annotations_text(text) if normalize else str(text or "") ### Normalize annotations (optional)
    value = re.sub(r"\bspk_?\d*\s*:\s*", " ", value, flags=re.IGNORECASE)  ### Remove spk_N: labels
    value = re.sub(r"\[\[EE\]\]", " ", value) ### Eliminar [[EE]]
    value = value.replace(BREAK_TOKEN, f" {_BREAK_PLACEHOLDER} ") ### Protect BREAK_TOKEN with placeholder
    ### Remove parenthesized content, ^, <, >, '
    value = re.sub(r"\(([^)]*)\)", r"\1", value)
    value = value.replace("^", "")
    value = value.replace("<", " ").replace(">", " ")
    value = value.replace("'", "")
    if lowercase: ### Lowercase (optional)
        value = value.lower()
    value = re.sub(r"[^0-9A-Za-z\u00C0-\u024F_]+", " ", value, flags=re.UNICODE) ### Strip non-alphanumeric characters
    value = re.sub(r"\s+", " ", value).strip()
    ### Restore placeholders
    value = value.replace(_BREAK_PLACEHOLDER.lower(), BREAK_TOKEN)
    value = value.replace(_BREAK_PLACEHOLDER, BREAK_TOKEN)
    return value


@lru_cache(maxsize=2)
def _get_stanza_pipeline(lang: str = "es") -> stanza.Pipeline:
    return stanza.Pipeline(
        lang=lang, processors="tokenize,pos,lemma", verbose=False, use_gpu=False, #### <---- OJO configuración de stanza
    )


def filter_pos(
    tokens: list[str],
    lang: str = "es",
    keep_tags: Collection[str] | None = None,
) -> list[str]:
    """
    Build a list of normalized tokens using lemmatization when available.

    This comprehension iterates over the original tokens and their
    corresponding linguistic annotations (from Stanza), and returns a
    new list where each token is replaced by its lemma if available;
    otherwise, the original token is preserved.

    The function assumes that `tokens` and `stanza_words` are aligned
    element-wise via `zip`, where each entry in `stanza_words` contains:
        - the original word text (unused here)
        - its POS tag (UPOS)
        - its lemma

    Transformation rule:
        - If lemma is not empty or None → use lemma
        - Otherwise → use original token

    Returns:
        A list of strings representing normalized tokens, optionally
        lemmatized.

    Example:
        tokens = ["corriendo", "perro"]
        stanza_words = [
            ("corriendo", "VERB", "correr"),
            ("perro", "NOUN", "perro")
        ]

        Result:
        ["correr", "perro"]
    """
    if keep_tags is None:
        keep_tags = _POS_KEEP_DEFAULT
    if not tokens:
        return tokens

    nlp = _get_stanza_pipeline(lang)
    text = " ".join(tokens)
    doc = nlp(text)

    stanza_words: list[tuple[str, str, str]] = []
    for sent in doc.sentences:
        for word in sent.words:
            stanza_words.append((word.text, word.upos, word.lemma))

    if len(stanza_words) != len(tokens):
        return tokens

    return [lemma if lemma else token for token, (_, upos, lemma) in zip(tokens, stanza_words) if upos in keep_tags] ### Recupera lemma (cuando existe) o token si UPOS en lista


def tokenize_segments(
    text: str,
    lowercase: bool = True,
    normalize: bool = True,
    return_segment_map: bool = False,
    clean_func: str = "clean_text",
    pos_filter: bool = False,
    pos_lang: str = "es",
) -> list[list[str]] | tuple[list[list[str]], list[int]]:
    """
    Tokenize cleaned transcript text into structured segments separated by BREAK_TOKEN.

    This function processes raw transcript text through a cleaning pipeline and
    then splits it into discourse segments using the special ``BREAK_TOKEN``.
    Each segment is further tokenized using a regex-based word tokenizer.

    Optionally, it can also produce a segment map that tracks which segment each
    token belongs to, which is useful for alignment, sequence modeling, and
    downstream NLP tasks.

    Processing steps:
        1. Clean and normalize input text using either ``clean_text`` or
           ``clean_text_all``.
        2. Split cleaned text into segments using ``BREAK_TOKEN`` as delimiter.
        3. Tokenize each segment using word-level regex extraction.
        4. Optionally apply POS-based filtering (e.g., via stanza).
        5. Build a structured representation of segments and tokens.
        6. Optionally generate a segment index map for each token.

    Args:
        text: Raw transcript text.
        lowercase: If True, converts text to lowercase during cleaning.
        normalize: If True, applies annotation normalization before cleaning.
        return_segment_map: If True, also returns a list mapping each token
            to its corresponding segment index.
        clean_func: Cleaning function to use:
            - "clean_text" for standard cleaning
            - "clean_text_all" for more aggressive cleaning
        pos_filter: If True, applies part-of-speech filtering to tokens using
            an external NLP model (e.g., stanza).
        pos_lang: Language code used for POS tagging (default is "es").

    Returns:
        If return_segment_map is False:
            A list of segments, where each segment is a list of tokens.

        If return_segment_map is True:
            A tuple containing:
                - segments: List of tokenized segments
                - segment_map: List mapping each token to its segment index

    Example:
        segments, segment_map = tokenize_segments(
            "hola mundo BREAK_TOKEN buenos días",
            return_segment_map=True
        )

        # segments:
        # [["hola", "mundo"], ["buenos", "días"]]
        #
        # segment_map:
        # [0, 0, 1, 1]
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
