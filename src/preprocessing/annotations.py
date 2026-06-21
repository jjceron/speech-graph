"""Classification and normalization of annotations."""

from __future__ import annotations

import re
from typing import Iterable

# --- Label sets ---

GRAPH_TOKEN_LABELS: set[str] = {"EE"}
BREAK_LABELS: set[str] = {"PAUSA", "DP", "DI", "D1"}
DROP_LABELS: set[str] = {"ES", "IF", "PS", "PNC", "IM", "SIN_RESPUESTA", "SIN_PREGUNTA"}
BREAK_TOKEN: str = "__TRANSCRIPT_BREAK__"

LABEL_ALIASES: dict[str, str] = {
    "EE": "EE", "EEE": "EE", "E E": "EE",
    "ES": "ES", "E S": "ES",
    "IF": "IF", "PS": "PS", "PNC": "PNC", "IM": "IM",
    "DI": "DI", "DP": "DP", "PAUSA": "PAUSA", "PAUA": "PAUSA", "PAUS": "PAUSA",
    "SIN RESPUESTA": "SIN_RESPUESTA", "SIN_RESPUESTA": "SIN_RESPUESTA", "SIN-RESPUESTA": "SIN_RESPUESTA",
    "SIN PREGUNTA": "SIN_PREGUNTA", "SIN_PREGUNTA": "SIN_PREGUNTA", "SIN-PREGUNTA": "SIN_PREGUNTA",
}

_DOUBLE_BRACKET_RE = re.compile(r"\[\[(.*?)\]\]", flags=re.DOTALL)
_STANDALONE_TIMESTAMP_RE = re.compile(
    r"\[\s*StartTime\s*=?\s*[^\]\s]+\s+EndTime\s*=?\s*[^\]]+\]",
    flags=re.IGNORECASE,
)


def canonical_label(label: str) -> str:
    """
    Normalize an annotation label to its canonical form.

    This function cleans and standardizes raw annotation labels extracted
    from transcript markup. It removes formatting noise (brackets,
    whitespace, timestamps, and punctuation), handles special patterns
    (e.g., "ES=..."), and maps known variants or misspellings to their
    canonical equivalents using a predefined alias dictionary
    (``LABEL_ALIASES``).

    The normalization process includes:
    - Stripping brackets, whitespace, and punctuation artifacts
    - Removing embedded newline characters
    - Eliminating time-related tokens (e.g., StartTime, EndTime)
    - Converting label variants to uppercase canonical keys
    - Resolving known aliases via ``LABEL_ALIASES``
    - Fallback extraction of the first meaningful token if needed

    Examples:
        canonical_label("EEE")        -> "EE"
        canonical_label("PAUA")       -> "PAUSA"
        canonical_label("ES=hola")    -> "ES"
        canonical_label("DI StartTime=08:18 EndTime=08:32") -> "DI"

    Args:
        label: Raw annotation label possibly containing noise or formatting
            inconsistencies.

    Returns:
        A canonicalized label string. If no valid label can be inferred,
        returns "EMPTY".
    """
    text = str(label or "").strip().strip("[]")
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip(" _-;,:.")
    if re.match(r"^ES\s*=", text, flags=re.IGNORECASE):
        return "ES"
    text = re.sub(r"\bStartTime\s*=?\s*[^\s\]]+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bStarTime\s*=?\s*[^\s\]]+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bEndTime\s*=?\s*[^\s\]]+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip(" _-;,:.")
    key = text.upper().replace("-", "_")
    key_space = key.replace("_", " ")
    for candidate in (text.upper(), key, key_space):
        if candidate in LABEL_ALIASES:
            return LABEL_ALIASES[candidate]
    first = re.split(r"[\s_=;,:.-]+", key)[0]
    if first in LABEL_ALIASES:
        return LABEL_ALIASES[first]
    return key or "EMPTY"


def classify_annotation(label: str) -> str:
    """Classify label as 'graph', 'break', 'drop', or 'unknown'."""
    canonical = canonical_label(label)
    if canonical in GRAPH_TOKEN_LABELS:
        return "graph"
    if canonical in BREAK_LABELS:
        return "break"
    if canonical in DROP_LABELS:
        return "drop"
    return "unknown"


def _normalize_double_bracket_markup(text: str) -> str:
    """
    Normalize malformed double-bracket annotation markup in transcript text.

    This function fixes common formatting issues in annotation tags of the form
    ``[[TAG]]`` used in transcripts. It corrects inconsistencies such as:

    - Extra opening brackets (e.g., ``[[[PS]]`` → ``[[PS]]``)
    - Missing or malformed closing brackets (e.g., ``[[PS]`` → ``[[PS]]``)
    - Incorrect bracket combinations (e.g., ``[[PS][`` → ``[[PS]]``)

    The following annotation labels are recognized and normalized:
    ``ES, EE, EEE, IF, PS, PNC, DI, DP, IM, PAUSA``.

    The function applies a series of regular expression substitutions to ensure
    all valid annotation tags follow a consistent ``[[TAG]]`` format, ignoring
    case sensitivity.

    Args:
        text: Input transcript text potentially containing malformed annotation
            markup.

    Returns:
        A normalized version of the input text where all supported annotation
        tags are properly formatted as ``[[TAG]]``.
    """
    out = str(text or "")
    out = re.sub(r"\[\[\[+", "[[", out) ### Corregir demasiados corchetes de apertura [[[[EE]] --> [[EE]]
    out = re.sub(
        r"\[\[\s*(ES|EE|EEE|IF|PS|PNC|DI|DP|IM|PAUSA)\s*\](?!\])",
        r"[[\1]]", out, flags=re.IGNORECASE,
    ) ### Corregir etiquetas mal cerradas o mal escritas [[EE] --> [[EE]] [[di]] --> [[DI]]
    out = re.sub(
        r"\[\[\s*(ES|EE|EEE|IF|PS|PNC|DI|DP|IM|PAUSA)\s*\]\[",
        r"[[\1]]", out, flags=re.IGNORECASE,
    ) ### Corregir otro tipo de error de formato [[EE]][ --> [[EE]]
    return out


def normalize_annotations_text(text: str) -> str:
    """
    Normalize annotation markup in raw transcript text.

    This function processes raw transcript text containing double-bracket
    annotations (e.g., [[EE]], [[PAUSA]], [[PS]]) and converts it into a
    clean, structured representation suitable for downstream linguistic
    analysis.

    The normalization process includes:
    - Fixing malformed annotation markup using `_normalize_double_bracket_markup`
    - Classifying annotations into graph, break, or drop categories
    - Preserving meaningful graph annotations (e.g., [[EE]]) after canonicalization
    - Converting break annotations (e.g., [[PAUSA]], [[DI]], [[DP]]) into
      `BREAK_TOKEN` to represent discourse segmentation
    - Dropping non-informative annotations (e.g., [[IF]], [[PS]])
    - Removing standalone timestamp metadata (StartTime/EndTime blocks)
    - Converting newlines into `BREAK_TOKEN` to preserve structural boundaries
    - Normalizing whitespace in the final output

    Annotation handling rules:
        [[EE]]                  -> preserved as graph annotation (normalized)
        [[PAUSA]]               -> BREAK_TOKEN
        [[DI]]                  -> BREAK_TOKEN
        [[DP]]                  -> BREAK_TOKEN
        <texto> [[DP]]          -> BREAK_TOKEN texto BREAK_TOKEN
        [[IF]]                  -> removed
        [[PS]]                  -> removed
        [[SIN_RESPUESTA]]       -> removed
        newlines                -> BREAK_TOKEN

    Args:
        text: Raw transcript text potentially containing annotation markup,
            timestamps, and formatting inconsistencies.

    Returns:
        A cleaned and normalized text string where annotations are either
        preserved, converted into segment boundaries, or removed.
    """
    raw = _normalize_double_bracket_markup(text) ### Normalize malformed double-bracket annotation markup in transcript text.

    def _replace(match: re.Match[str]) -> str: ### Definir qué hacer con cada [[...]]
        kind = classify_annotation(match.group(1))
        if kind == "graph":
            return f" [[{canonical_label(match.group(1))}]] " ### Normalizar etiqueta [[DI StartTime=08:18 EndTime=08:32]] --> [[DI]] [[ee]] --> [[EE]]
        if kind == "break":
            return f" {BREAK_TOKEN} "
        return " "

    out = _DOUBLE_BRACKET_RE.sub(_replace, raw)

    out = re.sub(r"<\s*([^>]+?)\s*>\s*" + re.escape(BREAK_TOKEN),
           rf" {BREAK_TOKEN} \1 {BREAK_TOKEN} ", out,) ### Generar BREAK_TOKEN antes de texto asociado (<>) a DI/DP. <caballero> [[DI]] --> __TRANSCRIPT_BREAK__ caballero __TRANSCRIPT_BREAK__
    out = _STANDALONE_TIMESTAMP_RE.sub(" ", out)
    return re.sub(r"\s+", " ", out).strip()


def extract_double_bracket_labels(text_or_tokens: str | Iterable[str]) -> list[str]:
    """
    Extract canonical annotation labels from double-bracket markup.

    This function extracts labels enclosed in double-bracket annotations
    (e.g., [[EE]], [[PAUSA]], [[PS]]) and returns them in their canonical
    form using ``canonical_label()``.

    The function supports two input types:

    1. String input:
       - The text is first normalized using ``normalize_annotations_text``
         to fix malformed markup and clean annotation structure.
       - All occurrences of ``[[...]]`` are extracted and converted into
         canonical labels.

    2. Iterable of tokens:
       - Each token is checked individually.
       - Only tokens that exactly match the pattern ``[[...]]`` are
         considered valid annotations.
       - Non-matching tokens are ignored.

    Args:
        text_or_tokens: Input data containing annotation markup. It can be
            either a raw transcript string or an iterable of token strings.

    Returns:
        A list of canonical annotation labels extracted from the input.
        If no valid annotations are found, returns an empty list.

    Examples:
        >>> extract_double_bracket_labels("Hola [[EE]] mundo [[PAUSA]]")
        ['EE', 'PAUSA']

        >>> extract_double_bracket_labels(["[[EE]]", "hola", "[[PS]]"])
        ['EE', 'PS']
    """
    if isinstance(text_or_tokens, str):
        text = normalize_annotations_text(text_or_tokens)
        return [canonical_label(m.group(1)) for m in _DOUBLE_BRACKET_RE.finditer(text)]
    labels: list[str] = []
    for token in text_or_tokens:
        m = re.fullmatch(r"\[\[(.*?)\]\]", str(token).strip())
        if m:
            labels.append(canonical_label(m.group(1)))
    return labels
