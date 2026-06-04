"""Classification and normalization of annotations."""

from __future__ import annotations

import re
from typing import Iterable

# --- Label sets ---

GRAPH_TOKEN_LABELS: set[str] = {"EE"}
BREAK_LABELS: set[str] = {"PAUSA", "DP", "DI"}
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
    """Normalize annotation label to canonical form.

    Examples: 'EEE'->'EE', 'PAUA'->'PAUSA', 'ES=...'->'ES'
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
    """Fix malformed double-bracket markup."""
    out = str(text or "")
    out = re.sub(r"\[\[\[+", "[[", out)
    out = re.sub(
        r"\[\[\s*(ES|EE|EEE|IF|PS|PNC|DI|DP|IM|PAUSA)\s*\](?!\])",
        r"[[\1]]", out, flags=re.IGNORECASE,
    )
    out = re.sub(
        r"\[\[\s*(ES|EE|EEE|IF|PS|PNC|DI|DP|IM|PAUSA)\s*\]\[",
        r"[[\1]]", out, flags=re.IGNORECASE,
    )
    return out


def normalize_annotations_text(text: str) -> str:
    """Normalize annotation markup in raw text.

    - [[EE]]        -> [[EE]]      (graph token preserved)
    - [[PAUSA]]/[[DI]]/[[DP]] -> BREAK_TOKEN (segment breaks)
    - [[IF]]/[[PS]] -> ' '         (dropped)
    - newlines      -> BREAK_TOKEN (segment breaks from file structure)
    """
    raw = _normalize_double_bracket_markup(text)
    raw = re.sub(r"\n+", f" {BREAK_TOKEN} ", raw)

    def _replace(match: re.Match[str]) -> str:
        kind = classify_annotation(match.group(1))
        if kind == "graph":
            return f" [[{canonical_label(match.group(1))}]] "
        if kind == "break":
            return f" {BREAK_TOKEN} "
        return " "

    out = _DOUBLE_BRACKET_RE.sub(_replace, raw)
    out = _STANDALONE_TIMESTAMP_RE.sub(" ", out)
    return re.sub(r"\s+", " ", out).strip()


def extract_double_bracket_labels(text_or_tokens: str | Iterable[str]) -> list[str]:
    """Extract canonical labels from [[...]] annotations."""
    if isinstance(text_or_tokens, str):
        text = normalize_annotations_text(text_or_tokens)
        return [canonical_label(m.group(1)) for m in _DOUBLE_BRACKET_RE.finditer(text)]
    labels: list[str] = []
    for token in text_or_tokens:
        m = re.fullmatch(r"\[\[(.*?)\]\]", str(token).strip())
        if m:
            labels.append(canonical_label(m.group(1)))
    return labels
