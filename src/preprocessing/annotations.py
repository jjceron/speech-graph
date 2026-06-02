from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

# Protocol labels described in ProtocoloTranscripciones-05222025.pdf.
# The code canonicalizes variants so raw .txt files do not need to be edited
# before analysis.
VALID_BASE_LABELS = {
    "EE", # as token
    # "IF",
    # "PS",
    "PAUSA", # breakline
    # "PNC",
    "DP", # breakline
    "DI", # breakline
    # "SIN_RESPUESTA",
    # "SIN_PREGUNTA",
    # "IM",
    # "ES",
}

LABEL_ALIASES = {
    "SIN RESPUESTA": "SIN_RESPUESTA",
    "SIN_RESPUESTA": "SIN_RESPUESTA",
    "SIN-RESPUESTA": "SIN_RESPUESTA",
    "SINREPUESTA": "SIN_RESPUESTA",
    "SIN REPUESTA": "SIN_RESPUESTA",
    "SIN_REPUESTA": "SIN_RESPUESTA",
    "SIN-REPUESTA": "SIN_RESPUESTA",
    "SIN PREGUNTA": "SIN_PREGUNTA",
    "SIN_PREGUNTA": "SIN_PREGUNTA",
    "SIN-PREGUNTA": "SIN_PREGUNTA",
    "EEE": "EE",
    "E E": "EE",
    "IF": "IF",
    "if": "IF",
    "PS": "PS",
    "ps": "PS",
    "PAUA": "PAUSA",
    "Pausa": "PAUSA",
    "PAUSA": "PAUSA",
    "PNC": "PNC",
    "DP": "DP",
    "DI": "DI",
    "IM": "IM",
}

DOUBLE_BRACKET_RE = re.compile(r"\[\[(.*?)\]\]", flags=re.DOTALL)
STANDALONE_TIMESTAMP_RE = re.compile(
    r"\[\s*StartTime\s*=?\s*[^\]\s]+\s+EndTime\s*=?\s*[^\]]+\]",
    flags=re.IGNORECASE,
)


def _strip_timestamp_fields(text: str) -> str:
    text = re.sub(r"\bStartTime\s*=?\s*[^\s\]]+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bStarTime\s*=?\s*[^\s\]]+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bEndTime\s*=?\s*[^\s\]]+", "", text, flags=re.IGNORECASE)
    # Common malformed variants: EndTime13:38 or EndTime16:18
    text = re.sub(r"\bEndTime\d{1,2}:\d{2}\s*=?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[_\s-]*StartTime[_\s:=.-].*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[_\s-]*StarTime[_\s:=.-].*$", "", text, flags=re.IGNORECASE)
    return text


def canonical_label(label: str, collapse_es_text: bool = True) -> str:
    """Return a protocol-level label category.

    Examples
    --------
    [[PAUSA StartTime=13:20 EndTime=13:38]] -> PAUSA
    [[SIN RESPUESTA]] / [[SIN REPUESTA]] -> SIN_RESPUESTA
    [[ES=sonríe]] -> ES when collapse_es_text=True
    """
    original = str(label).strip().strip("[]")
    text = original.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    text = text.strip(" _-;,:.")

    # Preserve ES as a class. The free text is useful qualitatively but too sparse
    # as a node/feature for the main graph analysis.
    if re.match(r"^ES\s*=", text, flags=re.IGNORECASE):
        return "ES" if collapse_es_text else re.sub(r"\s+", "_", text)

    text = _strip_timestamp_fields(text)
    text = re.sub(r"\s+", " ", text).strip(" _-;,:.")
    key = text.upper().replace("-", "_")
    key_space = key.replace("_", " ")

    if text in LABEL_ALIASES:
        return LABEL_ALIASES[text]
    if key in LABEL_ALIASES:
        return LABEL_ALIASES[key]
    if key_space in LABEL_ALIASES:
        return LABEL_ALIASES[key_space]

    # If the first token is a known label with extra malformed content, keep the category.
    first = re.split(r"[\s_=;,:.-]+", key)[0]
    if first in LABEL_ALIASES:
        return LABEL_ALIASES[first]
    if first in VALID_BASE_LABELS:
        return first

    if key in VALID_BASE_LABELS:
        return key
    return key or "EMPTY"


def normalize_double_bracket_markup(text: str) -> str:
    """Repair common manual-annotation bracket glitches without changing files.

    This intentionally handles only conservative cases found in the inspection:
    [[EE], [[IF], [[PS], [[[DI ...]], and [[PS][ .
    """
    out = str(text)
    out = re.sub(r"\[\[\[+", "[[", out)
    # [[EE] not followed by ] -> [[EE]]
    out = re.sub(r"\[\[\s*(EE|EEE|IF|PS|PNC|DI|DP|IM)\s*\](?!\])", r"[[\1]]", out, flags=re.IGNORECASE)
    # [[PS][ -> [[PS]]
    out = re.sub(r"\[\[\s*(EE|EEE|IF|PS|PNC|DI|DP|IM)\s*\]\[", r"[[\1]]", out, flags=re.IGNORECASE)
    return out


def normalize_annotations_text(text: str) -> str:
    """Normalize protocol annotations in text used by NLP.

    - Keeps protocol labels as double-bracket tokens.
    - Collapses timestamped labels to their analytic category.
    - Removes standalone [StartTime=... EndTime=...] technical timestamps.
    - Does not remove stopwords, repetitions or discourse labels.
    """
    raw = normalize_double_bracket_markup(text)

    placeholders: list[str] = []

    def _label_repl(match: re.Match[str]) -> str:
        canonical = canonical_label(match.group(1), collapse_es_text=True)
        placeholders.append(f"[[{canonical}]]")
        return f"@@LABEL_{len(placeholders)-1}@@"

    protected = DOUBLE_BRACKET_RE.sub(_label_repl, raw)
    protected = STANDALONE_TIMESTAMP_RE.sub(" ", protected)
    protected = re.sub(r"\s+", " ", protected)

    def _restore(match: re.Match[str]) -> str:
        idx = int(match.group(1))
        return placeholders[idx]

    restored = re.sub(r"@@LABEL_(\d+)@@", _restore, protected)
    return restored.strip()


def extract_double_bracket_labels(text_or_tokens: str | Iterable[str]) -> list[str]:
    if isinstance(text_or_tokens, str):
        text = normalize_annotations_text(text_or_tokens)
        return [canonical_label(m.group(1)) for m in DOUBLE_BRACKET_RE.finditer(text)]
    labels: list[str] = []
    for token in text_or_tokens:
        m = re.fullmatch(r"\[\[(.*?)\]\]", str(token).strip())
        if m:
            labels.append(canonical_label(m.group(1)))
    return labels


def safe_label_name(label: str) -> str:
    text = canonical_label(label)
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^0-9A-Za-zÁÉÍÓÚÜÑáéíóúüñ_.-]+", "_", text, flags=re.UNICODE)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "EMPTY"


@dataclass(frozen=True)
class ActivityClass:
    raw: str
    number: int | None
    canonical: str


def canonical_activity(value: object) -> ActivityClass:
    raw = str(value or "").strip()
    match = re.search(r"Actividad\s*([0-9]+)", raw, flags=re.IGNORECASE)
    if not match:
        match = re.search(r"\b([0-9]+)\b", raw)
    if match:
        num = int(match.group(1))
        return ActivityClass(raw=raw, number=num, canonical=f"Actividad{num}")
    return ActivityClass(raw=raw, number=None, canonical=raw or "UNSEGMENTED")
