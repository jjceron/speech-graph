from __future__ import annotations

import re
from dataclasses import dataclass

GRAPH_TOKEN_LABELS = {"EE"}
BREAK_LABELS = {"PAUSA", "DP", "DI"}
DROP_LABELS = {"ES", "IF", "PS", "PNC", "IM", "SIN_RESPUESTA", "SIN_PREGUNTA"}
BREAK_TOKEN = "__TRANSCRIPT_BREAK__"

LABEL_ALIASES = {
    "EE": "EE", "EEE": "EE", "E E": "EE",
    "ES": "ES", "E S": "ES",
    "IF": "IF", "PS": "PS", "PNC": "PNC", "IM": "IM",
    "DP": "DP", "DI": "DI",
    "PAUSA": "PAUSA", "PAUA": "PAUSA", "PAUS": "PAUSA", "Pausa": "PAUSA",
    "SIN RESPUESTA": "SIN_RESPUESTA", "SIN_RESPUESTA": "SIN_RESPUESTA",
    "SIN-RESPUESTA": "SIN_RESPUESTA", "SIN REPUESTA": "SIN_RESPUESTA",
    "SIN_REPUESTA": "SIN_RESPUESTA", "SIN-REPUESTA": "SIN_RESPUESTA",
    "SIN PREGUNTA": "SIN_PREGUNTA", "SIN_PREGUNTA": "SIN_PREGUNTA", "SIN-PREGUNTA": "SIN_PREGUNTA",
}

DOUBLE_BRACKET_RE = re.compile(r"\[\[(.*?)\]\]", flags=re.DOTALL)
STANDALONE_TIMESTAMP_RE = re.compile(
    r"\[\s*StartTime\s*=?\s*[^\]\s]+\s+EndTime\s*=?\s*[^\]]+\]",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class ActivityClass:
    raw: str
    number: int | None
    canonical: str


def _strip_timestamp_fields(text: str) -> str:
    out = re.sub(r"\bStartTime\s*=?\s*[^\s\]]+", "", text, flags=re.IGNORECASE)
    out = re.sub(r"\bStarTime\s*=?\s*[^\s\]]+", "", out, flags=re.IGNORECASE)
    out = re.sub(r"\bEndTime\s*=?\s*[^\s\]]+", "", out, flags=re.IGNORECASE)
    return out


def canonical_label(label: str) -> str:
    text = str(label or "").strip().strip("[]")
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip(" _-;,:.")
    if re.match(r"^ES\s*=", text, flags=re.IGNORECASE):
        return "ES"
    text = _strip_timestamp_fields(text)
    text = re.sub(r"\s+", " ", text).strip(" _-;,:.")
    key = text.upper().replace("-", "_")
    key_space = key.replace("_", " ")
    for candidate in (text, key, key_space):
        if candidate in LABEL_ALIASES:
            return LABEL_ALIASES[candidate]
    first = re.split(r"[\s_=;,:.-]+", key)[0]
    if first in LABEL_ALIASES:
        return LABEL_ALIASES[first]
    return key or "EMPTY"


def normalize_double_bracket_markup(text: str) -> str:
    out = str(text or "")
    out = re.sub(r"\[\[\[+", "[[", out)
    out = re.sub(r"\]\]\]+", "]]", out)
    out = re.sub(
        r"\[\[\s*(ES|EE|EEE|IF|PS|PNC|DI|DP|IM|PAUSA)\s*\](?!\])",
        r"[[\1]]",
        out,
        flags=re.IGNORECASE,
    )
    out = re.sub(
        r"\[\[\s*(ES|EE|EEE|IF|PS|PNC|DI|DP|IM|PAUSA)\s*\]\[",
        r"[[\1]]",
        out,
        flags=re.IGNORECASE,
    )
    return out


def normalize_annotations_text(text: str) -> str:
    raw = normalize_double_bracket_markup(text)

    def replace_label(match: re.Match[str]) -> str:
        label = canonical_label(match.group(1))
        if label in GRAPH_TOKEN_LABELS:
            return f" [[{label}]] "
        if label in BREAK_LABELS:
            return f" {BREAK_TOKEN} "
        return " "

    out = DOUBLE_BRACKET_RE.sub(replace_label, raw)
    out = STANDALONE_TIMESTAMP_RE.sub(" ", out)
    out = re.sub(r"\s+", " ", out).strip()
    return out


def canonical_activity(value: object) -> ActivityClass:
    raw = str(value or "").strip()
    match = re.search(r"Actividad\s*([0-9]+)", raw, flags=re.IGNORECASE) or re.search(r"\b([0-9]+)\b", raw)
    if match:
        number = int(match.group(1))
        return ActivityClass(raw=raw, number=number, canonical=f"Actividad{number}")
    return ActivityClass(raw=raw, number=None, canonical=raw or "UNSEGMENTED")
