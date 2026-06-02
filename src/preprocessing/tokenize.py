from __future__ import annotations

import re

from .annotations import BREAK_TOKEN, normalize_annotations_text

EE_PLACEHOLDER = "zzzeetokenzzz"
BREAK_PLACEHOLDER = "zzzbreaktokenzzz"
WORD_RE = re.compile(r"\[\[EE\]\]|[0-9A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+(?:-[0-9A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+)?", flags=re.UNICODE)


def clean_text_for_nlp(text: str, lowercase: bool = True, normalize_annotations: bool = True) -> str:
    value = normalize_annotations_text(text) if normalize_annotations else str(text or "")
    value = re.sub(r"\bspk_?\d*\s*:\s*", " ", value, flags=re.IGNORECASE)
    value = value.replace("[[EE]]", f" {EE_PLACEHOLDER} ")
    value = value.replace(BREAK_TOKEN, f" {BREAK_PLACEHOLDER} ")
    value = re.sub(r"\(([^)]*)\)", r"\1", value)
    value = value.replace("^", "")
    value = value.replace("<", " ").replace(">", " ")
    value = value.replace("'", "")
    if lowercase:
        value = value.lower()
    value = re.sub(r"[^0-9A-Za-zÁÉÍÓÚÜÑáéíóúüñ_]+", " ", value, flags=re.UNICODE)
    value = re.sub(r"\s+", " ", value).strip()
    value = value.replace(EE_PLACEHOLDER.lower(), "[[EE]]").replace(EE_PLACEHOLDER, "[[EE]]")
    value = value.replace(BREAK_PLACEHOLDER.lower(), BREAK_TOKEN).replace(BREAK_PLACEHOLDER, BREAK_TOKEN)
    return value


def tokenize_segments(text: str, lowercase: bool = True, normalize_annotations: bool = True) -> list[list[str]]:
    cleaned = clean_text_for_nlp(text, lowercase=lowercase, normalize_annotations=normalize_annotations)
    if not cleaned:
        return []
    segments: list[list[str]] = []
    for part in cleaned.split(BREAK_TOKEN):
        tokens = [m.group(0) for m in WORD_RE.finditer(part)]
        if tokens:
            segments.append(tokens)
    return segments


def tokenize(text: str, lowercase: bool = True, normalize_annotations: bool = True) -> list[str]:
    return [token for segment in tokenize_segments(text, lowercase, normalize_annotations) for token in segment]
