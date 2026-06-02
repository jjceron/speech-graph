from __future__ import annotations

import re


def normalize_code(value: object) -> str:
    text = str(value or "").strip()
    text = re.sub(r"\.(txt|wav|mp3)$", "", text, flags=re.IGNORECASE)
    for suffix in ("_CorrEtiq", "-CorrEtiq", " CorrEtiq"):
        if text.endswith(suffix):
            text = text[: -len(suffix)]
    return re.sub(r"\s+", "", text).upper()
