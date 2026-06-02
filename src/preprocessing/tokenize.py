from __future__ import annotations

import re
from typing import List

from .annotations import normalize_annotations_text

# Preserves [[...]] protocol tags as indivisible tokens because they are analytically meaningful.
# Standalone technical timestamps are removed by normalize_annotations_text before tokenization.
TOKEN_RE = re.compile(r"\[\[[^\]]+\]\]|\<[^>]+\>|\S+")


def tokenize(text: str, lowercase: bool = False, normalize_annotations: bool = True) -> List[str]:
    if normalize_annotations:
        text = normalize_annotations_text(text)
    tokens = TOKEN_RE.findall(text)
    if lowercase:
        # Do not lowercase protocol tags, because their identity is meaningful and canonicalized.
        tokens = [token if token.startswith("[[") else token.lower() for token in tokens]
    return tokens
