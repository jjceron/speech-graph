from __future__ import annotations

from pathlib import Path
from typing import Iterable


def normalize_code(value: object) -> str:
    """Normalize transcript/metadata codes for safer joins.

    This removes only file/format artifacts. It does not invent leading zeros that
    may have been lost if an Excel column was stored as numeric.
    """
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "nat"}:
        return ""
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    if text.endswith(".txt"):
        text = text[:-4]
    if text.endswith("-CorrEtiq"):
        text = text[:-9]
    return text.strip()


def find_transcript_files(directory: Path, pattern: str = "*.txt") -> list[Path]:
    return sorted(Path(directory).glob(pattern))


def ensure_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
