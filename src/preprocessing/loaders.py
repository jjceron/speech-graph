"""File loaders for simple .txt and transcript .txt formats."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .annotations import normalize_annotations_text


# --- Transcript parsing constants ---

_TRANSINFO_RE = re.compile(r"<TransInfo\s+([^>]+)>")
_TRANSINFO_ATTR_RE = re.compile(r"(\w+)=([^,>]+)")
_ACTIVITY_RE = re.compile(
    r"<<\s*NombreActividad\s*=\s*([^>]+)>>\s*"
    r"(?:\[\s*StartTime\s*=\s*([^\s\]]+)\s+EndTime\s*=\s*([^\]]+)\])?",
    flags=re.IGNORECASE,
)
_SPEAKER_RE = re.compile(r"^(spk_(?:\d+)?):\s*(.*)$", flags=re.IGNORECASE)
_STRUCTURAL_RE = re.compile(
    r"^</?\w+|^<Speaker\b|^<Speakers>|^</Speakers>", flags=re.IGNORECASE,
)


# --- Transcript data classes ---

@dataclass
class Activity:
    name: str
    start_time: str | None
    end_time: str | None
    raw_lines: list[str] = field(default_factory=list)

    def text(self) -> str:
        """Normalized text for this activity."""
        joined = " ".join(line.strip() for line in self.raw_lines if line.strip())
        return normalize_annotations_text(joined)


@dataclass
class Transcript:
    code: str
    path: Path
    scribe: str | None = None
    audio_filename: str | None = None
    date: str | None = None
    activities: list[Activity] = field(default_factory=list)


# --- Internal helpers ---

def _extract_code(filename: str) -> str:
    name = Path(filename).name
    if name.endswith(".txt"):
        name = name[:-4]
    for suffix in ("_CorrEtiq", "-CorrEtiq", " CorrEtiq"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name.strip()


def _parse_transinfo(lines: list[str]) -> dict[str, str]:
    for line in lines:
        match = _TRANSINFO_RE.search(line)
        if match:
            return {k: v.strip() for k, v in _TRANSINFO_ATTR_RE.findall(match.group(1))}
    return {}


def _canonical_activity_name(raw: str) -> str:
    m = re.search(r"Actividad\s*(\d+)", raw, flags=re.IGNORECASE)
    if m:
        return f"Actividad{m.group(1)}"
    m2 = re.search(r"(\d+)", raw)
    if m2:
        return f"Actividad{m2.group(1)}"
    return raw.strip() or "UNSEGMENTED"


# --- Public loaders ---

def load_simple_txt(filepath: str | Path) -> str:
    """Load a simple .txt file (dummy tests, preprocessed activities).

    Returns the raw text with newlines preserved.
    """
    return Path(filepath).read_text(encoding="utf-8", errors="replace").strip()


def parse_transcript(
    filepath: str | Path,
    include_speakers: Iterable[str] = ("spk_1",),
) -> Transcript:
    """Parse a transcript .txt file (data/raw/transcripts/*-CorrEtiq.txt).

    Args:
        filepath: Path to the transcript file.
        include_speakers: Speaker IDs whose lines to include.

    Returns:
        Transcript object with parsed activities.
    """
    path = Path(filepath)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    info = _parse_transinfo(lines)

    transcript = Transcript(
        code=_extract_code(path.name),
        path=path,
        scribe=info.get("scribe"),
        audio_filename=info.get("audio_filename"),
        date=info.get("date"),
    )

    include_set = {s.strip().lower() for s in include_speakers}
    current: Activity | None = None
    current_speaker: str | None = None

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue

        act_match = _ACTIVITY_RE.match(line)
        if act_match:
            if current is not None:
                transcript.activities.append(current)
            name = _canonical_activity_name(act_match.group(1).strip())
            current = Activity(
                name=name,
                start_time=act_match.group(2),
                end_time=act_match.group(3),
            )
            current_speaker = None
            continue

        spk_match = _SPEAKER_RE.match(line)
        if spk_match:
            raw_speaker = spk_match.group(1).lower()
            speaker_id = raw_speaker if re.search(r"\d", raw_speaker) else (current_speaker or "spk_1")
            current_speaker = speaker_id
            if current is None:
                current = Activity(name="UNSEGMENTED", start_time=None, end_time=None)
            if speaker_id in include_set:
                current.raw_lines.append(spk_match.group(2))
            continue

        if current is not None and current_speaker in include_set:
            if not _STRUCTURAL_RE.match(line):
                current.raw_lines.append(line)

    if current is not None:
        transcript.activities.append(current)

    return transcript


def load_transcript_txt(
    filepath: str | Path,
    include_speakers: Iterable[str] = ("spk_1",),
) -> list[dict]:
    """Load a transcript file and return activities as list of dicts.

    Each dict: {'name', 'start_time', 'end_time', 'text'}
    """
    transcript = parse_transcript(filepath, include_speakers=include_speakers)
    return [
        {
            "name": act.name,
            "start_time": act.start_time,
            "end_time": act.end_time,
            "text": act.text(),
        }
        for act in transcript.activities
        if act.text()
    ]
