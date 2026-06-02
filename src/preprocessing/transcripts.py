from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .annotations import canonical_activity, normalize_annotations_text

TRANSINFO_RE = re.compile(r"<TransInfo\s+([^>]+)>")
TRANSINFO_ATTR_RE = re.compile(r"(\w+)=([^,>]+)")
ACTIVITY_RE = re.compile(
    r"<<\s*NombreActividad\s*=\s*([^>]+)>>\s*(?:\[\s*StartTime\s*=\s*([^\s\]]+)\s+EndTime\s*=\s*([^\]]+)\])?",
    flags=re.IGNORECASE,
)
SPEAKER_RE = re.compile(r"^(spk_(?:\d+)?):\s*(.*)$", flags=re.IGNORECASE)
STRUCTURAL_LINE_RE = re.compile(r"^</?\w+|^<Speaker\b|^<Speakers>|^</Speakers>", flags=re.IGNORECASE)


@dataclass
class Activity:
    name: str
    start_time: str | None
    end_time: str | None
    speaker_lines: list[str] = field(default_factory=list)

    @property
    def canonical_name(self) -> str:
        return canonical_activity(self.name).canonical

    @property
    def number(self) -> int | None:
        return canonical_activity(self.name).number

    def text(self) -> str:
        return normalize_annotations_text(" ".join(line.strip() for line in self.speaker_lines if line.strip()))


@dataclass
class Transcript:
    code: str
    path: Path
    scribe: str | None = None
    audio_filename: str | None = None
    date: str | None = None
    activities: list[Activity] = field(default_factory=list)

    def text(self) -> str:
        return " ".join(activity.text() for activity in self.activities if activity.text())


def extract_code(filename: str) -> str:
    name = Path(filename).name
    if name.lower().endswith(".txt"):
        name = name[:-4]
    for suffix in ("_CorrEtiq", "-CorrEtiq", " CorrEtiq"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name.strip()


def parse_transinfo(lines: list[str]) -> dict[str, str]:
    for line in lines:
        match = TRANSINFO_RE.search(line)
        if match:
            return {k: v.strip() for k, v in TRANSINFO_ATTR_RE.findall(match.group(1))}
    return {}


def parse_transcript(path: Path, include_speakers: Iterable[str] = ("spk_1",)) -> Transcript:
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    info = parse_transinfo(lines)
    transcript = Transcript(
        code=extract_code(path.name),
        path=Path(path),
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
        activity_match = ACTIVITY_RE.match(line)
        if activity_match:
            if current is not None:
                transcript.activities.append(current)
            activity = canonical_activity(activity_match.group(1))
            current = Activity(activity.canonical, activity_match.group(2), activity_match.group(3))
            current_speaker = None
            continue
        speaker_match = SPEAKER_RE.match(line)
        if speaker_match:
            raw_speaker = speaker_match.group(1).lower()
            speaker_id = raw_speaker if re.search(r"\d", raw_speaker) else (current_speaker or "spk_1")
            current_speaker = speaker_id
            if current is None:
                current = Activity("UNSEGMENTED", None, None)
            if speaker_id in include_set:
                current.speaker_lines.append(speaker_match.group(2))
            continue
        if current is not None and current_speaker in include_set and not STRUCTURAL_LINE_RE.match(line):
            current.speaker_lines.append(line)

    if current is not None:
        transcript.activities.append(current)
    return transcript


def iter_transcripts(directory: Path, include_speakers: Iterable[str] = ("spk_1",), max_files: int | None = None) -> list[Transcript]:
    paths = sorted(Path(directory).glob("*.txt"))
    if max_files is not None:
        paths = paths[:max_files]
    return [parse_transcript(path, include_speakers) for path in paths]
