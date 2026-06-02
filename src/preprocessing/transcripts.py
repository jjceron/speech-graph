from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional

from .annotations import canonical_activity, normalize_annotations_text

TRANSINFO_RE = re.compile(r"<TransInfo\s+([^>]+)>")
TRANSINFO_ATTR_RE = re.compile(r"(\w+)=([^,>]+)")
ACTIVITY_RE = re.compile(
    r"<<\s*NombreActividad\s*=\s*([^>]+)>>\s*(?:\[\s*StartTime\s*=\s*([^\s\]]+)\s+EndTime\s*=\s*([^\]]+)\])?",
    flags=re.IGNORECASE,
)
SPEAKER_RE = re.compile(r"^(spk_\d+):\s*(.*)$", flags=re.IGNORECASE)
STRUCTURAL_LINE_RE = re.compile(r"^</?\w+|^<Speaker\b|^<Speakers>|^</Speakers>", flags=re.IGNORECASE)


@dataclass
class Activity:
    name: str
    start_time: Optional[str]
    end_time: Optional[str]
    speaker_lines: List[str] = field(default_factory=list)

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
    scribe: Optional[str] = None
    audio_filename: Optional[str] = None
    date: Optional[str] = None
    activities: List[Activity] = field(default_factory=list)

    def text(self) -> str:
        return " ".join(activity.text() for activity in self.activities if activity.text())


def extract_code(filename: str) -> str:
    name = filename
    if name.endswith(".txt"):
        name = name[:-4]
    for suffix in ("_CorrEtiq", "-CorrEtiq", " CorrEtiq"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


def parse_transinfo(lines: List[str]) -> dict[str, str]:
    for line in lines:
        match = TRANSINFO_RE.search(line)
        if not match:
            continue
        attrs = dict(TRANSINFO_ATTR_RE.findall(match.group(1)))
        return {k: v.strip() for k, v in attrs.items()}
    return {}


def parse_transcript(path: Path, include_speakers: Iterable[str]) -> Transcript:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    info = parse_transinfo(lines)
    transcript = Transcript(
        code=extract_code(path.name),
        path=path,
        scribe=info.get("scribe"),
        audio_filename=info.get("audio_filename"),
        date=info.get("date"),
    )

    include_set = {s.strip().lower() for s in include_speakers}
    current: Optional[Activity] = None
    current_speaker: Optional[str] = None

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue

        act_match = ACTIVITY_RE.match(line)
        if act_match:
            if current:
                transcript.activities.append(current)
            raw_name = act_match.group(1).strip()
            current = Activity(
                name=canonical_activity(raw_name).canonical,
                start_time=act_match.group(2),
                end_time=act_match.group(3),
            )
            current_speaker = None
            continue

        spk_match = SPEAKER_RE.match(line)
        if spk_match:
            speaker_id = spk_match.group(1).lower()
            current_speaker = speaker_id
            if current is None:
                current = Activity(name="UNSEGMENTED", start_time=None, end_time=None)
            if speaker_id in include_set:
                current.speaker_lines.append(spk_match.group(2))
            continue

        # Continuation lines are common in manually edited text files. If a line
        # does not introduce a new activity/speaker/metadata tag, attach it to
        # the previous included speaker instead of silently dropping it.
        if current is not None and current_speaker in include_set and not STRUCTURAL_LINE_RE.match(line):
            current.speaker_lines.append(line)

    if current:
        transcript.activities.append(current)

    return transcript


def iter_transcripts(
    directory: Path,
    include_speakers: Iterable[str],
    max_files: int | None = None,
) -> List[Transcript]:
    transcripts: List[Transcript] = []
    for i, path in enumerate(sorted(directory.glob("*.txt"))):
        if max_files is not None and i >= max_files:
            break
        transcripts.append(parse_transcript(path, include_speakers))
    return transcripts
