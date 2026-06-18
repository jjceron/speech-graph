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
    """
    Representation of a single activity contained in a transcript.

    An activity corresponds to a marked section of the transcript, typically
    introduced by a header such as::

        <<NombreActividad=Actividad1>> [StartTime=01:16 EndTime=02:15]

    The activity stores its name, optional timing information, and the raw
    transcript lines associated with the activity. The :meth:`text` method
    returns a normalized version of the activity text with transcript
    annotations processed by ``normalize_annotations_text``.

    Attributes:
        name: Activity identifier extracted from the activity header.
        start_time: Activity start time, if available.
        end_time: Activity end time, if available.
        raw_lines: Raw transcript lines belonging to the activity.
    """
    name: str
    start_time: str | None
    end_time: str | None
    raw_lines: list[str] = field(default_factory=list) ### Cada Activity tiene su propia lista

    def text(self) -> str:
        """
        Return the normalized text for this activity.

        All non-empty lines are stripped of leading and trailing whitespace,
        concatenated into a single string, and passed through
        ``normalize_annotations_text`` to normalize transcript annotations.

        Returns:
            A normalized text representation of the activity.
        """
        joined = " ".join(line.strip() for line in self.raw_lines if line.strip())
        return normalize_annotations_text(joined) ### La función está en annotations.py
        ### This function processes raw transcript text containing double-bracket
        ### annotations (e.g., [[EE]], [[PAUSA]], [[PS]]) and converts it into a
        ### clean, structured representation suitable for downstream linguistic analysis.


@dataclass
class Transcript:
    """
    Representation of a complete transcript and its associated metadata.

    This class acts as a data container for a transcript file, storing both
    metadata and the structured content extracted from it. Each transcript
    is identified by a unique code and may include information about the
    audio source, transcription details, and a list of segmented activities.

    Attributes:
        code: Unique identifier for the transcript or subject.
        path: File system path to the transcript file.
        scribe: Name of the person who transcribed the audio, if available.
        audio_filename: Name or identifier of the associated audio file.
        date: Date of the transcript or recording (if available).
        activities: List of Activity objects representing segmented parts
            of the transcript. Each activity contains its own text and
            temporal information.

    Example:
        transcript = Transcript(
            code="CSO-15-10B",
            path=Path("data/CSO-15-10B.txt"),
            scribe="MelisaSalazar",
            audio_filename="CSO-15-10B-YVBUAN",
            date="20052025",
            activities=[]
        )
    Transcript
    │
    ├── code
    ├── path
    ├── scribe
    ├── audio_filename
    ├── date
    │
    └── activities
        ├── Activity
        ├── Activity
        └── Activity
    """
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
    """
    Parse transcript-level metadata from the first ``<TransInfo>`` record found.

    The function scans the provided transcript lines until it finds a line
    matching the pattern::

        <TransInfo key1=value1, key2=value2, ...>

    Attribute names and values are extracted using the compiled regular
    expressions ``_TRANSINFO_RE`` and ``_TRANSINFO_ATTR_RE`` and returned
    as a dictionary. Leading and trailing whitespace is removed from each
    extracted value.

    Example:
        Input line::

            <TransInfo scribe=MelisaSalazar,
                       audio_filename=CSO-15-10B-YVBUAN,
                       date=20052025>

        Returned dictionary::

            {
                "scribe": "MelisaSalazar",
                "audio_filename": "CSO-15-10B-YVBUAN",
                "date": "20052025",
            }

    Args:
        lines: Lines from a transcript file. The ``<TransInfo>`` record is
            expected to appear near the beginning of the file before speaker
            and activity sections.

    Returns:
        A dictionary containing the metadata attributes extracted from the
        first ``<TransInfo>`` record found. Returns an empty dictionary if
        no valid ``<TransInfo>`` record is present.
    """
    for line in lines:
        match = _TRANSINFO_RE.search(line) ### Buscar información de la transcripción ::: <TransInfo scribe=MelisaSalazar, audio_filename=CSO-15-10B-YVBUAN, date=20052025>
        if match: ### Buscar coincidencia de encabezado de transcripción
            return {k: v.strip() for k, v in _TRANSINFO_ATTR_RE.findall(match.group(1))} ### Buscar metadatos en cada transcripción
    return {}


def _canonical_activity_name(raw: str) -> str:
    """
    Normalize a raw activity name into a canonical "ActividadN" format.

    This function extracts and standardizes activity identifiers from noisy
    or inconsistently formatted strings typically found in transcript
    metadata. It attempts to recover the activity number and return a
    normalized label of the form "ActividadN".

    The normalization logic follows these rules:
    - If the input contains "Actividad" followed by a number (e.g.,
      "Actividad 1", "actividad1"), it is normalized to "Actividad1".
    - If the resulting string is empty, "UNSEGMENTED" is returned.

    Args:
        raw: Raw activity name extracted from transcript text, possibly
            containing inconsistent formatting or noise.

    Returns:
        A canonical activity name in the form "ActividadN", or the cleaned
        original string if no numeric identifier is found, or "UNSEGMENTED"
        if the input is empty or invalid.

    Examples:
        _canonical_activity_name("Actividad 1")     -> "Actividad1"
        _canonical_activity_name("actividad10")     -> "Actividad10"
        _canonical_activity_name(" ")               -> "UNSEGMENTED"
    """
    m = re.search(r"Actividad\s*(\d+)", raw, flags=re.IGNORECASE)
    if m:
        return f"Actividad{m.group(1)}"
    return raw.strip() or "UNSEGMENTED"


def parse_transcript(
    filepath: str | Path,
    include_speakers: Iterable[str] = ("spk_1",),
    spk_first_only: bool = False,
) -> Transcript:
    """
    Parse a raw transcript text file into a structured Transcript object.

    This function reads a transcript file (e.g., `*-CorrEtiq.txt`) and converts
    it into a hierarchical `Transcript` object containing metadata and a list
    of `Activity` objects. It extracts transcript-level metadata, segments the
    text into activities, and groups speaker utterances within each activity.

    The parsing process includes:
    - Reading and preprocessing the raw transcript file
    - Extracting metadata from the `<TransInfo>` header
    - Creating a `Transcript` object with basic metadata (code, scribe,
      audio filename, date)
    - Detecting activity boundaries using activity markup
    - Detecting speaker lines and assigning them to the current activity
    - Optionally filtering speakers and limiting to first speaker occurrences
    - Collecting raw utterances into `Activity.raw_lines`

    Speaker filtering behavior:
        include_speakers:
            Iterable of speaker IDs to include (e.g., ["spk_1"]).
        spk_first_only:
            If True, only the first contiguous block of speech per speaker is
            retained within each activity.

    Args:
        filepath: Path to the transcript file.
        include_speakers: Iterable of speaker IDs whose utterances should be
            included in the parsed output.
        spk_first_only: If True, only keeps the first speech segment for each
            included speaker within each activity.

    Returns:
        A `Transcript` object containing:
            - Metadata extracted from the file header
            - A list of parsed `Activity` objects with grouped speaker text

    Example:
        transcript = parse_transcript(
            "data/raw/transcripts/CSO-15-10B-CorrEtiq.txt",
            include_speakers=("spk_1",),
            spk_first_only=False
        )
    """

    path = Path(filepath)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    info = _parse_transinfo(lines) ### Parse transcript-level metadata from the first ``<TransInfo>`` record found.

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
    first_done: set[str] = set()

    for raw_line in lines: ### Iterar sobre las líneas de la transcripción. Si es una línea vacía se sigue derecho.
        line = raw_line.strip()
        if not line:
            continue

        act_match = _ACTIVITY_RE.match(line) ### Identificar la actividad (Nombre de actividad, StartTime, EndTime)
        if act_match:
            if current is not None: ### Si ya existía alguna actividad entonces se guarda dicha actividad en el listado transcript.activities
                transcript.activities.append(current)
            name = _canonical_activity_name(act_match.group(1).strip())
            current = Activity(
                name=name,
                start_time=act_match.group(2),
                end_time=act_match.group(3),
            )
            current_speaker = None ### Reinicia el hablante activo
            first_done.clear() ### Reinicia el registro de quién ya habló
            continue

        spk_match = _SPEAKER_RE.match(line)
        if spk_match:
            raw_speaker = spk_match.group(1).lower()
            speaker_id = raw_speaker if re.search(r"\d", raw_speaker) else (current_speaker or "spk_1")
            current_speaker = speaker_id ### Identificar, normalizar y guardar el skp
            if current is None:
                current = Activity(name="UNSEGMENTED", start_time=None, end_time=None) ### Si hay texto pero no hay actividad. Realmente no croe que pase en las transcripciones
            if speaker_id in include_set: ### Si es un spk permitido (spk_1)
                if spk_first_only and speaker_id in first_done: ### Si spk ya hablo y es actividad de un única intervención
                    pass
                else:
                    first_done.add(speaker_id)
                    current.raw_lines.append(spk_match.group(2))
            continue

        if current is not None and current_speaker in include_set: ### Agregar otras intervenciones de spk en actividades que valga. Realmente ninguna por el moemnto. Tal vez Actividad3
            if not _STRUCTURAL_RE.match(line):
                if not (spk_first_only and current_speaker in first_done):
                    current.raw_lines.append(line)

    if current is not None: ### Garantizar que se agrega última Actividad cuando ya se recorrio todo la transcripción
        transcript.activities.append(current)

    return transcript


def load_transcript_txt(
    filepath: str | Path,
    include_speakers: Iterable[str] = ("spk_1",),
    spk_first_only: bool = False,
) -> list[dict]:
    """
    Load a transcript text file and return a simplified list of activity dictionaries.

    This function acts as a lightweight wrapper over ``parse_transcript``.
    It parses a raw transcript file into a structured ``Transcript`` object
    and then converts each activity into a flat dictionary format suitable
    for downstream analysis, machine learning pipelines, or export.

    Each returned dictionary contains:
        - name: Activity name (e.g., "Actividad1")
        - start_time: Activity start time if available
        - end_time: Activity end time if available
        - text: Cleaned and normalized textual content of the activity

    Activities with empty or non-informative text are automatically excluded.

    Args:
        filepath: Path to the transcript text file.
        include_speakers: Iterable of speaker IDs to include in the output
            (e.g., ["spk_1"]).
        spk_first_only: If True, only the first speech segment per speaker
            is retained within each activity.

    Returns:
        A list of dictionaries, each representing a parsed activity with
        its metadata and cleaned text content.

    Example:
        load_transcript_txt(
            "data/raw/transcripts/example.txt",
            include_speakers=("spk_1",),
            spk_first_only=False
        )
    """
    transcript = parse_transcript(filepath, include_speakers=include_speakers, spk_first_only=spk_first_only) ### Parse a raw transcript text file into a structured Transcript object.
    return [
        {
            "name": act.name,
            "start_time": act.start_time,
            "end_time": act.end_time,
            "text": act.text(), ### Conserva los signos de puntuación, interrogación y exclamación
        }
        for act in transcript.activities
        if act.text()
    ]
