from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.preprocessing import canonical_activity, iter_transcripts, tokenize_segments

SPANISH_STOPWORDS = {
    "a", "al", "algo", "algunas", "algunos", "ante", "antes", "como", "con", "contra",
    "cual", "cuando", "de", "del", "desde", "donde", "durante", "e", "el", "ella",
    "ellas", "ellos", "en", "entre", "era", "eran", "eres", "es", "esa", "esas", "ese",
    "eso", "esos", "esta", "estaba", "estaban", "estado", "estamos", "estan", "están",
    "estar", "este", "esto", "estos", "estoy", "fue", "fueron", "ha", "habia", "había",
    "han", "hasta", "hay", "la", "las", "le", "les", "lo", "los", "mas", "más", "me",
    "mi", "mis", "mucho", "muy", "no", "nos", "o", "otra", "otras", "otro", "otros",
    "para", "pero", "por", "porque", "que", "qué", "se", "ser", "si", "sí", "sin",
    "sobre", "su", "sus", "tambien", "también", "te", "tenia", "tenía", "tiene",
    "tienen", "todo", "todos", "tu", "un", "una", "unas", "uno", "unos", "y", "ya",
}


def _safe_filename(value: str) -> str:
    name = re.sub(r"[^\w.\-]+", "_", str(value or "").strip(), flags=re.UNICODE)
    return name.strip("_") or "unknown"


def _segments_to_text(segments: list[list[str]]) -> str:
    return "\n".join(" ".join(segment) for segment in segments if segment).strip()


def _without_stopwords(segments: list[list[str]]) -> list[list[str]]:
    output: list[list[str]] = []
    for segment in segments:
        kept = [token for token in segment if token == "[[EE]]" or token.lower() not in SPANISH_STOPWORDS]
        if kept:
            output.append(kept)
    return output


def save_processed_tasks(
    transcripts_dir: Path,
    output_dir: Path,
    include_speakers: Iterable[str],
    valid_activities: set[int],
    lowercase: bool = True,
    save_ansi: bool = False,
    save_sw: bool = False,
    max_files: int | None = None,
) -> pd.DataFrame:
    if not save_ansi and not save_sw:
        return pd.DataFrame()

    root = Path(output_dir) / "activities_processed"
    root.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for transcript in iter_transcripts(Path(transcripts_dir), include_speakers, max_files=max_files):
        for activity_index, activity in enumerate(transcript.activities, start=1):
            act = canonical_activity(activity.name)
            if act.number is None or act.number not in valid_activities:
                continue
            segments = tokenize_segments(activity.text(), lowercase=lowercase)
            filename = f"{_safe_filename(transcript.code)}.txt"
            ansi_path = ""
            sw_path = ""
            if save_ansi:
                path = root / f"MainText-Task{act.number}-ANSI" / filename
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(_segments_to_text(segments) + "\n", encoding="utf-8")
                ansi_path = str(path)
            if save_sw:
                path = root / f"MainText-Task{act.number}-WStop" / filename
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(_segments_to_text(_without_stopwords(segments)) + "\n", encoding="utf-8")
                sw_path = str(path)
            rows.append(
                {
                    "code": transcript.code,
                    "file": transcript.path.name,
                    "activity": act.canonical,
                    "activity_number": int(act.number),
                    "activity_index": int(activity_index),
                    "segment_count": int(len(segments)),
                    "token_count": int(sum(len(segment) for segment in segments)),
                    "ansi_path": ansi_path,
                    "wstop_path": sw_path,
                }
            )

    manifest = pd.DataFrame(rows)
    manifest.to_csv(root / "processed_activities_manifest.csv", index=False)
    return manifest
