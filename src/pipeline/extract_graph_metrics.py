from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.graphs import CANONICAL_METRICS, compute_metrics, compute_metrics_from_segments
from src.preprocessing import canonical_activity, iter_transcripts, sliding_windows, tokenize_segments


def parse_csv_list(text: str | None) -> list[str]:
    return [part.strip() for part in str(text or "").split(",") if part.strip()]


def parse_int_set(text: str | None) -> set[int]:
    values: set[int] = set()
    for part in parse_csv_list(text):
        activity = canonical_activity(part)
        if activity.number is not None:
            values.add(activity.number)
        else:
            try:
                values.add(int(part))
            except ValueError:
                pass
    return values


def aggregate_windows(rows: list[dict[str, float]]) -> dict[str, float]:
    out: dict[str, float] = {"window_count": int(len(rows))}
    if not rows:
        for metric in CANONICAL_METRICS:
            out[f"mean_{metric}"] = np.nan
            out[f"std_{metric}"] = np.nan
        return out
    df = pd.DataFrame(rows)
    for metric in CANONICAL_METRICS:
        values = pd.to_numeric(df.get(metric), errors="coerce")
        out[f"mean_{metric}"] = float(values.mean()) if values.notna().any() else np.nan
        out[f"std_{metric}"] = float(values.std(ddof=0)) if values.notna().any() else np.nan
    return out


def window_metric_rows(segments: list[list[str]], window_size: int, step: int, allow_short: bool) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    window_index = 0
    for segment_index, segment in enumerate(segments):
        for window_tokens, start, end in sliding_windows(segment, window_size, step, allow_short):
            metrics = compute_metrics(window_tokens)
            metrics.update(
                {
                    "window_index": int(window_index),
                    "segment_index": int(segment_index),
                    "window_start": int(start),
                    "window_end": int(end),
                    "window_size_actual": int(len(window_tokens)),
                }
            )
            rows.append(metrics)
            window_index += 1
    return rows


def process_activity(segments: list[list[str]], window_size: int, step: int, allow_short: bool) -> tuple[dict[str, float], list[dict[str, float]]]:
    windows = window_metric_rows(segments, window_size, step, allow_short)
    summary = aggregate_windows(windows)
    global_metrics = compute_metrics_from_segments(segments)
    for key, value in global_metrics.items():
        summary[f"global_{key}"] = value
    return summary, windows


def extract_graph_metrics(
    transcripts_dir: Path,
    output_csv: Path,
    window_size: int = 30,
    step: int = 1,
    allow_short: bool = False,
    lowercase: bool = True,
    include_speakers: Iterable[str] = ("spk_1",),
    max_files: int | None = None,
    progress_every: int = 25,
    valid_activities: set[int] | None = None,
    window_rows_csv: Path | None = None,
    invalid_activities_csv: Path | None = None,
) -> pd.DataFrame:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if window_rows_csv:
        window_rows_csv.parent.mkdir(parents=True, exist_ok=True)
    valid_activities = valid_activities or {1, 2, 3, 4, 5, 6, 7}

    summary_rows: list[dict] = []
    raw_window_rows: list[dict] = []
    invalid_rows: list[dict] = []
    transcripts = iter_transcripts(transcripts_dir, include_speakers, max_files=max_files)
    total = len(transcripts)

    for transcript_index, transcript in enumerate(transcripts, start=1):
        if progress_every and (transcript_index == 1 or transcript_index % progress_every == 0 or transcript_index == total):
            print(f"transcript {transcript_index}/{total}: {transcript.code}")
        for activity_index, activity in enumerate(transcript.activities, start=1):
            act = canonical_activity(activity.name)
            if act.number not in valid_activities:
                invalid_rows.append(
                    {
                        "code": transcript.code,
                        "file": transcript.path.name,
                        "raw_activity": activity.name,
                        "canonical_activity": act.canonical,
                        "activity_number": act.number if act.number is not None else "",
                        "activity_index": activity_index,
                    }
                )
                continue
            text = activity.text()
            segments = tokenize_segments(text, lowercase=lowercase)
            tokens = [token for segment in segments for token in segment]
            summary, windows = process_activity(segments, window_size, step, allow_short)
            row = {
                "code": transcript.code,
                "file": transcript.path.name,
                "level": "activity",
                "activity": act.canonical,
                "activity_number": int(act.number),
                "activity_index": int(activity_index),
                "start_time": activity.start_time or "",
                "end_time": activity.end_time or "",
                "scheme_window_size": int(window_size),
                "window_step": int(step),
                "token_count": int(len(tokens)),
                "segment_count": int(len(segments)),
                "valid_window": int(len(tokens) >= window_size and len(windows) > 0),
            }
            row.update(summary)
            summary_rows.append(row)
            for window in windows:
                window_row = {
                    "code": transcript.code,
                    "file": transcript.path.name,
                    "level": "activity",
                    "activity": act.canonical,
                    "activity_number": int(act.number),
                    "activity_index": int(activity_index),
                    "scheme_window_size": int(window_size),
                    "window_step": int(step),
                }
                window_row.update(window)
                raw_window_rows.append(window_row)

    df = pd.DataFrame(summary_rows)
    ordered = [
        "code", "file", "level", "activity", "activity_number", "activity_index", "start_time", "end_time",
        "scheme_window_size", "window_step", "token_count", "segment_count", "window_count", "valid_window",
    ]
    metric_cols = [f"mean_{m}" for m in CANONICAL_METRICS] + [f"std_{m}" for m in CANONICAL_METRICS] + [f"global_{m}" for m in CANONICAL_METRICS]
    cols = [c for c in ordered + metric_cols if c in df.columns] + [c for c in df.columns if c not in set(ordered + metric_cols)]
    df = df[cols] if not df.empty else df
    df.to_csv(output_csv, index=False)

    if window_rows_csv:
        window_df = pd.DataFrame(raw_window_rows)
        w_ordered = [
            "code", "file", "level", "activity", "activity_number", "activity_index", "scheme_window_size",
            "window_step", "window_index", "segment_index", "window_start", "window_end", "window_size_actual",
        ] + CANONICAL_METRICS
        cols = [c for c in w_ordered if c in window_df.columns] + [c for c in window_df.columns if c not in w_ordered]
        window_df = window_df[cols] if not window_df.empty else window_df
        window_df.to_csv(window_rows_csv, index=False)

    if invalid_activities_csv:
        invalid_activities_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(invalid_rows).to_csv(invalid_activities_csv, index=False)
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract activity-level speech graph metrics")
    parser.add_argument("--transcripts-dir", default="data/processed/Transcripciones")
    parser.add_argument("--output-csv", default="outputs/01_run/graph_metrics.csv")
    parser.add_argument("--window-rows-csv", default="")
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--allow-short", action="store_true")
    parser.add_argument("--lowercase", dest="lowercase", action="store_true", default=True)
    parser.add_argument("--preserve-case", dest="lowercase", action="store_false")
    parser.add_argument("--include-speakers", default="spk_1")
    parser.add_argument("--valid-activities", default="1,2,3,4,5,6,7")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_csv = Path(args.output_csv)
    extract_graph_metrics(
        transcripts_dir=Path(args.transcripts_dir),
        output_csv=output_csv,
        window_rows_csv=Path(args.window_rows_csv) if args.window_rows_csv else None,
        invalid_activities_csv=output_csv.parent / "analysis" / "invalid_activities.csv",
        window_size=args.window_size,
        step=args.step,
        allow_short=args.allow_short,
        lowercase=args.lowercase,
        include_speakers=parse_csv_list(args.include_speakers),
        valid_activities=parse_int_set(args.valid_activities),
        max_files=args.max_files,
        progress_every=args.progress_every,
    )


if __name__ == "__main__":
    main()
