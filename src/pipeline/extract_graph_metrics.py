from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.analysis import parse_csv_list
from src.graphs import MODEL_METRICS, OUTPUT_METRICS, compute_metrics, compute_metrics_from_segments
from src.preprocessing import canonical_activity, iter_transcripts, sliding_windows, tokenize_segments

DEFAULT_RANDOM_METRICS = "lcc,lsc,edges,re,pe,density,asp,cc,l1,l2,l3"


def parse_activity_window_sizes(text: str | None) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for item in parse_csv_list(text):
        if ":" not in item:
            continue
        key, value = item.split(":", 1)
        key = key.strip()
        if key.lower() != "default":
            key = canonical_activity(key).canonical
        mapping[key] = int(value.strip())
    return mapping


def random_zscores(tokens: list[str], base_metrics: dict[str, float], random_times: int, random_metrics: Iterable[str], seed: int) -> dict[str, float]:
    if random_times <= 0 or len(tokens) < 2:
        return {}
    rng = random.Random(seed)
    values: dict[str, list[float]] = {metric: [] for metric in random_metrics}
    for _ in range(random_times):
        shuffled = tokens[:]
        rng.shuffle(shuffled)
        metrics = compute_metrics(shuffled)
        for metric in random_metrics:
            if metric in metrics:
                values[metric].append(float(metrics[metric]))
    out: dict[str, float] = {}
    for metric in random_metrics:
        sample = [value for value in values.get(metric, []) if math.isfinite(float(value))]
        base = float(base_metrics.get(metric, float("nan")))
        if not sample or not math.isfinite(base):
            out[f"z_{metric}"] = float("nan")
            continue
        mean = sum(sample) / len(sample)
        std = (sum((value - mean) ** 2 for value in sample) / len(sample)) ** 0.5
        out[f"z_{metric}"] = 0.0 if std == 0 or not math.isfinite(std) else (base - mean) / std
    return out


def _aggregate_windows(rows: list[dict]) -> dict[str, float]:
    if not rows:
        return {"window_count": 0}
    df = pd.DataFrame(rows)
    keep = [col for col in OUTPUT_METRICS + [f"z_{metric}" for metric in MODEL_METRICS] if col in df.columns]
    output: dict[str, float] = {"window_count": int(len(rows))}
    for col in keep:
        values = pd.to_numeric(df[col], errors="coerce")
        output[f"mean_{col}"] = float(values.mean()) if values.notna().any() else float("nan")
        output[f"std_{col}"] = float(values.std(ddof=0)) if values.notna().any() else float("nan")
    return output


def _window_rows(segments: list[list[str]], window_size: int, step: int, allow_short: bool, random_times: int, random_seed: int, random_metrics: Iterable[str]) -> list[dict]:
    rows: list[dict] = []
    index = 0
    for segment_index, segment in enumerate(segments):
        for window_tokens, start, end in sliding_windows(segment, window_size, step, allow_short=allow_short):
            metrics = compute_metrics(window_tokens)
            metrics.update(random_zscores(window_tokens, metrics, random_times, random_metrics, random_seed + index))
            metrics.update(
                {
                    "window_index": int(index),
                    "segment_index": int(segment_index),
                    "window_start": int(start),
                    "window_end": int(end),
                    "window_size_actual": int(len(window_tokens)),
                }
            )
            rows.append(metrics)
            index += 1
    return rows


def _window_size_for_activity(activity_name: str, default_size: int, activity_windows: dict[str, int]) -> int:
    activity = canonical_activity(activity_name).canonical
    return int(activity_windows.get(activity, activity_windows.get("default", default_size)))


def _valid_activity(number: int | None, valid_activities: set[int]) -> bool:
    return number is not None and int(number) in valid_activities


def extract_graph_metrics(
    transcripts_dir: Path,
    output_csv: Path,
    window_size: int = 30,
    step: int = 1,
    allow_short: bool = False,
    lowercase: bool = True,
    include_speakers: Iterable[str] = ("spk_1",),
    valid_activities: set[int] | None = None,
    random_times: int = 0,
    random_seed: int = 42,
    random_metrics: Iterable[str] = tuple(parse_csv_list(DEFAULT_RANDOM_METRICS)),
    max_files: int | None = None,
    progress_every: int = 25,
    activity_window_sizes: dict[str, int] | None = None,
    window_rows_csv: Path | None = None,
) -> pd.DataFrame:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if window_rows_csv:
        window_rows_csv.parent.mkdir(parents=True, exist_ok=True)
    valid_activities = valid_activities or set(range(1, 8))
    activity_windows = activity_window_sizes or {}

    summary_rows: list[dict] = []
    all_window_rows: list[dict] = []
    invalid_activity_rows: list[dict] = []
    transcripts = iter_transcripts(Path(transcripts_dir), include_speakers, max_files=max_files)
    total = len(transcripts)

    for transcript_index, transcript in enumerate(transcripts, start=1):
        if progress_every and (transcript_index == 1 or transcript_index % progress_every == 0 or transcript_index == total):
            print(f"transcript {transcript_index}/{total}: {transcript.code}")
        for activity_index, activity in enumerate(transcript.activities, start=1):
            act = canonical_activity(activity.name)
            if not _valid_activity(act.number, valid_activities):
                invalid_activity_rows.append({"code": transcript.code, "file": transcript.path.name, "raw_activity": activity.name, "activity_index": activity_index})
                continue
            segments = tokenize_segments(activity.text(), lowercase=lowercase)
            tokens = [token for segment in segments for token in segment]
            current_window_size = _window_size_for_activity(activity.name, window_size, activity_windows)
            windows = _window_rows(segments, current_window_size, step, allow_short, random_times, random_seed, random_metrics)
            aggregated = _aggregate_windows(windows)
            for key, value in compute_metrics_from_segments(segments).items():
                aggregated[f"global_{key}"] = value
            row = {
                "code": transcript.code,
                "file": transcript.path.name,
                "level": "activity",
                "activity": act.canonical,
                "activity_number": int(act.number),
                "activity_index": int(activity_index),
                "start_time": activity.start_time or "",
                "end_time": activity.end_time or "",
                "window_size": int(current_window_size),
                "window_step": int(step),
                "token_count": int(len(tokens)),
                "segment_count": int(len(segments)),
                "valid_window": int(len(windows) > 0),
            }
            row.update(aggregated)
            summary_rows.append(row)
            for window in windows:
                all_window_rows.append(
                    {
                        "code": transcript.code,
                        "file": transcript.path.name,
                        "level": "activity",
                        "activity": act.canonical,
                        "activity_number": int(act.number),
                        "activity_index": int(activity_index),
                        "window_size": int(current_window_size),
                        "window_step": int(step),
                    }
                    | window
                )

    df = pd.DataFrame(summary_rows)
    df.to_csv(output_csv, index=False)

    if window_rows_csv:
        window_df = pd.DataFrame(all_window_rows)
        ordered = [
            "file", "code", "level", "activity", "activity_number", "activity_index",
            "window_size", "window_step", "window_index", "segment_index", "window_start", "window_end", "window_size_actual",
        ] + OUTPUT_METRICS
        cols = [col for col in ordered if col in window_df.columns] + [col for col in window_df.columns if col not in ordered]
        window_df = window_df[cols] if not window_df.empty else window_df
        window_df.to_csv(window_rows_csv, index=False)

    if invalid_activity_rows:
        invalid_path = output_csv.parent / "analysis" / "invalid_activities.csv"
        invalid_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(invalid_activity_rows).drop_duplicates().to_csv(invalid_path, index=False)
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract activity-level speech graph metrics")
    parser.add_argument("--transcripts-dir", default="data/processed/Transcripciones")
    parser.add_argument("--output-csv", default="outputs/graph_metrics.csv")
    parser.add_argument("--window-rows-csv", default="")
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument("--activity-window-sizes", default="")
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--allow-short", action="store_true")
    parser.add_argument("--lowercase", dest="lowercase", action="store_true", default=True)
    parser.add_argument("--preserve-case", dest="lowercase", action="store_false")
    parser.add_argument("--include-speakers", default="spk_1")
    parser.add_argument("--valid-activities", default="1,2,3,4,5,6,7")
    parser.add_argument("--random-times", type=int, default=0)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--random-metrics", default=DEFAULT_RANDOM_METRICS)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    extract_graph_metrics(
        transcripts_dir=Path(args.transcripts_dir),
        output_csv=Path(args.output_csv),
        window_rows_csv=Path(args.window_rows_csv) if args.window_rows_csv else None,
        window_size=args.window_size,
        step=args.step,
        allow_short=args.allow_short,
        lowercase=args.lowercase,
        include_speakers=parse_csv_list(args.include_speakers),
        valid_activities={int(x) for x in parse_csv_list(args.valid_activities)},
        random_times=args.random_times,
        random_seed=args.random_seed,
        random_metrics=parse_csv_list(args.random_metrics),
        max_files=args.max_files,
        progress_every=args.progress_every,
        activity_window_sizes=parse_activity_window_sizes(args.activity_window_sizes),
    )


if __name__ == "__main__":
    main()
