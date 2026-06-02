from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional
import math

import pandas as pd

from src.emotions import EmotionLexicon, emotion_features, load_emotion_lexicon
from src.features import label_features
from src.graphs import compute_metrics
from src.preprocessing import canonical_activity, iter_transcripts, sliding_windows, tokenize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract speech graph metrics")
    parser.add_argument(
        "--transcripts-dir",
        default="data/processed/Transcripciones",
        help="Directory with transcript .txt files",
    )
    parser.add_argument(
        "--output-csv",
        default="outputs/graph_metrics.csv",
        help="Output CSV path",
    )
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--no-allow-short", action="store_true")
    parser.add_argument("--lowercase", action="store_true")
    parser.add_argument(
        "--include-speakers",
        default="spk_1",
        help="Comma-separated speakers to include (default: spk_1)",
    )
    parser.add_argument("--by-activity", action="store_true")
    parser.add_argument("--lexicon-path", default=None)
    parser.add_argument("--random-times", type=int, default=0)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=25, help="Print progress every N transcripts; 0 disables progress messages.")
    parser.add_argument(
        "--random-metrics",
        default="lcc,lsc,edges,repeated_edges,density,asp,l1,l2,l3",
        help="Comma-separated metrics to z-score",
    )
    return parser.parse_args()


def aggregate_rows(rows: List[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {"window_count": 0}

    df = pd.DataFrame(rows)
    mean = df.mean(numeric_only=True)
    std = df.std(numeric_only=True, ddof=0)

    output: Dict[str, float] = {"window_count": int(len(rows))}
    for col, value in mean.items():
        output[f"mean_{col}"] = float(value)
    for col, value in std.items():
        output[f"std_{col}"] = float(value)

    return output


def random_zscores(
    tokens: List[str],
    base_metrics: Dict[str, float],
    random_times: int,
    random_metrics: Iterable[str],
    seed: int,
) -> Dict[str, float]:
    if random_times <= 0 or len(tokens) < 2:
        return {}

    rng = random.Random(seed)
    values: Dict[str, List[float]] = {m: [] for m in random_metrics}

    for _ in range(random_times):
        shuffled = tokens[:]
        rng.shuffle(shuffled)
        metrics = compute_metrics(shuffled)
        for metric in random_metrics:
            if metric in metrics:
                values[metric].append(metrics[metric])

    zscores: Dict[str, float] = {}
    for metric in random_metrics:
        sample = values.get(metric, [])
        if not sample:
            zscores[f"z_{metric}"] = float("nan")
            continue
        clean_sample = [float(x) for x in sample if x is not None and math.isfinite(float(x))]
        base_value = float(base_metrics.get(metric, float("nan")))
        if not clean_sample or not math.isfinite(base_value):
            zscores[f"z_{metric}"] = float("nan")
            continue
        mean = sum(clean_sample) / len(clean_sample)
        var = sum((x - mean) ** 2 for x in clean_sample) / len(clean_sample)
        std = var ** 0.5
        if std == 0 or not math.isfinite(std):
            zscores[f"z_{metric}"] = 0.0
        else:
            zscores[f"z_{metric}"] = (base_value - mean) / std

    return zscores


def process_tokens(
    tokens: List[str],
    window_size: int,
    step: int,
    allow_short: bool,
    random_times: int,
    random_seed: int,
    random_metrics: Iterable[str],
) -> Dict[str, float]:
    rows: List[Dict[str, float]] = []
    for idx, (window_tokens, _, _) in enumerate(
        sliding_windows(tokens, window_size, step, allow_short=allow_short)
    ):
        metrics = compute_metrics(window_tokens)
        metrics.update(
            random_zscores(
                window_tokens,
                metrics,
                random_times,
                random_metrics,
                random_seed + idx,
            )
        )
        rows.append(metrics)

    output = aggregate_rows(rows)
    # Full-text graph metrics are useful for QC and sensitivity, but final
    # interpretation should usually prioritize windowed metrics to control verboseness.
    for key, value in compute_metrics(tokens).items():
        output[f"global_{key}"] = value
    return output


def extract_graph_metrics(
    transcripts_dir: Path,
    output_csv: Path,
    window_size: int = 30,
    step: int = 1,
    allow_short: bool = True,
    lowercase: bool = False,
    include_speakers: Iterable[str] = ("spk_1",),
    by_activity: bool = False,
    lexicon_path: Path | None = None,
    random_times: int = 0,
    random_seed: int = 42,
    random_metrics: Iterable[str] = ("lcc", "lsc", "edges", "repeated_edges", "density", "asp", "l1", "l2", "l3"),
    max_files: int | None = None,
    progress_every: int = 25,
) -> pd.DataFrame:
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    lexicon: Optional[EmotionLexicon] = None
    if lexicon_path:
        lexicon = load_emotion_lexicon(Path(lexicon_path))

    rows: List[Dict[str, float]] = []

    transcripts = iter_transcripts(Path(transcripts_dir), include_speakers, max_files=max_files)
    total_transcripts = len(transcripts)

    for transcript_index, transcript in enumerate(transcripts, start=1):
        if progress_every and (transcript_index == 1 or transcript_index % progress_every == 0 or transcript_index == total_transcripts):
            print(f"  transcript {transcript_index}/{total_transcripts}: {transcript.code}")
        if by_activity:
            for idx, activity in enumerate(transcript.activities, start=1):
                text = activity.text()
                tokens = tokenize(text, lowercase=lowercase)
                metrics = process_tokens(
                    tokens,
                    window_size,
                    step,
                    allow_short,
                    random_times,
                    random_seed,
                    random_metrics,
                )
                metrics.update(emotion_features(tokens, lexicon))
                metrics.update(label_features(tokens))
                act = canonical_activity(activity.name)
                row: Dict[str, float] = {
                    "code": transcript.code,
                    "file": transcript.path.name,
                    "level": "activity",
                    "activity": act.canonical,
                    "activity_number": act.number if act.number is not None else "",
                    "activity_index": idx,
                    "start_time": activity.start_time or "",
                    "end_time": activity.end_time or "",
                    "token_count": len(tokens),
                    "valid_window30": int(len(tokens) >= window_size),
                }
                row.update(metrics)
                rows.append(row)

        full_text = transcript.text()
        full_tokens = tokenize(full_text, lowercase=lowercase)
        full_metrics = process_tokens(
            full_tokens,
            window_size,
            step,
            allow_short,
            random_times,
            random_seed,
            random_metrics,
        )
        full_metrics.update(emotion_features(full_tokens, lexicon))
        full_metrics.update(label_features(full_tokens))
        full_row: Dict[str, float] = {
            "code": transcript.code,
            "file": transcript.path.name,
            "level": "file",
            "activity": "",
            "activity_number": "",
            "activity_index": 0,
            "start_time": "",
            "end_time": "",
            "token_count": len(full_tokens),
            "valid_window30": int(len(full_tokens) >= window_size),
        }
        full_row.update(full_metrics)
        rows.append(full_row)

    df = pd.DataFrame(rows)
    label_cols = [
        c for c in df.columns
        if c.startswith("label_count_") or c.startswith("label_ratio_")
    ]
    if label_cols:
        df[label_cols] = df[label_cols].fillna(0)
    df.to_csv(output_csv, index=False)
    return df


def main() -> None:
    args = parse_args()
    include_speakers = [s.strip() for s in args.include_speakers.split(",") if s.strip()]
    allow_short = not args.no_allow_short
    random_metrics = [m.strip() for m in args.random_metrics.split(",") if m.strip()]

    extract_graph_metrics(
        transcripts_dir=Path(args.transcripts_dir),
        output_csv=Path(args.output_csv),
        window_size=args.window_size,
        step=args.step,
        allow_short=allow_short,
        lowercase=args.lowercase,
        include_speakers=include_speakers,
        by_activity=args.by_activity,
        lexicon_path=Path(args.lexicon_path) if args.lexicon_path else None,
        random_times=args.random_times,
        random_seed=args.random_seed,
        random_metrics=random_metrics,
        max_files=args.max_files,
        progress_every=args.progress_every,
    )


if __name__ == "__main__":
    main()
