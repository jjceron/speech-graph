"""Z-score pipeline: random shuffle within segments per window.

Saves z_params_table, z_means, z_median alongside raw metrics.

Usage:
    py -m src.pipeline.zscore_sg --task 2 --windows 10,20,30,40 --n_random 100
    py -m src.pipeline.zscore_sg --task 7 --windows 20,30,40,50 --n_random 100
    py -m src.pipeline.zscore_sg --task 6 --windows 30,40,50,150,160,170,180,190,200 --n_random 100
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.preprocessing.loaders import load_transcript_txt
from src.preprocessing.tokenizer import tokenize_segments
from src.graphs.metrics import compute_metrics, METRICS
from src.graphs.windowing import sliding_windows
from src.analysis.random_graph import generate_random_graphs, compute_z_scores


TASK_ACTIVITIES = {
    2: "Actividad2",
    6: "Actividad6",
    7: "Actividad7",
}


OUTPUT_COLUMNS = [
    "file", "z_wc", "z_nodes", "z_edges", "z_re", "z_pe", "z_l1", "z_l2", "z_l3",
    "z_lcc", "z_lsc", "z_atd", "z_density", "z_diameter", "z_asp", "z_cc",
]


def load_metadata(metadata_path: str | Path) -> pd.DataFrame:
    df = pd.read_excel(metadata_path)
    df["Cod"] = df["Cod"].astype(str).str.strip()
    return df


def get_subject_code(filename: str) -> str:
    name = Path(filename).stem
    for suffix in ("_CorrEtiq", "-CorrEtiq", " CorrEtiq"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name.strip()


def process_single_subject(
    transcript_path: str | Path,
    activity_name: str,
    window_size: int,
    step: int,
    n_random: int,
    seed: int,
    include_speakers: tuple[str, ...] = ("spk_1",),
    task: int | None = None,
) -> list[dict] | None:
    spk_first_only = task is not None and task in {6, 7}
    activities = load_transcript_txt(transcript_path, include_speakers=include_speakers, spk_first_only=spk_first_only)
    act = None
    for a in activities:
        if a["name"] == activity_name:
            act = a
            break
    if act is None:
        return None

    text = act["text"]
    if not text.strip():
        return None

    segments, segment_map = tokenize_segments(text, return_segment_map=True)
    flat_tokens = [t for seg in segments for t in seg]
    if not flat_tokens:
        return None

    z_rows = []
    for window_tokens, start, end, boundaries in sliding_windows(
        flat_tokens, window_size, step, allow_short=False, segment_boundaries=segment_map
    ):
        original = compute_metrics(window_tokens, segment_boundaries=boundaries)
        original["wc"] = len(window_tokens)

        random_list = generate_random_graphs(
            window_tokens, boundaries, n_random=n_random, seed=seed
        )

        zs = compute_z_scores(original, random_list)
        zs["file"] = Path(transcript_path).name
        z_rows.append(zs)

    return z_rows


def save_results(
    rows: list[dict],
    task_num: int,
    window_size: int,
    output_dir: str | Path,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tag = f"T{task_num}W{window_size}"

    df = pd.DataFrame(rows)
    cols = [c for c in OUTPUT_COLUMNS if c in df.columns]
    df = df[cols]

    params_path = output_dir / f"z_params_table{tag}.txt"
    df.to_csv(params_path, index=False, sep=",")

    numeric_cols = [c for c in df.columns if c not in ("file",) and df[c].dtype in (int, float, np.int64, np.float64)]
    mean_df = df.groupby("file")[numeric_cols].mean().reset_index()
    mean_path = output_dir / f"z_means_params_table{tag}.txt"
    mean_df.to_csv(mean_path, index=False, sep=",")

    median_df = df.groupby("file")[numeric_cols].median().reset_index()
    median_path = output_dir / f"z_median_params_table{tag}.txt"
    median_df.to_csv(median_path, index=False, sep=",")

    print(f"  Saved: {params_path.name} ({len(rows)} rows)")
    print(f"  Saved: {mean_path.name} ({len(mean_df)} subjects)")
    print(f"  Saved: {median_path.name} ({len(median_df)} subjects)")


def resolve_step(step_str: str, window_size: int) -> int:
    """Resolve step to integer: '1' -> 1, '50%' or '50percent' -> max(1, window * 50 // 100)."""
    s = step_str.strip().lower()
    if s.endswith("%"):
        pct = int(s[:-1].strip())
        return max(1, window_size * pct // 100)
    if s.endswith("percent"):
        pct = int(s[:-7].strip())
        return max(1, window_size * pct // 100)
    return int(s)


def run_pipeline(
    task: int,
    windows: list[int],
    step: str = "1",
    n_random: int = 100,
    seed: int = 42,
    transcripts_dir: str | Path = "data/raw/transcripts",
    metadata_path: str | Path = "data/raw/metadata.xlsx",
    output_dir: str | Path = "data/processed/metrics",
    include_speakers: tuple[str, ...] = ("spk_1",),
) -> None:
    activity_name = TASK_ACTIVITIES.get(task)
    if activity_name is None:
        print(f"Error: unknown task {task}. Valid tasks: {list(TASK_ACTIVITIES.keys())}")
        return

    metadata = load_metadata(metadata_path)
    subject_codes = set(metadata["Cod"].tolist())

    transcript_files = sorted(
        f for f in os.listdir(transcripts_dir) if f.endswith(".txt")
    )

    print(f"Task {task}: {activity_name}")
    print(f"Windows: {windows}")
    print(f"Step raw: {step}")
    print(f"N random: {n_random}")
    print(f"Seed: {seed}")
    print(f"Transcripts: {len(transcript_files)} files")
    print(f"Subjects in metadata: {len(subject_codes)}")
    print()

    for window_size in windows:
        resolved = resolve_step(step, window_size)
        print(f"--- Window W{window_size} (step={resolved}) ---")
        rows = []
        processed = 0

        for filename in transcript_files:
            subject_code = get_subject_code(filename)
            if subject_code not in subject_codes:
                continue

            filepath = os.path.join(transcripts_dir, filename)
            result = process_single_subject(
                filepath, activity_name, window_size, resolved, n_random, seed, include_speakers, task=task
            )
            if result is None:
                continue

            rows.extend(result)
            processed += 1

        print(f"  Processed: {processed} subjects, {len(rows)} windows")

        if rows:
            out_dir = os.path.join(output_dir, f"Task{task}")
            save_results(rows, task, window_size, out_dir)
        else:
            print(f"  No data for W{window_size}")
        print()

    print("Done.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Z-score pipeline: random shuffle per window and compute z-scores",
        prog="python -m src.pipeline.zscore_sg",
    )
    parser.add_argument(
        "--task", type=int, required=True, choices=[2, 6, 7],
        help="Activity number to process (2, 6, or 7)",
    )
    parser.add_argument(
        "--windows", type=str, required=True,
        help="Comma-separated window sizes (e.g. 10,20,30,40)",
    )
    parser.add_argument(
        "--step", type=str, default="1",
        help="Step size for sliding window: integer (e.g. 1) or percentage (e.g. 50%% or 50percent) (default: 1)",
    )
    parser.add_argument(
        "--n-random", type=int, default=100,
        help="Number of random shuffles per window (default: 100)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--transcripts-dir", default="data/raw/transcripts",
        help="Directory with transcript .txt files",
    )
    parser.add_argument(
        "--metadata", default="data/raw/metadata.xlsx",
        help="Path to metadata Excel file",
    )
    parser.add_argument(
        "--output-dir", default="data/processed/metrics",
        help="Output directory for results (same as raw metrics)",
    )
    parser.add_argument(
        "--include-speakers", default="spk_1",
        help="Comma-separated speaker IDs to include (default: spk_1)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    windows = [int(w.strip()) for w in args.windows.split(",")]
    speakers = tuple(s.strip() for s in args.include_speakers.split(","))

    run_pipeline(
        task=args.task,
        windows=windows,
        step=args.step,
        n_random=args.n_random,
        seed=args.seed,
        transcripts_dir=args.transcripts_dir,
        metadata_path=args.metadata,
        output_dir=args.output_dir,
        include_speakers=speakers,
    )


if __name__ == "__main__":
    main()
