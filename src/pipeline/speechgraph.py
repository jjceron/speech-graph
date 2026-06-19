"""Speech graph pipeline: extract metrics per task per window.

Usage:
    py -m src1.pipeline.speechgraph --task 2 --windows 10,20,30,40
    py -m src1.pipeline.speechgraph --task 6 --windows 30,40,50,150,200
    py -m src1.pipeline.speechgraph --task 7 --windows 20,30,40,50
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


# --- Task configuration ---

TASK_ACTIVITIES = {
    2: "Actividad2",
    6: "Actividad6",
    7: "Actividad7",
}

DEFAULT_WINDOWS = {
    2: [10, 20, 30, 40],
    6: [30, 40, 50, 150, 160, 170, 180, 190, 200],
    7: [20, 30, 40, 50],
}


# --- Metadata ---

def load_metadata(metadata_path: str | Path) -> pd.DataFrame:
    """Load metadata Excel file."""
    df = pd.read_excel(metadata_path)
    df["Cod"] = df["Cod"].astype(str).str.strip()
    return df


def get_subject_code(filename: str) -> str:
    """Extract subject code from transcript filename."""
    name = Path(filename).stem
    for suffix in ("_CorrEtiq", "-CorrEtiq", " CorrEtiq"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name.strip()


# --- Core processing ---

def process_single_subject(
    transcript_path: str | Path,
    activity_name: str,
    window_size: int,
    step: int,
    include_speakers: tuple[str, ...] = ("spk_1",),
    task: int | None = None,
) -> tuple[list[dict], list[list[str]], list[str]] | None:
    """Process one subject for one activity and one window size.

    Returns (window_rows, segments, flat_tokens), or None if not found.
    """
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

    window_rows = []
    for window_tokens, start, end, _ in sliding_windows(
        flat_tokens, window_size, step, allow_short=False
    ):
        m = compute_metrics(window_tokens)
        m["wc"] = len(window_tokens)
        window_rows.append(m)

    return window_rows, segments, flat_tokens


# --- Output ---

OUTPUT_COLUMNS = [
    "file", "wc", "nodes", "edges", "re", "pe", "l1", "l2", "l3",
    "lcc", "lsc", "atd", "density", "diameter", "asp", "cc",
]


def save_results(
    rows: list[dict],
    task_num: int,
    window_size: int,
    output_dir: str | Path,
) -> None:
    """Save params_table, mean (per-subject across windows), and median files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tag = f"T{task_num}W{window_size}"

    df = pd.DataFrame(rows)
    cols = [c for c in OUTPUT_COLUMNS if c in df.columns]
    df = df[cols]

    # Full params table (one row per window)
    params_path = output_dir / f"params_table{tag}.txt"
    df.to_csv(params_path, index=False, sep=",")

    # Per-subject mean across windows
    numeric_cols = [c for c in df.columns if c not in ("file",) and df[c].dtype in (int, float, np.int64, np.float64)]
    mean_df = df.groupby("file")[numeric_cols].mean().reset_index()
    mean_path = output_dir / f"means_params_table{tag}.txt"
    mean_df.to_csv(mean_path, index=False, sep=",")

    # Per-subject median across windows
    median_df = df.groupby("file")[numeric_cols].median().reset_index()
    median_path = output_dir / f"median_params_table{tag}.txt"
    median_df.to_csv(median_path, index=False, sep=",")

    print(f"  Saved: {params_path.name} ({len(rows)} rows)")
    print(f"  Saved: {mean_path.name} ({len(mean_df)} subjects)")
    print(f"  Saved: {median_path.name} ({len(median_df)} subjects)")


def save_activity_text(
    segments: list[list[str]],
    task_num: int,
    subject_code: str,
    output_dir: str | Path = "data/processed/tasks",
) -> None:
    """Save processed activity text to a file with newlines between segments."""
    out_dir = Path(output_dir) / f"MainText-Task{task_num}-ANSI"
    out_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{subject_code}.txt"
    filepath = out_dir / filename

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(" ".join(seg) for seg in segments))


# --- Main ---

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
    transcripts_dir: str | Path = "data/raw/transcripts",
    metadata_path: str | Path = "data/raw/metadata.xlsx",
    output_dir: str | Path = "data/processed/metrics",
    include_speakers: tuple[str, ...] = ("spk_1",),
    save_processed_text: bool = False,
) -> None:
    """Run the full pipeline for a task and list of window sizes."""
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
    print(f"Transcripts: {len(transcript_files)} files")
    print(f"Subjects in metadata: {len(subject_codes)}")
    print()

    if save_processed_text:
        print("Saving processed texts...")
        saved = 0
        step0 = resolve_step(step, windows[0])
        for filename in transcript_files:
            subject_code = get_subject_code(filename)
            if subject_code not in subject_codes:
                continue
            filepath = os.path.join(transcripts_dir, filename)
            result = process_single_subject(filepath, activity_name, windows[0], step0, include_speakers, task=task)
            if result is not None:
                _, segments, _ = result
                save_activity_text(segments, task, subject_code)
                saved += 1
        print(f"  Saved {saved} processed texts")
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
            result = process_single_subject(filepath, activity_name, window_size, resolved, include_speakers, task=task)
            if result is None:
                continue

            window_rows, _, _ = result
            for wr in window_rows:
                wr["file"] = filename
                rows.append(wr)
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
        description="Speech graph metrics pipeline per task and window size",
        prog="python -m src1.pipeline.speechgraph",
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
        "--transcripts-dir", default="data/raw/transcripts",
        help="Directory with transcript .txt files",
    )
    parser.add_argument(
        "--metadata", default="data/raw/metadata.xlsx",
        help="Path to metadata Excel file",
    )
    parser.add_argument(
        "--output-dir", default="data/processed/metrics",
        help="Output directory for results",
    )
    parser.add_argument(
        "--include-speakers", default="spk_1",
        help="Comma-separated speaker IDs to include (default: spk_1)",
    )
    parser.add_argument(
        "--save-processed-text", action="store_true",
        help="Save processed activity texts to data/processed/tasks/",
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
        transcripts_dir=args.transcripts_dir,
        metadata_path=args.metadata,
        output_dir=args.output_dir,
        include_speakers=speakers,
        save_processed_text=args.save_processed_text,
    )


if __name__ == "__main__":
    main()
