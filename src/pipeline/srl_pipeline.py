"""SRL graph pipeline: run SRL, build semantic graphs, compute metrics.

Usage:
    python -m src.pipeline.srl_pipeline --task 2 --window-size 3 --step 1
    python -m src.pipeline.srl_pipeline --task 7 --window-size 5 --step 2 \\
        --clean-func clean_text_all
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.graphs.sent_window import sentence_windows
from src.graphs.srl_graphs import aggregate_relations, build_srl_graph
from src.graphs.srl_metrics import METRICS, compute_srl_metrics
from src.preprocessing.loaders import load_transcript_txt
from src.preprocessing.srl_processor import (
    PRONOUNS_DIC,
    clean_text_for_srl,
    init_srl,
    process_text_for_srl,
)


TASK_ACTIVITIES = {
    2: "Actividad2",
    6: "Actividad6",
    7: "Actividad7",
}

DEFAULT_WINDOW = 3
DEFAULT_STEP = 1

GRAPH_TYPES = [
    ("ap", "ap_relations"),
    ("pa", "pa_relations"),
    ("semantic", "semantic_relations"),
]

OUTPUT_COLUMNS = [
    "file", "window", "wc", "nodes", "edges", "re", "pe", "l1", "l2", "l3",
    "lcc", "lsc", "atd", "density", "diameter", "asp", "cc",
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


def build_suffix(clean_func: str, pos_filter: bool) -> str:
    parts = []
    if clean_func == "clean_text_all":
        parts.append("all")
    if pos_filter:
        parts.append("pos")
    return "_" + "_".join(parts) if parts else ""


def process_single_subject(
    transcript_path: str | Path,
    activity_name: str,
    clean_func: str = "clean_text",
    lowercase: bool = True,
    include_speakers: tuple[str, ...] = ("spk_1",),
) -> list[dict] | None:
    """Process one transcript: load, clean, run SRL.

    Returns:
        List of per-sentence SRL results, or None if no activity found.
    """
    activities = load_transcript_txt(transcript_path, include_speakers=include_speakers)
    act = None
    for a in activities:
        if a["name"] == activity_name:
            act = a
            break
    if act is None or not act["text"].strip():
        return None

    text = clean_text_for_srl(act["text"], clean_func=clean_func, lowercase=lowercase)
    if not text.strip():
        return None

    nlp_stanza, srl_pipe, stop_words, pronouns_dic = init_srl()
    return process_text_for_srl(text, nlp_stanza, srl_pipe, stop_words, pronouns_dic)


def save_results(
    rows: list[dict],
    task_num: int,
    window_size: int,
    graph_type: str,
    output_dir: str | Path,
    suffix: str = "",
) -> None:
    """Save params_table, mean, and median files for one graph type."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tag = f"T{task_num}W{window_size}_{graph_type}{suffix}"

    df = pd.DataFrame(rows)
    cols = [c for c in OUTPUT_COLUMNS if c in df.columns]
    df = df[cols]

    params_path = output_dir / f"params_table_{tag}.txt"
    df.to_csv(params_path, index=False, sep=",")

    numeric_cols = [
        c for c in df.columns
        if c not in ("file", "window") and df[c].dtype in (int, float, np.int64, np.float64)
    ]
    mean_df = df.groupby("file")[numeric_cols].mean().reset_index()
    mean_path = output_dir / f"means_params_table_{tag}.txt"
    mean_df.to_csv(mean_path, index=False, sep=",")

    median_df = df.groupby("file")[numeric_cols].median().reset_index()
    median_path = output_dir / f"median_params_table_{tag}.txt"
    median_df.to_csv(median_path, index=False, sep=",")

    print(f"  Saved: {params_path.name} ({len(rows)} rows)")
    print(f"  Saved: {mean_path.name} ({len(mean_df)} subjects)")
    print(f"  Saved: {median_path.name} ({len(median_df)} subjects)")


def run_pipeline(
    task: int,
    windows: list[int],
    step: int = DEFAULT_STEP,
    clean_func: str = "clean_text",
    lowercase: bool = True,
    pos_filter: bool = False,
    transcripts_dir: str | Path = "data/raw/transcripts",
    metadata_path: str | Path = "data/raw/metadata.xlsx",
    srl_json_dir: str | Path = "data/raw/srl_json",
    output_dir: str | Path = "data/processed/metrics",
    include_speakers: tuple[str, ...] = ("spk_1",),
    force_srl: bool = False,
) -> None:
    """Run the full SRL pipeline for a task.

    Phase 1: Run SRL on all subjects, save per-task JSON.
    Phase 2: For each window size, build graphs and compute metrics.
    """
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
    print(f"Windows: {windows}, Step: {step}")
    print(f"Clean func: {clean_func}")
    print(f"Transcripts: {len(transcript_files)} files")
    print(f"Subjects in metadata: {len(subject_codes)}")
    print()

    srl_json_path = Path(srl_json_dir)
    srl_json_path.mkdir(parents=True, exist_ok=True)
    task_json = srl_json_path / f"Task{task}.json"
    suffix = build_suffix(clean_func, pos_filter)

    # ---- Phase 1: Run SRL ----
    if not task_json.exists() or force_srl:
        print("=" * 50)
        print("PHASE 1: Running SRL on all subjects")
        print("=" * 50)

        all_subject_data: list[dict] = []
        processed = 0
        skipped_no_activity = 0
        skipped_no_results = 0

        for filename in transcript_files:
            subject_code = get_subject_code(filename)
            if subject_code not in subject_codes:
                continue

            filepath = os.path.join(transcripts_dir, filename)
            sentence_results = process_single_subject(
                filepath, activity_name,
                clean_func=clean_func, lowercase=lowercase,
                include_speakers=include_speakers,
            )

            if sentence_results is None:
                skipped_no_activity += 1
                continue

            if not sentence_results:
                skipped_no_results += 1
                continue

            for entry in sentence_results:
                entry["uid"] = subject_code
                entry["task"] = task
            all_subject_data.extend(sentence_results)
            processed += 1

            if processed % 20 == 0:
                print(f"  SRL processed: {processed} subjects...")

        print(f"  SRL complete: {processed} subjects with results")
        print(f"  Skipped (no activity): {skipped_no_activity}")
        print(f"  Skipped (no SRL verbs): {skipped_no_results}")
        print(f"  Total sentence entries: {len(all_subject_data)}")

        with open(task_json, "w", encoding="utf-8") as f:
            json.dump(all_subject_data, f, ensure_ascii=False, indent=2)
        print(f"  Saved: {task_json}")
    else:
        print(f"Using existing SRL JSON: {task_json}")
        with open(task_json, "r", encoding="utf-8") as f:
            all_subject_data = json.load(f)

    # ---- Phase 2: Windowing → Graphs → Metrics ----
    print()
    print("=" * 50)
    print("PHASE 2: Building graphs and computing metrics")
    print("=" * 50)

    # Group entries by subject, maintaining sentence order
    by_subject: dict[str, list[dict]] = defaultdict(list)
    for entry in all_subject_data:
        uid = entry["uid"]
        by_subject[uid].append(entry)

    # Sort each subject's entries by sentence_id
    for uid in by_subject:
        by_subject[uid].sort(key=lambda e: e.get("sentence_id", 0))

    out_dir = os.path.join(output_dir, f"Task{task}")
    total_subjects = len(by_subject)

    for window_size in windows:
        print(f"\n--- Window W{window_size} ---")
        all_rows: dict[str, list[dict]] = {gt: [] for gt, _ in GRAPH_TYPES}
        total_windows = 0

        for uid, sentences in by_subject.items():
            n_sentences = len(sentences)

            for window_id, start, end, window_sents in sentence_windows(
                sentences, window_size=window_size, step=step,
            ):
                for graph_type, rel_key in GRAPH_TYPES:
                    rels = aggregate_relations(window_sents, rel_key)
                    if not rels:
                        continue
                    G = build_srl_graph(rels)
                    metrics = compute_srl_metrics(G, window_size=window_size)
                    metrics["file"] = uid
                    metrics["window"] = window_id
                    all_rows[graph_type].append(metrics)

                total_windows += 1

        print(f"  Subjects: {total_subjects}, Total windows: {total_windows}")

        for graph_type in all_rows:
            rows = all_rows[graph_type]
            if rows:
                save_results(rows, task, window_size, graph_type, out_dir, suffix=suffix)
            else:
                print(f"  {graph_type.upper()}: no data")

    print("\nDone.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SRL graph pipeline: SRL → semantic graphs → metrics",
        prog="python -m src.pipeline.srl_pipeline",
    )
    parser.add_argument(
        "--task", type=int, required=True, choices=[2, 6, 7],
        help="Activity number to process (2, 6, or 7)",
    )
    parser.add_argument(
        "--window-size", type=int, default=DEFAULT_WINDOW,
        help=f"Number of sentences per window (default: {DEFAULT_WINDOW})",
    )
    parser.add_argument(
        "--step", type=int, default=DEFAULT_STEP,
        help=f"Step size between windows (default: {DEFAULT_STEP})",
    )
    parser.add_argument(
        "--multiple-windows", type=str, default=None,
        help="Comma-separated window sizes for batch processing (e.g. 2,3,5)",
    )
    parser.add_argument(
        "--clean-func", default="clean_text", choices=["clean_text", "clean_text_all"],
        help="Cleaning function (default: clean_text)",
    )
    parser.add_argument(
        "--pos-filter", action="store_true",
        help="Apply POS tag filtering (note: applies to tokens, not SRL)",
    )
    parser.add_argument(
        "--no-lowercase", action="store_true",
        help="Do not lowercase text",
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
        "--srl-json-dir", default="data/raw/srl_json",
        help="Directory for SRL JSON output (default: data/raw/srl_json)",
    )
    parser.add_argument(
        "--output-dir", default="data/processed/metrics",
        help="Output directory for results (default: data/processed/metrics)",
    )
    parser.add_argument(
        "--include-speakers", default="spk_1",
        help="Comma-separated speaker IDs to include (default: spk_1)",
    )
    parser.add_argument(
        "--force-srl", action="store_true",
        help="Re-run SRL even if JSON cache exists",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    windows: list[int] = []
    if args.multiple_windows:
        windows = [int(w.strip()) for w in args.multiple_windows.split(",")]
    else:
        windows = [args.window_size]

    speakers = tuple(s.strip() for s in args.include_speakers.split(","))

    run_pipeline(
        task=args.task,
        windows=windows,
        step=args.step,
        clean_func=args.clean_func,
        lowercase=not args.no_lowercase,
        pos_filter=args.pos_filter,
        transcripts_dir=args.transcripts_dir,
        metadata_path=args.metadata,
        srl_json_dir=args.srl_json_dir,
        output_dir=args.output_dir,
        include_speakers=speakers,
        force_srl=args.force_srl,
    )


if __name__ == "__main__":
    main()
