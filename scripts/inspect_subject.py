"""Inspect metrics for a single subject, full-text or per-window.

Usage:
    python -m scripts.inspect_subject --subject CDMS-10-4A-JURAN
    python -m scripts.inspect_subject --subject CDMS-10-4A-JURAN --task 6 --windows 30,40,50
    python -m scripts.inspect_subject --subject CDMS-10-4A-JURAN --windows 30 --step 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from src.preprocessing.loaders import load_transcript_txt
from src.preprocessing.tokenizer import tokenize_segments
from src.graphs.metrics import compute_metrics, METRICS
from src.graphs.windowing import sliding_windows

TASK_ACTIVITIES = {2: "Actividad2", 6: "Actividad6", 7: "Actividad7"}
METRIC_ORDER = ["wc", "nodes", "edges", "re", "pe", "l1", "l2", "l3",
                "lcc", "lsc", "atd", "density", "diameter", "asp", "cc"]


def fmt(v: float, width: int = 8) -> str:
    if isinstance(v, float):
        return f"{v:>{width}.4f}" if abs(v) < 1000 else f"{v:>{width}.1f}"
    return f"{v:>{width}}"


def print_full_text(tokens: list[str], m: dict) -> None:
    print(f"  N:        {m['nodes']}")
    print(f"  E:        {m['edges']}")
    print(f"  ATD:      {m['atd']:.6f}")
    print(f"  LCC:      {m['lcc']}")
    print(f"  LSC:      {m['lsc']}")
    print(f"  PE:       {m['pe']}")
    print(f"  RE:       {m['re']}")
    print(f"  L1:       {m['l1']}")
    print(f"  L2:       {m['l2']}")
    print(f"  L3:       {m['l3']}")
    print(f"  Density:  {m['density']:.6f}")
    print(f"  Diameter: {m['diameter']}")
    print(f"  ASP:      {m['asp']:.6f}")
    print(f"  CC:       {m['cc']:.6f}")


def print_windows(windows_data: list[dict]) -> None:
    rows = windows_data
    header = ["w_id"] + METRIC_ORDER
    print("  " + "  ".join(f"{h:>8}" for h in header))
    print("  " + "  ".join("-" * 8 for _ in header))

    for i, row in enumerate(rows):
        vals = [str(i)] + [fmt(row.get(k, 0)) for k in METRIC_ORDER]
        print("  " + "  ".join(f"{v:>8}" for v in vals))

    means = {k: np.mean([r.get(k, 0) for r in rows]) for k in METRIC_ORDER}
    vals = ["MEAN"] + [fmt(means.get(k, 0)) for k in METRIC_ORDER]
    print("  " + "  ".join(f"{v:>8}" for v in vals))


def inspect_subject(
    subject_code: str,
    task: int = 6,
    windows: list[int] | None = None,
    step: int = 1,
    transcripts_dir: str = "data/raw/transcripts",
    include_speakers: tuple[str, ...] = ("spk_1",),
    spk_first_only: bool = True,
    file_path: str | None = None,
) -> None:
    if file_path is not None:
        text = Path(file_path).read_text(encoding="utf-8")
        segments, segment_map = tokenize_segments(text, return_segment_map=True)
        flat_tokens = [t for seg in segments for t in seg]
        if not flat_tokens:
            print("Empty tokens")
            return

        if windows is None:
            seg_bool = [False] + [segment_map[i] != segment_map[i-1] for i in range(1, len(segment_map))]
            m = compute_metrics(flat_tokens, segment_boundaries=seg_bool)
            print(f"\nFile: {file_path} | Full text ({len(flat_tokens)} tokens)")
            print_full_text(flat_tokens, m)
        else:
            for w in windows:
                window_rows = []
                for window_tokens, start, end, boundaries in sliding_windows(
                    flat_tokens, w, step, allow_short=False, segment_boundaries=segment_map
                ):
                    m = compute_metrics(window_tokens, segment_boundaries=boundaries)
                    m["wc"] = len(window_tokens)
                    window_rows.append(m)

                print(f"\nFile: {file_path} | Window: {w}")
                print_windows(window_rows)
        return

    activity_name = TASK_ACTIVITIES.get(task)
    if activity_name is None:
        print(f"Unknown task {task}")
        return

    transcript_path = None
    for f in Path(transcripts_dir).iterdir():
        if f.stem.startswith(subject_code) and f.suffix == ".txt":
            transcript_path = f
            break

    if transcript_path is None:
        print(f"Subject '{subject_code}' not found in {transcripts_dir}")
        return

    acts = load_transcript_txt(
        transcript_path,
        include_speakers=include_speakers,
        spk_first_only=spk_first_only,
    )
    act = None
    for a in acts:
        if a["name"] == activity_name:
            act = a
            break

    if act is None:
        print(f"Activity '{activity_name}' not found for {subject_code}")
        return

    segments, segment_map = tokenize_segments(act["text"], return_segment_map=True)
    flat_tokens = [t for seg in segments for t in seg]
    if not flat_tokens:
        print("Empty tokens")
        return

    if windows is None:
        seg_bool = [False] + [segment_map[i] != segment_map[i-1] for i in range(1, len(segment_map))]
        m = compute_metrics(flat_tokens, segment_boundaries=seg_bool)
        print(f"\nSubject: {subject_code} | Task: {task} | Full text ({len(flat_tokens)} tokens)")
        print_full_text(flat_tokens, m)
    else:
        for w in windows:
            window_rows = []
            for window_tokens, start, end, boundaries in sliding_windows(
                flat_tokens, w, step, allow_short=False, segment_boundaries=segment_map
            ):
                m = compute_metrics(window_tokens, segment_boundaries=boundaries)
                m["wc"] = len(window_tokens)
                window_rows.append(m)

            print(f"\nSubject: {subject_code} | Task: {task} | Window: {w}")
            print_windows(window_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect metrics for a single subject",
    )
    parser.add_argument("--subject", "-s", required=True, help="Subject code (e.g. CDMS-10-4A-JURAN)")
    parser.add_argument("--task", "-t", type=int, default=6, choices=[2, 6, 7], help="Task / activity (default: 6)")
    parser.add_argument("--windows", "-w", type=str, default=None,
                        help="Comma-separated window sizes (default: full text)")
    parser.add_argument("--step", type=int, default=1, help="Step size for sliding windows (default: 1)")
    parser.add_argument("--transcripts-dir", default="data/raw/transcripts",
                        help="Transcripts directory (default: data/raw/transcripts)")
    parser.add_argument("--file", "-f", default=None,
                        help="Direct path to a .txt file (bypasses subject/activity lookup)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    windows = None
    if args.windows is not None:
        windows = [int(x.strip()) for x in args.windows.split(",")]
    inspect_subject(
        subject_code=args.subject,
        task=args.task,
        windows=windows,
        step=args.step,
        transcripts_dir=args.transcripts_dir,
        file_path=args.file,
    )


if __name__ == "__main__":
    main()
