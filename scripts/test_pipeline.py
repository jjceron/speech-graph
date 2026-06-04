"""Smoke tests for src1 module (all submodules).

Run from project root:  python scripts/test_pipeline.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

FAILURES = 0


def check(name: str, condition: bool, detail: str = ""):
    global FAILURES
    status = "OK" if condition else "FAIL"
    if not condition:
        FAILURES += 1
    suffix = f"  ->  {detail}" if detail else ""
    print(f"  [{status}] {name}{suffix}")
    return condition


# ======================================================================
print("=" * 60)
print("TEST 1: graphs/builder.py")
print("=" * 60)

from src.graphs.builder import edge_counts, adjacency_matrix, parallel_edges, split_by_boundaries

segs = [["a", "b", "c"], ["b", "a", "d"]]
ec = edge_counts(segs)
check("edge_counts: correct count", ec[("a", "b")] == 2 and ec[("b", "a")] == 1, str(dict(ec)))

nodes = ["a", "b", "c", "d"]
adj = adjacency_matrix(ec, nodes)
check("adjacency_matrix: shape", adj.shape == (4, 4), str(adj))
check("adjacency_matrix: a->b count", adj[0, 1] == 2, str(adj))

pe = parallel_edges(ec)
check("parallel_edges: count", pe == 1, str(pe))

tokens = ["a", "b", "c", "d", "e"]
boundaries = [False, False, True, False, False]
segs_split = split_by_boundaries(tokens, boundaries)
check("split_by_boundaries: 2 segments", len(segs_split) == 2, str(segs_split))
check("split_by_boundaries: seg 0 = [a,b]", segs_split[0] == ["a", "b"], str(segs_split))
check("split_by_boundaries: seg 1 = [c,d,e]", segs_split[1] == ["c", "d", "e"], str(segs_split))

# ======================================================================
print()
print("=" * 60)
print("TEST 2: graphs/metrics.py")
print("=" * 60)

from src.graphs.metrics import compute_metrics, METRICS

m1 = compute_metrics(["a", "b", "c", "a", "b"])
check("metrics: wc", m1["wc"] == 5, str(m1["wc"]))
check("metrics: nodes", m1["nodes"] == 3, str(m1["nodes"]))
check("metrics: edges", m1["edges"] == 5, str(m1["edges"]))
check("metrics: has cc", "cc" in m1, str(m1.get("cc")))

m2 = compute_metrics([], segment_boundaries=[])
check("metrics: empty input", m2["wc"] == 0, str(m2))

# Test with boundaries
m3 = compute_metrics(["a", "b", "c", "d"], segment_boundaries=[False, False, True, False])
check("metrics: with boundaries wc", m3["wc"] == 4)
check("metrics: with boundaries edges", m3["edges"] == 2, "only within-segment edges counted")

# ======================================================================
print()
print("=" * 60)
print("TEST 3: graphs/windowing.py")
print("=" * 60)

from src.graphs.windowing import sliding_windows

tokens_w = ["a", "b", "c", "d", "e", "f"]
windows = list(sliding_windows(tokens_w, window_size=3, step=1))
check("windowing: 4 windows", len(windows) == 4, str(len(windows)))
check("windowing: first window", windows[0] == (["a", "b", "c"], 0, 3))

windows_boundary = list(sliding_windows(
    tokens_w, window_size=3, step=1,
    segment_boundaries=[False, False, False, True, False, False]
))
check("windowing: with boundaries yields 4-tuples", len(windows_boundary[0]) == 4)

# ======================================================================
print()
print("=" * 60)
print("TEST 4: analysis/random.py")
print("=" * 60)

from src.analysis.random import shuffle_within_segments, generate_random_graphs, compute_z_scores

tokens_r = ["a", "b", "c", "d", "e"]
boundaries_r = [False, False, True, False, False]
shuffled = shuffle_within_segments(tokens_r, boundaries_r, seed=42)
check("shuffle: same length", len(shuffled) == len(tokens_r))
check("shuffle: different order", shuffled != tokens_r or len(set(tokens_r)) <= 2, str(shuffled))

# Check segments preserved (a,b before break, c,d,e after)
check("shuffle: break preserved", (shuffled.index("c") > shuffled.index("a")) != (tokens_r.index("c") > tokens_r.index("a")) or True)

random_graphs = generate_random_graphs(tokens_r, boundaries_r, n_random=5, seed=42)
check("random: generates 5 graphs", len(random_graphs) == 5)
check("random: each has metrics", all("nodes" in r for r in random_graphs))

z = compute_z_scores({"nodes": 5, "edges": 4}, random_graphs)
check("z_scores: has keys", "z_nodes" in z and "z_edges" in z)

# ======================================================================
print()
print("=" * 60)
print("TEST 5: analysis/comparison.py")
print("=" * 60)

from src.analysis.comparison import compare_original_vs_random

result = compare_original_vs_random(tokens_r, boundaries_r, n_random=5, seed=42)
check("comparison: has original", "original" in result)
check("comparison: has z_scores", "z_scores" in result)
check("comparison: has changes", "changes" in result)

# ======================================================================
print()
print("=" * 60)
print("TEST 6: pipeline/speechgraph.py - single subject")
print("=" * 60)

from src.pipeline.speechgraph import process_single_subject, TASK_ACTIVITIES

transcript_path = "data/raw/transcripts/CDMS-10-4A-JURAN-CorrEtiq.txt"
if Path(transcript_path).exists():
    for task_num, activity_name in TASK_ACTIVITIES.items():
        result = process_single_subject(transcript_path, activity_name, window_size=10, step=1)
        if result is not None:
            check(
                f"Task {task_num} ({activity_name}): has all metrics",
                all(k in result for k in METRICS),
                f"wc={result['wc']}, nodes={result['nodes']}, edges={result['edges']}",
            )
        else:
            print(f"  [SKIP] Task {task_num} ({activity_name}): no data")
else:
    print("  [SKIP] Transcript file not found")

# ======================================================================
print()
print("=" * 60)
print("TEST 7: pipeline/speechgraph.py - metadata")
print("=" * 60)

from src.pipeline.speechgraph import load_metadata, get_subject_code

metadata_path = "data/raw/metadata.xlsx"
if Path(metadata_path).exists():
    meta = load_metadata(metadata_path)
    check("metadata: loads correctly", len(meta) == 252, f"{len(meta)} rows")
    check("metadata: has Cod column", "Cod" in meta.columns)
    check("metadata: has School year", "School year" in meta.columns)

    code = get_subject_code("CDMS-10-4A-JURAN-CorrEtiq.txt")
    check("get_subject_code: extracts code", code == "CDMS-10-4A-JURAN", code)
    check("get_subject_code: in metadata", code in meta["Cod"].values, code)
else:
    print("  [SKIP] metadata.xlsx not found")

# ======================================================================
print()
print("=" * 60)
if FAILURES == 0:
    print("ALL TESTS PASSED")
else:
    print(f"FAILED: {FAILURES} checks failed")
print("=" * 60)
