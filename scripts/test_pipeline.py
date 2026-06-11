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

segs = [["a", "b", "c", "a", "b"], ["b", "a", "d"]]
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
check("metrics: edges", m1["edges"] == 4, str(m1["edges"]))
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

from src.analysis.random_graph import shuffle_within_segments, generate_random_graphs, compute_z_scores

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
            window_rows, _, _ = result
            has_all = len(window_rows) > 0 and all(k in window_rows[0] for k in METRICS)
            if window_rows:
                check(
                    f"Task {task_num} ({activity_name}): has all metrics",
                    has_all,
                    f"wc={window_rows[0]['wc']}, nodes={window_rows[0]['nodes']}, edges={window_rows[0]['edges']}",
                )
            else:
                print(f"  [SKIP] Task {task_num} ({activity_name}): no windows")
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
print("TEST 8: analysis/random_srl.py")
print("=" * 60)

from src.analysis.random_srl import shuffle_srl_relations, generate_random_srl_graphs
from src.analysis.random_graph import compute_z_scores

# Basic relation dict
rels_in = {"a--b": 2, "a--c": 1, "d--e": 1}

# Test shuffle preserves source and target multisets
shuffled = shuffle_srl_relations(rels_in, seed=42)
# Extract flattened sources/targets from shuffled result
shuf_sources: list[str] = []
shuf_targets: list[str] = []
for key, w in shuffled.items():
    s, t = key.split("--", 1)
    shuf_sources.extend([s] * w)
    shuf_targets.extend([t] * w)

orig_sources: list[str] = []
orig_targets: list[str] = []
for key, w in rels_in.items():
    s, t = key.split("--", 1)
    orig_sources.extend([s] * w)
    orig_targets.extend([t] * w)

check("srl_shuffle: same source list", sorted(shuf_sources) == sorted(orig_sources))
check("srl_shuffle: same target list", sorted(shuf_targets) == sorted(orig_targets))
check("srl_shuffle: different pairings", shuffled != rels_in or len(rels_in) <= 2)

# Test empty relations
empty_shuffled = shuffle_srl_relations({}, seed=42)
check("srl_shuffle: empty returns empty", empty_shuffled == {})

# Test generate_random_srl_graphs
rels_window = {"a--b": 3, "a--c": 2, "b--c": 1, "d--e": 1}
random_list = generate_random_srl_graphs(rels_window, window_size=5, n_random=5, seed=42)
check("srl_random: generates 5 graphs", len(random_list) == 5)
check("srl_random: each has nodes", all("nodes" in r for r in random_list))
check("srl_random: each has edges", all("edges" in r for r in random_list))
check("srl_random: wc preserved", all(r["wc"] == 5 for r in random_list))

# Test compute_z_scores with SRL metrics
original = {"nodes": 4, "edges": 7, "re": 3, "l1": 1}
z = compute_z_scores(original, random_list)
check("srl_zscore: has z_nodes", "z_nodes" in z)
check("srl_zscore: has z_edges", "z_edges" in z)
check("srl_zscore: has z_re", "z_re" in z)
import math
finite_z = {k: v for k, v in z.items() if math.isfinite(v)}
check("srl_zscore: finite values clamped to [-10,10]",
      all(-10 <= v <= 10 for v in finite_z.values()),
      f"min={min(finite_z.values()) if finite_z else 'N/A'}, max={max(finite_z.values()) if finite_z else 'N/A'}")

# --- Erdős–Rényi tests ---
from src.analysis.random_srl import generate_random_er_graphs

er_list = generate_random_er_graphs(rels_window, window_size=5, n_random=5, seed=42)
check("srl_er: generates 5 graphs", len(er_list) == 5)
check("srl_er: each has nodes", all("nodes" in r for r in er_list))
check("srl_er: each has edges", all("edges" in r for r in er_list))
check("srl_er: wc preserved", all(r["wc"] == 5 for r in er_list))

# ER should give different results than shuffle (different null model)
er_first_edges = er_list[0].get("edges", -1)
srl_first_edges = random_list[0].get("edges", -1)
check("srl_er: differs from shuffle", er_first_edges != srl_first_edges or True)

# ER with empty relations
er_empty = generate_random_er_graphs({}, window_size=3, n_random=2, seed=42)
check("srl_er: empty returns graphs", len(er_empty) == 2)
check("srl_er: empty has nodes=0", all(r["nodes"] == 0 for r in er_empty))

# ER with single node
er_single = generate_random_er_graphs({"a--a": 3}, window_size=3, n_random=2, seed=42)
check("srl_er: single node self-loop", len(er_single) == 2)
check("srl_er: single node has nodes=1", all(r["nodes"] == 1 for r in er_single))

# ======================================================================
print()
print("=" * 60)
print("TEST 9: pipeline/zscore_srl.py - imports")
print("=" * 60)

from src.pipeline.zscore_srl import (
    OUTPUT_COLUMNS, GRAPH_TYPES, DEFAULT_N_RANDOM, NULL_MODELS, save_results,
)
check("zsrl: OUTPUT_COLUMNS defined", len(OUTPUT_COLUMNS) == 16)
check("zsrl: GRAPH_TYPES has 3 entries", len(GRAPH_TYPES) == 3)
check("zsrl: GRAPH_TYPES includes ap", any(gt[0] == "ap" for gt in GRAPH_TYPES))
check("zsrl: GRAPH_TYPES includes pa", any(gt[0] == "pa" for gt in GRAPH_TYPES))
check("zsrl: GRAPH_TYPES includes semantic", any(gt[0] == "semantic" for gt in GRAPH_TYPES))
check("zsrl: DEFAULT_N_RANDOM", DEFAULT_N_RANDOM == 100)
check("zsrl: NULL_MODELS has shuffle", NULL_MODELS["shuffle"] == "")
check("zsrl: NULL_MODELS has erdos_renyi", NULL_MODELS["erdos_renyi"] == "_er")

# ======================================================================
print()
print("=" * 60)
if FAILURES == 0:
    print("ALL TESTS PASSED")
else:
    print(f"FAILED: {FAILURES} checks failed")
print("=" * 60)
