"""Smoke tests for src1.preprocessing module.

Run from project root:  python scripts/test_preprocessing.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing.annotations import (
    canonical_label, classify_annotation,
    normalize_annotations_text, BREAK_TOKEN,
)
from src.preprocessing.tokenizer import clean_text, tokenize, tokenize_segments
from src.preprocessing.loaders import (
    load_simple_txt, load_transcript_txt, parse_transcript,
)

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
print("TEST 1: annotations.py")
print("=" * 60)

check("canonical_label: EEE->EE", canonical_label("EEE") == "EE")
check("canonical_label: PAUA->PAUSA", canonical_label("PAUA") == "PAUSA")
check("canonical_label: ES=val->ES", canonical_label("ES=something") == "ES")
check("canonical_label: DI->DI", canonical_label("DI") == "DI")

check("classify: EE->graph", classify_annotation("EE") == "graph")
check("classify: PAUSA->break", classify_annotation("PAUSA") == "break")
check("classify: DI->break", classify_annotation("DI") == "break")
check("classify: IF->drop", classify_annotation("IF") == "drop")
check("classify: PS->drop", classify_annotation("PS") == "drop")
check("classify: unknown->unknown", classify_annotation("something") == "unknown")

n1 = normalize_annotations_text("hola [[EE]] mundo [[PAUSA]] fin")
check("normalize: [[EE]] preserved", "[[EE]]" in n1, n1)
check("normalize: [[PAUSA]]->BREAK_TOKEN", BREAK_TOKEN in n1 and "[[PAUSA]]" not in n1, n1)

n2 = normalize_annotations_text("linea uno\nlinea dos")
check("normalize: newline->BREAK_TOKEN", BREAK_TOKEN in n2, n2)

n3 = normalize_annotations_text("a b [[DI]] c d")
check("normalize: [[DI]]->BREAK_TOKEN", n3 == f"a b {BREAK_TOKEN} c d", n3)

n4 = normalize_annotations_text("a b [[IF]] c d [[PS]] e f")
check("normalize: drops [[IF]]", "[[IF]]" not in n4, n4)
check("normalize: drops [[PS]]", "[[PS]]" not in n4, n4)

n5 = normalize_annotations_text("[StartTime=01:16 EndTime=01:44] hola")
check("normalize: standalone timestamp removed", "StartTime" not in n5, n5)

# ======================================================================
print()
print("=" * 60)
print("TEST 2: tokenizer.py")
print("=" * 60)

c1 = clean_text("spk_1: Hola [[EE]] mundo (risas) ^ <test>")
check("clean: removes spk_N:", "spk_1" not in c1, c1)
check("clean: preserves [[EE]]", "[[EE]]" in c1, c1)
check("clean: removes parens", "risas" not in c1, c1)
check("clean: removes ^", "^" not in c1, c1)

tok1 = tokenize("casa a de [[PAUSA]] mesa y silla")
check("tokenize: basic", tok1[0] == "casa" and "a" in tok1, str(tok1))

tok2 = tokenize("spk_1: Hola mundo [[EE]] test")
check("tokenize: removes spk_1:", "spk_1" not in tok2, str(tok2))

segs, sm = tokenize_segments("a b [[PAUSA]] c d [[DI]] e", return_segment_map=True)
check("segments: 3 segments", len(segs) == 3, str(segs))
check("segment_map: a in seg 0", sm[0] == 0, str(sm))
check("segment_map: c in seg 1", sm[2] == 1, str(sm))
check("segment_map: e in seg 2", sm[4] == 2, str(sm))
check("segment_map: len matches tokens", len(sm) == sum(len(s) for s in segs))

# ======================================================================
print()
print("=" * 60)
print("TEST 3: loaders.py - simple dummy")
print("=" * 60)

dummy_path = Path("data/dummy/dummy_test01.txt")
if dummy_path.exists():
    dummy_text = load_simple_txt(dummy_path)
    check("load_simple: has [[PAUSA]]", "[[PAUSA]]" in dummy_text, dummy_text[:80])
    tok_d = tokenize(dummy_text)
    check("load_simple: tokenizes", len(tok_d) > 0, f"{len(tok_d)} tokens")
    segs_d, sm_d = tokenize_segments(dummy_text, return_segment_map=True)
    check("load_simple: segments exist", len(segs_d) > 0, f"{len(segs_d)} segments")
    check("load_simple: seg_map length", len(sm_d) == len(tok_d))
else:
    print("  [SKIP] dummy_test01.txt not found")

# ======================================================================
print()
print("=" * 60)
print("TEST 4: loaders.py - real transcript (CDMS-10-4A-JURAN)")
print("=" * 60)

transcript_path = Path("data/raw/transcripts/CDMS-10-4A-JURAN-CorrEtiq.txt")
if transcript_path.exists():
    activities = load_transcript_txt(transcript_path, include_speakers=("spk_1",))
    check("transcript: activities found", len(activities) == 7, f"{len(activities)} activities")
    for act in activities:
        toks = tokenize(act["text"])
        segs, sm = tokenize_segments(act["text"], return_segment_map=True)
        seg_count = len(segs)
        name = act["name"]
        check(
            f"{name}: tokens>0 and seg_map consistent",
            len(toks) > 0 and len(sm) == len(toks),
            f"{len(toks)} tokens, {seg_count} segments, seg_map={len(sm)}",
        )
        print(f"         text preview: {act['text'][:70]}...")
else:
    print(f"  [SKIP] {transcript_path} not found")

# ======================================================================
print()
print("=" * 60)
if FAILURES == 0:
    print("ALL TESTS PASSED")
else:
    print(f"FAILED: {FAILURES} checks failed")
print("=" * 60)
