from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

TRANSCRIPTS_DIR = BASE_DIR / "data" / "processed" / "Transcripciones"
METADATA_XLSX = BASE_DIR / "data" / "processed" / "df_dataset.xlsx"
OUTPUTS_DIR = BASE_DIR / "outputs"

SPEAKER_ID = "spk_1"
TOKEN_PATTERN = r"\[\[[^\]]+\]\]|\<[^>]+\>|\S+"
TOKEN_LOWERCASE = False

WINDOW_SIZE = 30
WINDOW_STEP = 1
ALLOW_SHORT_WINDOWS = True

RANDOM_GRAPHS = 0
RANDOM_SEED = 13
RANDOM_METRICS = ["lcc", "lsc", "edges", "repeated_edges", "density", "asp"]

LEXICON_PATH: Path | None = None
LEXICON_LOWERCASE = True
