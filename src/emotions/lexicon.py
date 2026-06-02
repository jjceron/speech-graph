from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Set

import pandas as pd


@dataclass
class EmotionLexicon:
    categories: Dict[str, Set[str]]

    def all_words(self) -> Set[str]:
        words: Set[str] = set()
        for values in self.categories.values():
            words.update(values)
        return words


def _load_dataframe(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=None, engine="python")
    cols = [str(c).strip().lower() for c in df.columns]
    df.columns = cols

    if "word" in cols:
        return df

    if len(cols) == 1:
        df = pd.read_csv(path, sep=None, engine="python", header=None)
        df.columns = ["word"]
        return df

    return df


def load_emotion_lexicon(path: Path) -> EmotionLexicon:
    df = _load_dataframe(path)
    cols = [str(c).lower() for c in df.columns]

    if "word" not in cols:
        raise ValueError("Lexicon must include a 'word' column or be a single-column word list")

    df["word"] = df["word"].astype(str).str.strip().str.lower()

    categories: Dict[str, Set[str]] = {}

    if "category" in cols:
        for _, row in df.iterrows():
            word = row["word"]
            category = str(row["category"]).strip().lower()
            categories.setdefault(category, set()).add(word)
        return EmotionLexicon(categories=categories)

    if "valence" in cols:
        for _, row in df.iterrows():
            word = row["word"]
            valence = float(row["valence"])
            category = "positive" if valence > 0 else "negative" if valence < 0 else "neutral"
            categories.setdefault(category, set()).add(word)
        return EmotionLexicon(categories=categories)

    if "polarity" in cols:
        for _, row in df.iterrows():
            word = row["word"]
            category = str(row["polarity"]).strip().lower()
            categories.setdefault(category, set()).add(word)
        return EmotionLexicon(categories=categories)

    categories["emotion"] = set(df["word"].tolist())
    return EmotionLexicon(categories=categories)


def emotion_features(
    tokens: Iterable[str],
    lexicon: Optional[EmotionLexicon],
    lowercase: bool = True,
) -> Dict[str, float]:
    if not lexicon:
        return {}

    token_list = list(tokens)
    if not token_list:
        return {"emotion_ratio": 0.0}

    counts: Dict[str, int] = {name: 0 for name in lexicon.categories}
    total = 0

    for token in token_list:
        key = token.lower() if lowercase else token
        matched = False
        for category, words in lexicon.categories.items():
            if key in words:
                counts[category] += 1
                matched = True
        if matched:
            total += 1

    features: Dict[str, float] = {
        "emotion_ratio": total / len(token_list),
    }

    for category, count in counts.items():
        features[f"emotion_ratio_{category}"] = count / len(token_list)

    return features
