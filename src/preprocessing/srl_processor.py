"""SRL (Semantic Role Labeling) processor for Spanish speech transcripts."""

from __future__ import annotations

import re
from collections import defaultdict

import numpy as np
import stanza
from nltk.corpus import stopwords
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

from .annotations import BREAK_TOKEN, normalize_annotations_text

SRL_MODEL_NAME = "mbruton/gal_sp_mBERT"

PRONOUNS_DIC: dict[str, str] = {
    "yo": "yo", "mí": "yo", "me": "yo", "conmigo": "yo",
    "mi": "yo", "mío": "yo", "mía": "yo", "míos": "yo", "mías": "yo",
    "tú": "tú", "ti": "tú", "te": "tú", "contigo": "tú",
    "vos": "tú",
    "tu": "tú", "tuyo": "tú", "tuya": "tú", "tuyos": "tú", "tuyas": "tú",
    "él": "él", "ella": "ella", "usted": "usted",
    "se": "él", "lo": "él", "la": "ella", "le": "él",
    "su": "él", "suyo": "él", "suya": "él", "suyos": "él", "suyas": "él",
    "nosotros": "nosotros", "nosotras": "nosotros",
    "nos": "nosotros",
    "nuestro": "nosotros", "nuestra": "nosotros",
    "nuestros": "nosotros", "nuestras": "nosotros",
    "vosotros": "vosotros", "vosotras": "vosotros",
    "os": "vosotros",
    "vuestro": "vosotros", "vuestra": "vosotros",
    "vuestros": "vosotros", "vuestras": "vosotros",
    "ellos": "ellos", "ellas": "ellos",
    "los": "ellos", "las": "ellos", "les": "ellos",
    "esto": "esto", "este": "este", "esta": "esta",
    "estos": "estos", "estas": "estas",
    "eso": "eso", "ese": "ese", "esa": "esa",
    "esos": "esos", "esas": "esas",
    "aquello": "aquello", "aquel": "aquel", "aquella": "aquella",
    "aquellos": "aquellos", "aquellas": "aquellas",
}

def clean_text_for_srl(
    text: str,
    clean_func: str = "clean_text",
    lowercase: bool = True,
) -> str:
    """Lightweight cleaning for SRL processing.

    Unlike tokenizer.clean_text, this preserves sentence punctuation
    (. ! ?) needed for SRL sentence segmentation.

    Args:
        text: Raw transcript text.
        clean_func: ``"clean_text"`` (preserves [[EE]]) or ``"clean_text_all"`` (removes [[EE]]).
        lowercase: Lowercase the text.

    Returns:
        Cleaned text suitable for SRL sentence splitting.
    """
    value = normalize_annotations_text(text)
    value = re.sub(r"\bspk_?\d*\s*:\s*", " ", value, flags=re.IGNORECASE)
    if clean_func == "clean_text_all":
        value = re.sub(r"\[\[EE\]\]", " ", value)
    value = value.replace(BREAK_TOKEN, " ")
    value = value.replace("^", "")
    value = value.replace("<", " ").replace(">", " ")
    value = value.replace("'", "")
    if lowercase:
        value = value.lower()
    value = re.sub(r"\s+", " ", value).strip()
    return value


_srl_pipeline: pipeline | None = None
_stanza_pipeline: stanza.Pipeline | None = None
_stop_words: set[str] | None = None


def init_srl() -> tuple[stanza.Pipeline, pipeline, set[str], dict[str, str]]:
    global _stanza_pipeline, _srl_pipeline, _stop_words

    if _stanza_pipeline is None:
        _stanza_pipeline = stanza.Pipeline(
            "es", processors="tokenize,lemma", verbose=False, use_gpu=False,
        )
    if _srl_pipeline is None:
        tokenizer = AutoTokenizer.from_pretrained(SRL_MODEL_NAME)
        model = AutoModelForTokenClassification.from_pretrained(SRL_MODEL_NAME)
        _srl_pipeline = pipeline(
            "token-classification", model=model, tokenizer=tokenizer,
            aggregation_strategy="none",
        )
    if _stop_words is None:
        _stop_words = set(stopwords.words("spanish"))

    return _stanza_pipeline, _srl_pipeline, _stop_words, PRONOUNS_DIC


def split_into_sentences(text: str, nlp_stanza: stanza.Pipeline) -> list[str]:
    try:
        doc = nlp_stanza(text)
        return [s.text for s in doc.sentences]
    except Exception:
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def predict_srl(sentence: str, srl_pipe: pipeline) -> dict:
    words = sentence.split()
    word_positions: list[tuple[int, int]] = []
    search_start = 0
    for w in words:
        start = sentence.index(w, search_start)
        end = start + len(w)
        word_positions.append((start, end))
        search_start = end

    results = srl_pipe(sentence)

    word_labels = ["O"] * len(words)
    word_scores = [0.0] * len(words)

    for r in results:
        cs = r["start"]
        for i, (ws, we) in enumerate(word_positions):
            if ws <= cs < we:
                label = r["entity"]
                score = r["score"]
                if label != "O" or word_labels[i] == "O":
                    if score > word_scores[i]:
                        word_labels[i] = label
                        word_scores[i] = score
                break

    verb_groups: dict[str, dict] = {}
    for token, label in zip(words, word_labels):
        if label == "O":
            continue
        m = re.match(r"^r(\d+):(.+)$", label)
        if not m:
            continue
        vidx = m.group(1)
        role = m.group(2)
        if vidx not in verb_groups:
            verb_groups[vidx] = {"root": None, "arg0": [], "arg1": [], "arg2": []}
        if role == "root":
            verb_groups[vidx]["root"] = token
        elif role == "arg0":
            verb_groups[vidx]["arg0"].append(token)
        elif role == "arg1":
            verb_groups[vidx]["arg1"].append(token)
        elif role == "arg2":
            verb_groups[vidx]["arg2"].append(token)

    verbs_info = []
    for vidx in sorted(verb_groups.keys(), key=int):
        vg = verb_groups[vidx]
        verb = vg["root"]
        if not verb:
            continue
        desc_parts = [f"[V:{verb}]"]
        for rname in ("arg0", "arg1", "arg2"):
            if vg[rname]:
                desc_parts.append(f"[{rname.upper()}: {' '.join(vg[rname])}]")
        verbs_info.append({
            "verb": verb,
            "description": " ".join(desc_parts),
        })

    return {"verbs": verbs_info, "words": words}


def get_sentence_lemmas(sentence: str, nlp_stanza: stanza.Pipeline) -> dict[str, str]:
    try:
        doc = nlp_stanza(sentence)
        return {word.text.lower(): word.lemma for word in doc.sentences[0].words}
    except Exception:
        return {}


def lemmatize_word(word: str, sent_lemmas: dict[str, str], nlp_stanza: stanza.Pipeline) -> str:
    wl = word.lower()
    if wl in sent_lemmas:
        return sent_lemmas[wl]
    try:
        doc = nlp_stanza(word)
        return doc.sentences[0].words[0].lemma
    except Exception:
        return word


def semantic_graphs(
    sentence: str,
    semroles: dict,
    nlp_stanza: stanza.Pipeline,
    sent_lemmas: dict[str, str],
    pronouns_dic: dict[str, str],
    stop_words: set[str],
) -> tuple[dict, dict, dict, dict]:
    semantic_content: list[list] = []

    for verb_info in semroles["verbs"]:
        semantic_elements: list = []
        vlemma = lemmatize_word(verb_info["verb"], sent_lemmas, nlp_stanza)
        semantic_elements.append(vlemma)

        role_elements = re.findall(r"\[.*?\]", verb_info["description"])
        for elem in role_elements:
            inner = elem[1:-1]
            if ": " not in inner:
                continue
            parts = inner.split(": ")
            role_name = parts[0]
            words_raw = parts[1].split(" ")
            resolved = []
            for w in words_raw:
                wl = w.lower()
                if wl in pronouns_dic:
                    resolved.append(pronouns_dic[wl])
                else:
                    resolved.append(lemmatize_word(w, sent_lemmas, nlp_stanza))
            if role_name in ("ARG0", "ARG1", "ARG2"):
                semantic_elements.append([role_name, resolved])

        semantic_content.append(semantic_elements)

    sorted_segments = []
    for each in semantic_content:
        seg: list = [np.nan, np.nan, np.nan, np.nan]
        if each:
            seg[0] = each[0]
        for item in each[1:]:
            if item[0] == "ARG0":
                seg[1] = item[1]
            elif item[0] == "ARG1":
                seg[2] = item[1]
            elif item[0] == "ARG2":
                seg[3] = item[1]
        sorted_segments.append(seg)

    ap_matrix: dict[tuple[str, str], int] = {}
    pa_matrix: dict[tuple[str, str], int] = {}
    semantic_matrix: dict[tuple[str, str], int] = {}

    for each in semantic_content:
        if not each:
            continue
        predicate = each[0]
        actor: list[str] = []
        undergoer1: list[str] = []
        undergoer2: list[str] = []

        for item in each[1:]:
            if item[0] == "ARG0":
                actor = item[1]
            elif item[0] == "ARG1":
                undergoer1 = item[1]
            elif item[0] == "ARG2":
                undergoer2 = item[1]

        has_actor = bool(actor)
        has_undergoer = bool(undergoer1) or bool(undergoer2)

        # AP: actor → patient
        if has_actor and has_undergoer:
            for a in actor:
                if a in pronouns_dic or a not in stop_words:
                    if undergoer1:
                        for u in undergoer1:
                            if u in pronouns_dic or u not in stop_words:
                                k = (a, u)
                                ap_matrix[k] = ap_matrix.get(k, 0) + 1
                    if undergoer2:
                        for u in undergoer2:
                            if u in pronouns_dic or u not in stop_words:
                                k = (a, u)
                                ap_matrix[k] = ap_matrix.get(k, 0) + 1

        # PA: predicate → argument
        arguments: list[str] = []
        for a in actor:
            if a in pronouns_dic or a not in stop_words:
                arguments.append(a)
        for u in undergoer1:
            if u in pronouns_dic or u not in stop_words:
                arguments.append(u)
        for u in undergoer2:
            if u in pronouns_dic or u not in stop_words:
                arguments.append(u)

        for arg in arguments:
            k = (predicate, arg)
            pa_matrix[k] = pa_matrix.get(k, 0) + 1

    # Semantic = AP ∪ PA
    for k, v in ap_matrix.items():
        semantic_matrix[k] = semantic_matrix.get(k, 0) + v
    for k, v in pa_matrix.items():
        semantic_matrix[k] = semantic_matrix.get(k, 0) + v

    return semroles, ap_matrix, pa_matrix, semantic_matrix


def process_text_for_srl(
    text: str,
    nlp_stanza: stanza.Pipeline,
    srl_pipe: pipeline,
    stop_words: set[str],
    pronouns_dic: dict[str, str],
) -> list[dict]:
    sentences = split_into_sentences(text, nlp_stanza)
    results: list[dict] = []

    for sid, sentence in enumerate(sentences):
        sentence = sentence.strip()
        if not sentence or len(sentence.split()) < 2:
            continue

        try:
            sent_lemmas = get_sentence_lemmas(sentence, nlp_stanza)
            semroles = predict_srl(sentence, srl_pipe)
            if not semroles["verbs"]:
                continue

            _, ap_m, pa_m, sem_m = semantic_graphs(
                sentence, semroles, nlp_stanza, sent_lemmas,
                pronouns_dic, stop_words,
            )

            results.append({
                "content": sentence,
                "sentence_id": sid,
                "ap_relations": {f"{a}--{p}": c for (a, p), c in ap_m.items()},
                "pa_relations": {f"{p}--{a}": c for (p, a), c in pa_m.items()},
                "semantic_relations": {f"{x}--{y}": c for (x, y), c in sem_m.items()},
                "srl_verbs": [v["verb"] for v in semroles["verbs"]],
            })
        except Exception:
            continue

    return results
