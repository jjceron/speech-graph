#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auditoría textual de transcripciones corregidas/etiquetadas.

Uso desde la raíz del proyecto:
    py inspect_transcriptions.py --transcripts data/processed/Transcripciones --out outputs/transcription_inspect

Salida: exactamente dos archivos en la carpeta --out:
    1) transcription_inspection_summary.csv
    2) transcription_inspection_report.md

La auditoría NO modifica transcripciones. Solo inspecciona estructura, hablantes,
actividades, timestamps y etiquetas [[...]] esperadas según el protocolo.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

# -----------------------------
# Configuración de protocolo
# -----------------------------

CANONICAL_DOUBLE_BRACKET_LABELS = [
    "EE",
    "IF",
    "PS",
    "PAUSA",
    "PNC",
    "DP",
    "DI",
    "SIN_RESPUESTA",
    "SIN_PREGUNTA",
    "IM",
    "ES",
]

# Etiquetas/marcaciones no [[...]], pero esperadas por protocolo.
STRUCTURAL_MARKERS = [
    "NombreActividad",
    "ActivityTimestamp",
    "EmphasisApostrophe",
    "IncompleteWordParentheses",
    "SegmentedWordCaret",
    "ShortPauseDot",
]

# Variantes comunes observadas o esperables.
TAG_ALIASES = {
    "SIN RESPUESTA": "SIN_RESPUESTA",
    "SIN_RESPUESTA": "SIN_RESPUESTA",
    "SIN-RESPUESTA": "SIN_RESPUESTA",
    "SINRESPONSE": "SIN_RESPUESTA",
    "SIN RE PUESTA": "SIN_RESPUESTA",
    "SIN_REPUESTA": "SIN_RESPUESTA",  # typo frecuente
    "SIN REPUESTA": "SIN_RESPUESTA",  # typo frecuente
    "SIN PREGUNTA": "SIN_PREGUNTA",
    "SIN_PREGUNTA": "SIN_PREGUNTA",
    "SIN-PREGUNTA": "SIN_PREGUNTA",
    "PAUA": "PAUSA",
    "PAUSA": "PAUSA",
    "Pausa": "PAUSA",
    "pausa": "PAUSA",
    "EEE": "EE",
    "EE": "EE",
    "ee": "EE",
    "IF": "IF",
    "if": "IF",
    "PS": "PS",
    "ps": "PS",
    "PNC": "PNC",
    "pnc": "PNC",
    "DP": "DP",
    "dp": "DP",
    "DI": "DI",
    "di": "DI",
    "IM": "IM",
    "im": "IM",
}

SEVERITY_ORDER = {"error": 3, "warning": 2, "info": 1}

# -----------------------------
# Regex
# -----------------------------

RE_DOUBLE_TAG = re.compile(r"\[\[(.*?)\]\]", re.DOTALL)
RE_ACTIVITY = re.compile(r"<<\s*NombreActividad\s*=\s*Actividad\s*([0-9A-Za-z]+)\s*>>", re.IGNORECASE)
RE_ANY_ANGLE_ACTIVITY = re.compile(r"<<\s*([^<>]*NombreActividad[^<>]*)\s*>>", re.IGNORECASE)
RE_TIMESTAMP = re.compile(
    r"\[\s*StartTime\s*=\s*(\d{1,2}:\d{2})\s+EndTime\s*=\s*(\d{1,2}:\d{2})\s*\]",
    re.IGNORECASE,
)
RE_TIME_IN_TAG = re.compile(
    r"StartTime\s*[=_]\s*\d{1,2}[:_]\d{2}.*?EndTime\s*[=_]\s*\d{1,2}[:_]\d{2}",
    re.IGNORECASE,
)
RE_SPEAKER_LINE = re.compile(r"^\s*(spk_\d+)\s*:\s*(.*)$", re.IGNORECASE)
RE_GOOD_SPEAKER_LINE = re.compile(r"^\s*(spk_[01])\s*:\s*(.*)$", re.IGNORECASE)
RE_TRANSINFO = re.compile(r"<\s*TransInfo\b", re.IGNORECASE)
RE_SPEAKERS_BLOCK = re.compile(r"<\s*Speakers\b", re.IGNORECASE)
RE_SPEAKER_DEF_0 = re.compile(r"<\s*Speaker\s+id\s*=\s*spk_0\b", re.IGNORECASE)
RE_SPEAKER_DEF_1 = re.compile(r"<\s*Speaker\s+id\s*=\s*spk_1\b", re.IGNORECASE)
RE_MALFORMED_OPEN = re.compile(r"\[\[[^\]]*$")
RE_MALFORMED_CLOSE = re.compile(r"^[^\[]*\]\]")
RE_PAREN_INCOMPLETE = re.compile(r"\b\w*\([A-Za-zÁÉÍÓÚáéíóúÑñ]+\)\w*\b")
RE_SEGMENTED = re.compile(r"\b\w+\^\w+\b")
RE_SHORT_PAUSE = re.compile(r"\(\.\)")
RE_ES_TAG = re.compile(r"^ES\s*=")
RE_WORD = re.compile(r"\b[\wÁÉÍÓÚÜÑáéíóúüñ]+\b", re.UNICODE)

# -----------------------------
# Utilidades
# -----------------------------

def normalize_code_from_filename(path: Path) -> str:
    stem = path.stem
    stem = re.sub(r"[_-]?CorrEtiq$", "", stem, flags=re.IGNORECASE)
    return stem.strip()


def clean_tag_text(raw: str) -> str:
    txt = raw.strip()
    txt = re.sub(r"\s+", " ", txt)
    return txt


def canonicalize_tag(raw: str) -> Tuple[str, str, bool]:
    """Devuelve (canonical, base_detected, is_variant)."""
    txt = clean_tag_text(raw)

    # ES=texto
    if RE_ES_TAG.match(txt):
        return "ES", "ES", False

    # Quitar timestamps para obtener base.
    no_time = re.sub(r"StartTime\s*[=_]\s*\d{1,2}[:_]\d{2}", "", txt, flags=re.IGNORECASE)
    no_time = re.sub(r"EndTime\s*[=_]\s*\d{1,2}[:_]\d{2}", "", no_time, flags=re.IGNORECASE)
    no_time = re.sub(r"[_\-]+", " ", no_time)
    base = no_time.strip()
    base_first = base.split()[0] if base else txt

    # Casos multi-palabra.
    upper_base = base.upper().replace("-", "_")
    upper_txt = txt.upper().replace("-", "_")
    if "SIN" in upper_base and "RESP" in upper_base:
        canonical = "SIN_RESPUESTA"
        variant = upper_base not in {"SIN RESPUESTA", "SIN_RESPUESTA"}
        return canonical, base, variant
    if "SIN" in upper_base and "PREG" in upper_base:
        canonical = "SIN_PREGUNTA"
        variant = upper_base not in {"SIN PREGUNTA", "SIN_PREGUNTA"}
        return canonical, base, variant

    # PAUSA/DP/DI con tiempo y posibles underscores.
    for prefix in ["PAUSA", "DP", "DI"]:
        if upper_txt.startswith(prefix):
            return prefix, base_first, base_first.upper() != prefix

    # Alias simples.
    if txt in TAG_ALIASES:
        canonical = TAG_ALIASES[txt]
        return canonical, txt, canonical != txt
    if base_first in TAG_ALIASES:
        canonical = TAG_ALIASES[base_first]
        return canonical, base_first, canonical != base_first

    # Directo.
    upper_first = base_first.upper()
    if upper_first in CANONICAL_DOUBLE_BRACKET_LABELS:
        return upper_first, base_first, base_first != upper_first

    return "UNKNOWN", base_first or txt, False


def has_valid_time_inside_tag(raw: str) -> bool:
    return bool(RE_TIME_IN_TAG.search(raw))


def words_outside_tags(text: str) -> List[str]:
    no_double = RE_DOUBLE_TAG.sub(" ", text)
    no_time = RE_TIMESTAMP.sub(" ", no_double)
    no_activity = RE_ANY_ANGLE_ACTIVITY.sub(" ", no_time)
    return RE_WORD.findall(no_activity)


def line_number_from_offset(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def add_issue(issues: List[dict], severity: str, issue_type: str, message: str, line: int | None = None, raw: str = "", fix: str = "") -> None:
    issues.append({
        "severity": severity,
        "issue_type": issue_type,
        "message": message,
        "line": line or "",
        "raw": raw[:220].replace("\n", " "),
        "suggested_fix": fix,
    })


def summarize_issues(issues: List[dict], max_items: int = 8) -> str:
    if not issues:
        return ""
    sorted_issues = sorted(issues, key=lambda x: (-SEVERITY_ORDER.get(x["severity"], 0), str(x["issue_type"]), str(x["line"])))
    compact = []
    for issue in sorted_issues[:max_items]:
        line = f"L{issue['line']}: " if issue.get("line") else ""
        compact.append(f"{issue['severity']}:{issue['issue_type']}:{line}{issue['message']}")
    if len(sorted_issues) > max_items:
        compact.append(f"... +{len(sorted_issues)-max_items} issues")
    return " | ".join(compact)

# -----------------------------
# Inspección por archivo
# -----------------------------

def inspect_file(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    code = normalize_code_from_filename(path)
    issues: List[dict] = []

    # Header
    has_transinfo = bool(RE_TRANSINFO.search(text))
    has_speakers = bool(RE_SPEAKERS_BLOCK.search(text))
    has_spk0_def = bool(RE_SPEAKER_DEF_0.search(text))
    has_spk1_def = bool(RE_SPEAKER_DEF_1.search(text))

    if not has_transinfo:
        add_issue(issues, "warning", "missing_header", "No se encontró <TransInfo ...>.", fix="Agregar cabezote TransInfo según protocolo.")
    if not has_speakers:
        add_issue(issues, "warning", "missing_speakers_block", "No se encontró bloque <Speakers>.", fix="Agregar bloque Speakers con spk_0 y spk_1.")
    if not has_spk0_def:
        add_issue(issues, "warning", "missing_spk0_definition", "No se encontró definición <Speaker id=spk_0 .../>.")
    if not has_spk1_def:
        add_issue(issues, "warning", "missing_spk1_definition", "No se encontró definición <Speaker id=spk_1 .../>.")
    if "corretiq" not in path.stem.lower():
        add_issue(issues, "info", "filename_suffix", "El nombre del archivo no contiene CorrEtiq.", fix="Verificar si el archivo final debería llevar sufijo CorrEtiq.")

    # Speakers y texto por speaker.
    speaker_counts = Counter()
    speaker_chars = Counter()
    unknown_speaker_lines = 0
    spk1_text_parts = []
    spk0_text_parts = []

    for i, line in enumerate(lines, start=1):
        m_any = RE_SPEAKER_LINE.match(line)
        if m_any:
            spk = m_any.group(1).lower()
            content = m_any.group(2)
            speaker_counts[spk] += 1
            speaker_chars[spk] += len(content)
            if spk == "spk_1":
                spk1_text_parts.append(content)
            elif spk == "spk_0":
                spk0_text_parts.append(content)
            elif not RE_GOOD_SPEAKER_LINE.match(line):
                unknown_speaker_lines += 1
                add_issue(issues, "error", "unknown_speaker", f"Línea con hablante no esperado: {spk}.", line=i, raw=line, fix="Usar spk_0 para entrevistador y spk_1 para entrevistado.")

    spk1_text = "\n".join(spk1_text_parts)
    spk0_text = "\n".join(spk0_text_parts)
    spk1_tokens = words_outside_tags(spk1_text)
    spk0_tokens = words_outside_tags(spk0_text)

    if speaker_counts.get("spk_1", 0) == 0:
        add_issue(issues, "error", "missing_spk1_lines", "No se encontraron líneas spk_1.")
    if len(spk1_tokens) < 30:
        add_issue(issues, "warning", "low_spk1_tokens", f"Texto de spk_1 menor a 30 tokens: {len(spk1_tokens)}.", fix="Verificar si hay suficiente verbalización para ventana w30.")
    if len(spk0_tokens) > 0 and len(spk1_tokens) > 0:
        spk0_ratio = len(spk0_tokens) / max(1, len(spk0_tokens) + len(spk1_tokens))
        if spk0_ratio > 0.35:
            add_issue(issues, "warning", "high_spk0_ratio", f"Alta proporción de tokens spk_0: {spk0_ratio:.2%}.", fix="Revisar si quedaron verbalizaciones irrelevantes del entrevistador.")
    else:
        spk0_ratio = 0.0

    # Actividades y timestamps.
    activities = RE_ACTIVITY.findall(text)
    any_activity_tags = RE_ANY_ANGLE_ACTIVITY.findall(text)
    if not any_activity_tags:
        add_issue(issues, "warning", "missing_activity_tags", "No se encontraron etiquetas <<NombreActividad=Actividad#>>.")
    malformed_activity_count = max(0, len(any_activity_tags) - len(activities))
    if malformed_activity_count:
        add_issue(issues, "warning", "malformed_activity_tag", f"Hay {malformed_activity_count} etiquetas NombreActividad con formato no estándar.")

    timestamps = RE_TIMESTAMP.findall(text)
    activity_lines = []
    for i, line in enumerate(lines, start=1):
        if RE_ANY_ANGLE_ACTIVITY.search(line):
            activity_lines.append((i, line))
            if not RE_TIMESTAMP.search(line):
                add_issue(issues, "warning", "activity_without_timestamp_same_line", "Actividad sin timestamp en la misma línea.", line=i, raw=line, fix="Ubicar [StartTime=MM:SS EndTime=MM:SS] junto a NombreActividad cuando aplique.")

    # Etiquetas [[...]]
    raw_tags = []
    canonical_counts = Counter()
    raw_tag_counts = Counter()
    variants = Counter()
    unknown_tags = Counter()
    tag_time_missing_counts = Counter()

    for m in RE_DOUBLE_TAG.finditer(text):
        raw = clean_tag_text(m.group(1))
        line = line_number_from_offset(text, m.start())
        canonical, base, is_variant = canonicalize_tag(raw)
        raw_tags.append(raw)
        raw_tag_counts[raw] += 1
        canonical_counts[canonical] += 1

        if canonical == "UNKNOWN":
            unknown_tags[raw] += 1
            add_issue(issues, "warning", "unknown_double_bracket_tag", f"Etiqueta [[...]] no reconocida: [[{raw}]].", line=line, raw=f"[[{raw}]]", fix="Comparar contra listado canónico del protocolo.")
        if is_variant:
            variants[f"{raw} -> {canonical}"] += 1
            add_issue(issues, "info", "tag_variant", f"Variante normalizable: [[{raw}]] -> {canonical}.", line=line, raw=f"[[{raw}]]")

        if canonical in {"PAUSA", "DP"} and not has_valid_time_inside_tag(raw):
            tag_time_missing_counts[canonical] += 1
            add_issue(issues, "warning", "tag_missing_time", f"[[{canonical}]] sin StartTime/EndTime.", line=line, raw=f"[[{raw}]]", fix=f"Usar [[{canonical} StartTime=MM:SS EndTime=MM:SS]] si dura más de 3 segundos.")
        if canonical == "ES" and not RE_ES_TAG.match(raw):
            add_issue(issues, "warning", "es_format", f"Evento simple sin formato ES=texto: [[{raw}]].", line=line, raw=f"[[{raw}]]", fix="Usar [[ES=acción en tercera persona singular]].")

    # Malformaciones de corchetes.
    open_count = text.count("[[")
    close_count = text.count("]]")
    malformed_tag_count = 0
    if open_count != close_count:
        malformed_tag_count += abs(open_count - close_count)
        add_issue(issues, "error", "unbalanced_double_brackets", f"Cantidad de [[ ({open_count}) distinta de ]] ({close_count}).", fix="Revisar etiquetas abiertas/cerradas.")
    # Patrones comunes de etiqueta rota por línea.
    for i, line in enumerate(lines, start=1):
        if "[[" in line or "]]" in line:
            if line.count("[[") != line.count("]] ") and line.count("[[") != line.count("]]"):
                # Evita duplicar demasiado, solo marca casos obvios.
                if RE_MALFORMED_OPEN.search(line) or RE_MALFORMED_CLOSE.search(line):
                    malformed_tag_count += 1
                    add_issue(issues, "warning", "possibly_malformed_tag_line", "Posible etiqueta mal cerrada en línea.", line=i, raw=line)

    # Reglas contextuales simples para SIN_RESPUESTA.
    for i, line in enumerate(lines, start=1):
        if "SIN" in line.upper() and "RESP" in line.upper() and "[[" in line:
            m_spk = RE_SPEAKER_LINE.match(line)
            content = m_spk.group(2) if m_spk else line
            words = words_outside_tags(content)
            # No contar StartTime/EndTime; si hay más de 3 palabras reales fuera de etiquetas, sospechoso.
            words_clean = [w for w in words if w.lower() not in {"starttime", "endtime"}]
            if len(words_clean) > 3:
                add_issue(issues, "warning", "sin_respuesta_with_extra_text", "SIN_RESPUESTA aparece junto con texto adicional en la misma línea.", line=i, raw=line, fix="Según protocolo, si no hay respuesta, la línea debería quedar esencialmente como [[SIN RESPUESTA]].")

    # Marcaciones no [[...]]
    incomplete_count = len(RE_PAREN_INCOMPLETE.findall(text))
    segmented_count = len(RE_SEGMENTED.findall(text))
    short_pause_count = len(RE_SHORT_PAUSE.findall(text))
    apostrophe_emphasis_count = len(re.findall(r"\w'\w", text))

    # Score simple de estado.
    error_count = sum(1 for x in issues if x["severity"] == "error")
    warning_count = sum(1 for x in issues if x["severity"] == "warning")
    info_count = sum(1 for x in issues if x["severity"] == "info")
    if error_count > 0:
        qc_status = "REVISAR_ERROR"
    elif warning_count >= 5:
        qc_status = "REVISAR_ADVERTENCIAS"
    elif warning_count > 0:
        qc_status = "OK_CON_ADVERTENCIAS"
    else:
        qc_status = "OK"

    row = {
        "code": code,
        "file": str(path),
        "qc_status": qc_status,
        "error_count": error_count,
        "warning_count": warning_count,
        "info_count": info_count,
        "issue_summary": summarize_issues(issues),
        "issues_json": json.dumps(issues, ensure_ascii=False),
        "chars": len(text),
        "lines": len(lines),
        "has_transinfo": int(has_transinfo),
        "has_speakers_block": int(has_speakers),
        "has_spk0_definition": int(has_spk0_def),
        "has_spk1_definition": int(has_spk1_def),
        "spk0_lines": speaker_counts.get("spk_0", 0),
        "spk1_lines": speaker_counts.get("spk_1", 0),
        "unknown_speaker_lines": unknown_speaker_lines,
        "spk0_token_count": len(spk0_tokens),
        "spk1_token_count": len(spk1_tokens),
        "spk0_token_ratio": round(spk0_ratio, 4),
        "activity_count": len(activities),
        "activities_detected": ";".join(activities),
        "malformed_activity_count": malformed_activity_count,
        "timestamp_count": len(timestamps),
        "double_bracket_tag_count": len(raw_tags),
        "unknown_tag_count": sum(unknown_tags.values()),
        "malformed_tag_count": malformed_tag_count,
        "tag_variants_count": sum(variants.values()),
        "tag_variants": "; ".join(f"{k} ({v})" for k, v in variants.most_common()),
        "unknown_tags": "; ".join(f"{k} ({v})" for k, v in unknown_tags.most_common()),
        "raw_tags_top": "; ".join(f"{k} ({v})" for k, v in raw_tag_counts.most_common(20)),
        "incomplete_word_parentheses_count": incomplete_count,
        "segmented_word_caret_count": segmented_count,
        "short_pause_dot_count": short_pause_count,
        "apostrophe_emphasis_count": apostrophe_emphasis_count,
    }

    for lab in CANONICAL_DOUBLE_BRACKET_LABELS:
        row[f"tag_{lab}_count"] = canonical_counts.get(lab, 0)
    for lab in ["PAUSA", "DP"]:
        row[f"tag_{lab}_missing_time_count"] = tag_time_missing_counts.get(lab, 0)

    return row

# -----------------------------
# Reporte agregado
# -----------------------------

def write_csv(rows: List[dict], path: Path) -> None:
    if not rows:
        fieldnames = ["code", "file", "qc_status"]
    else:
        # Mantener orden del primer row y agregar cualquier columna extra.
        fieldnames = list(rows[0].keys())
        extras = sorted({k for r in rows for k in r.keys()} - set(fieldnames))
        fieldnames.extend(extras)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def pct(x: int, n: int) -> str:
    return "0.0%" if n == 0 else f"{100*x/n:.1f}%"


def write_md(rows: List[dict], path: Path, transcripts_dir: Path) -> None:
    n = len(rows)
    status_counts = Counter(r["qc_status"] for r in rows)
    total_errors = sum(int(r["error_count"]) for r in rows)
    total_warnings = sum(int(r["warning_count"]) for r in rows)
    total_infos = sum(int(r["info_count"]) for r in rows)

    tag_totals = Counter()
    tag_file_nonzero = Counter()
    for r in rows:
        for lab in CANONICAL_DOUBLE_BRACKET_LABELS:
            c = int(r.get(f"tag_{lab}_count", 0) or 0)
            tag_totals[lab] += c
            if c > 0:
                tag_file_nonzero[lab] += 1

    # Issue type counts desde issues_json.
    issue_type_counts = Counter()
    issue_examples = defaultdict(list)
    variant_counts = Counter()
    unknown_counts = Counter()
    for r in rows:
        try:
            issues = json.loads(r.get("issues_json", "[]"))
        except Exception:
            issues = []
        for issue in issues:
            key = f"{issue.get('severity')}::{issue.get('issue_type')}"
            issue_type_counts[key] += 1
            if len(issue_examples[key]) < 5:
                issue_examples[key].append((r["code"], issue.get("line", ""), issue.get("message", "")))
        for item in str(r.get("tag_variants", "")).split("; "):
            if item.strip():
                variant_counts[item.strip()] += 1
        for item in str(r.get("unknown_tags", "")).split("; "):
            if item.strip():
                unknown_counts[item.strip()] += 1

    # Top files by warnings/errors.
    rows_sorted = sorted(rows, key=lambda r: (int(r["error_count"]), int(r["warning_count"]), int(r["tag_variants_count"]), int(r["unknown_tag_count"])), reverse=True)

    lines = []
    lines.append("# Inspección de transcripciones corregidas y etiquetadas\n")
    lines.append(f"**Fecha de ejecución:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append(f"**Carpeta inspeccionada:** `{transcripts_dir}`\n")
    lines.append("**Salida generada:** este reporte Markdown y `transcription_inspection_summary.csv`.\n")

    lines.append("## 1. Etiquetas esperadas según protocolo\n")
    lines.append("### Etiquetas entre doble corchete `[[...]]`\n")
    lines.append("- `[[EE]]`: elementos extralexicales.\n")
    lines.append("- `[[IF]]`: inicio falso.\n")
    lines.append("- `[[PS]]`: prolongación de sonidos.\n")
    lines.append("- `[[PAUSA StartTime=MM:SS EndTime=MM:SS]]`: pausa de 3 segundos o más. Para pausas menores se usa `(.)`.\n")
    lines.append("- `[[PNC]]`: palabra no clara.\n")
    lines.append("- `[[DP StartTime=MM:SS EndTime=MM:SS]]`: diálogo entre las partes.\n")
    lines.append("- `[[DI StartTime=MM:SS EndTime=MM:SS]]` o `[[DI]]`: diálogo interno; suele acompañarse de texto entre `<...>`.\n")
    lines.append("- `[[SIN RESPUESTA]]`: ausencia de respuesta del entrevistado.\n")
    lines.append("- `[[SIN PREGUNTA]]`: pregunta/estímulo omitido por el entrevistador.\n")
    lines.append("- `[[IM]]`: inquietud motora.\n")
    lines.append("- `[[ES=texto]]`: evento simple producido por el entrevistado.\n")
    lines.append("\n### Marcaciones estructurales o no encerradas en `[[...]]`\n")
    lines.append("- `<<NombreActividad=Actividad#>>`: inicio de actividad.\n")
    lines.append("- `[StartTime=MM:SS EndTime=MM:SS]`: marca de tiempo de actividad o fragmento.\n")
    lines.append("- Apóstrofo `'`: acentuación de sonido o sílaba.\n")
    lines.append("- Paréntesis dentro de palabra, por ejemplo `(en)tonces` o `pu(d)o`: palabra incompleta.\n")
    lines.append("- Circunflejo `^`, por ejemplo `car^nero`: término segmentado.\n")

    lines.append("## 2. Resumen general\n")
    lines.append(f"- Archivos `.txt` inspeccionados: **{n}**.\n")
    lines.append(f"- Total de errores: **{total_errors}**.\n")
    lines.append(f"- Total de advertencias: **{total_warnings}**.\n")
    lines.append(f"- Total de notas informativas: **{total_infos}**.\n")
    lines.append("\n### Estado QC por archivo\n")
    lines.append("| Estado | n | % |\n|---|---:|---:|\n")
    for status in ["OK", "OK_CON_ADVERTENCIAS", "REVISAR_ADVERTENCIAS", "REVISAR_ERROR"]:
        c = status_counts.get(status, 0)
        lines.append(f"| {status} | {c} | {pct(c, n)} |\n")

    lines.append("\n## 3. Inventario de etiquetas canónicas\n")
    lines.append("| Etiqueta | apariciones totales | archivos con etiqueta | % archivos |\n|---|---:|---:|---:|\n")
    for lab in CANONICAL_DOUBLE_BRACKET_LABELS:
        lines.append(f"| {lab} | {tag_totals[lab]} | {tag_file_nonzero[lab]} | {pct(tag_file_nonzero[lab], n)} |\n")

    lines.append("\n## 4. Principales tipos de alerta\n")
    if issue_type_counts:
        lines.append("| Severidad / tipo | n | ejemplos |\n|---|---:|---|\n")
        for key, c in issue_type_counts.most_common(20):
            examples = "; ".join(f"{code} L{line}: {msg}" for code, line, msg in issue_examples[key])
            lines.append(f"| {key} | {c} | {examples} |\n")
    else:
        lines.append("No se detectaron alertas.\n")

    lines.append("\n## 5. Variantes y etiquetas desconocidas\n")
    lines.append("### Variantes normalizables detectadas\n")
    if variant_counts:
        for item, c in variant_counts.most_common(30):
            lines.append(f"- {item}: {c}\n")
    else:
        lines.append("- No se detectaron variantes normalizables.\n")

    lines.append("\n### Etiquetas desconocidas `[[...]]`\n")
    if unknown_counts:
        for item, c in unknown_counts.most_common(30):
            lines.append(f"- {item}: {c}\n")
    else:
        lines.append("- No se detectaron etiquetas desconocidas.\n")

    lines.append("\n## 6. Archivos que más requieren revisión\n")
    lines.append("| code | status | errores | advertencias | variantes | desconocidas | resumen |\n|---|---|---:|---:|---:|---:|---|\n")
    for r in rows_sorted[:30]:
        if int(r["error_count"]) == 0 and int(r["warning_count"]) == 0 and int(r["tag_variants_count"]) == 0 and int(r["unknown_tag_count"]) == 0:
            continue
        summary = str(r.get("issue_summary", "")).replace("|", "/")
        lines.append(
            f"| {r['code']} | {r['qc_status']} | {r['error_count']} | {r['warning_count']} | "
            f"{r['tag_variants_count']} | {r['unknown_tag_count']} | {summary} |\n"
        )

    lines.append("\n## 7. Cómo usar este reporte\n")
    lines.append("1. Revisar primero los archivos con `REVISAR_ERROR`.\n")
    lines.append("2. Luego revisar variantes de etiquetas, especialmente si afectan `SIN_RESPUESTA`, `SIN_PREGUNTA`, `PAUSA`, `EE`, `IF`, `PS`, `DI` o `DP`.\n")
    lines.append("3. Confirmar que las etiquetas con duración (`PAUSA`, `DP` y cuando aplique `DI`) tengan `StartTime` y `EndTime`.\n")
    lines.append("4. Revisar archivos con alta proporción de `spk_0`, porque pueden conservar verbalizaciones del entrevistador que no deben entrar al análisis NLP.\n")
    lines.append("5. Usar `transcription_inspection_summary.csv` para filtrar por `qc_status`, `issue_summary`, `unknown_tags` y columnas `tag_*_count`.\n")

    path.write_text("".join(lines), encoding="utf-8")

# -----------------------------
# CLI
# -----------------------------

def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspecciona transcripciones corregidas/etiquetadas y genera 2 archivos de salida.")
    parser.add_argument("--transcripts", default="data/processed/Transcripciones", help="Carpeta con archivos .txt")
    parser.add_argument("--out", default="outputs/transcription_inspect", help="Carpeta de salida")
    parser.add_argument("--recursive", action="store_true", help="Buscar .txt recursivamente")
    args = parser.parse_args(list(argv) if argv is not None else None)

    transcripts_dir = Path(args.transcripts)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not transcripts_dir.exists():
        raise SystemExit(f"No existe la carpeta de transcripciones: {transcripts_dir}")

    pattern = "**/*.txt" if args.recursive else "*.txt"
    files = sorted(transcripts_dir.glob(pattern))
    if not files:
        raise SystemExit(f"No se encontraron .txt en: {transcripts_dir}")

    rows = []
    for idx, path in enumerate(files, start=1):
        print(f"[{idx}/{len(files)}] inspeccionando {path.name}")
        try:
            rows.append(inspect_file(path))
        except Exception as exc:
            rows.append({
                "code": normalize_code_from_filename(path),
                "file": str(path),
                "qc_status": "REVISAR_ERROR",
                "error_count": 1,
                "warning_count": 0,
                "info_count": 0,
                "issue_summary": f"error:read_file:{exc}",
                "issues_json": json.dumps([{
                    "severity": "error",
                    "issue_type": "read_file",
                    "message": str(exc),
                    "line": "",
                    "raw": "",
                    "suggested_fix": "Revisar codificación o integridad del archivo.",
                }], ensure_ascii=False),
            })

    csv_path = out_dir / "transcription_inspection_summary.csv"
    md_path = out_dir / "transcription_inspection_report.md"
    write_csv(rows, csv_path)
    write_md(rows, md_path, transcripts_dir)

    print("\nListo. Se generaron exactamente estos 2 archivos:")
    print(f"  {csv_path}")
    print(f"  {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
