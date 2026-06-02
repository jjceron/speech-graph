#!/usr/bin/env python
"""
inspect_transcript_labels.py

Analiza archivos .txt de transcripciones y reporta inconsistencias en etiquetas
con formato [[ETIQUETA]].

Uso desde la raíz del proyecto:
    py -m scripts.inspect_transcript_labels

Ruta por defecto analizada:
    data/processed/Transcripciones/*.txt

También puedes pasar otra ruta:
    py -m scripts.inspect_transcript_labels --root data/processed/Transcripciones
    py -m scripts.inspect_transcript_labels --root /otra/carpeta --csv reportes/labels.csv
"""

from __future__ import annotations

import argparse
import csv
import difflib
import re
import sys
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

# Ajusta esta lista si tu codebook oficial tiene más etiquetas.
# El script usa esta lista para sugerir typos como SIN REPUESTA -> SIN RESPUESTA.
DEFAULT_EXPECTED_LABELS = {
    "EE",
    "PS",
    "IF",
    "DI",
    "DP",
    "ES",
    "PAUSA",
    "SIN RESPUESTA",
    "SIN PREGUNTA",
}

DEFAULT_ROOT = Path("data/processed/Transcripciones")

# Captura candidatos tipo [[EE]], [[DI StartTime=...]], [IF]], [[ES], [[EE]]]
# También captura [StartTime=...] pero eso se ignora porque no es etiqueta [[...]].
TAG_CANDIDATE_RE = re.compile(
    r"(?P<open>\[{1,4})(?P<body>[^\[\]\n]{1,180}?)(?P<close>\]{1,4})"
)

ATTR_START_RE = re.compile(r"\s+(?:StartTime|EndTime)\s*=", re.IGNORECASE)
LETTER_RE = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]")


@dataclass(frozen=True)
class TagOccurrence:
    file: Path
    line_no: int
    col_no: int
    raw_token: str
    inner: str
    label_name: str
    normalized_name: str
    context: str


@dataclass(frozen=True)
class Issue:
    file: Path
    line_no: int
    col_no: int
    kind: str
    found: str
    message: str
    suggestion: str = ""
    context: str = ""


def read_text_with_fallback(path: Path) -> str:
    """Lee txt intentando codificaciones comunes."""
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    # latin-1 casi nunca falla, pero dejamos fallback defensivo.
    return path.read_text(encoding="utf-8", errors="replace")


def strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def normalize_label_name(label: str) -> str:
    """Normaliza para comparar: mayúsculas, espacios colapsados y sin tildes."""
    label = strip_accents(label.strip())
    label = re.sub(r"\s+", " ", label)
    return label.upper()


def extract_label_name(inner: str) -> str:
    """
    Extrae el nombre/código de la etiqueta, separando atributos.

    Ejemplos:
        'DI StartTime=01:00 EndTime=01:03' -> 'DI'
        'PAUSA StartTime=01:00 EndTime=01:03' -> 'PAUSA'
        'ES=golpea la mesa' -> 'ES'
        'SIN RESPUESTA' -> 'SIN RESPUESTA'
    """
    inner = re.sub(r"\s+", " ", inner.strip())

    attr_match = ATTR_START_RE.search(inner)
    if attr_match:
        return inner[: attr_match.start()].strip()

    if "=" in inner:
        return inner.split("=", 1)[0].strip()

    return inner


def replace_label_name(inner: str, new_label_name: str) -> str:
    """Reemplaza solo el nombre de la etiqueta y conserva atributos/comentarios."""
    inner = inner.strip()
    attr_match = ATTR_START_RE.search(inner)
    if attr_match:
        return new_label_name + inner[attr_match.start() :]
    if "=" in inner:
        return new_label_name + inner[inner.index("=") :]
    return new_label_name


def looks_like_label_body(body: str) -> bool:
    """Heurística para no confundir [StartTime=...] con etiquetas."""
    label_name = extract_label_name(body)
    if not label_name or not LETTER_RE.search(label_name):
        return False

    # Evita que metadatos con corchetes simples sean tratados como etiquetas.
    if normalize_label_name(label_name) in {"STARTTIME", "ENDTIME"}:
        return False

    # Etiquetas/códigos esperados: EE, PS, IF, DI, SIN RESPUESTA, PAUSA, ES, etc.
    return bool(
        re.fullmatch(
            r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ][A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9_\- ]{0,60}",
            label_name,
        )
    )


def make_context(line: str, start: int, end: int, width: int = 55) -> str:
    left = max(0, start - width)
    right = min(len(line), end + width)
    prefix = "…" if left > 0 else ""
    suffix = "…" if right < len(line) else ""
    return prefix + line[left:right].strip() + suffix


def in_any_span(pos: int, spans: list[tuple[int, int]]) -> bool:
    return any(start <= pos < end for start, end in spans)


def end_in_any_span(end_pos: int, spans: list[tuple[int, int]]) -> bool:
    return any(start < end_pos <= end for start, end in spans)


def levenshtein_distance(a: str, b: str, limit: int = 3) -> int:
    """Distancia Levenshtein con corte temprano para typos pequeños."""
    if a == b:
        return 0
    if abs(len(a) - len(b)) > limit:
        return limit + 1

    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        row_min = current[0]
        for j, cb in enumerate(b, start=1):
            insert = current[j - 1] + 1
            delete = previous[j] + 1
            replace = previous[j - 1] + (ca != cb)
            value = min(insert, delete, replace)
            current.append(value)
            row_min = min(row_min, value)
        if row_min > limit:
            return limit + 1
        previous = current
    return previous[-1]


def find_closest_expected(
    normalized_name: str, expected_by_normalized: dict[str, str]
) -> str:
    if normalized_name in expected_by_normalized:
        return expected_by_normalized[normalized_name]

    candidates = difflib.get_close_matches(
        normalized_name,
        expected_by_normalized.keys(),
        n=1,
        cutoff=0.78,
    )
    return expected_by_normalized[candidates[0]] if candidates else ""


def iter_txt_files(root: Path, recursive: bool = False) -> list[Path]:
    pattern = "**/*.txt" if recursive else "*.txt"
    return sorted(p for p in root.glob(pattern) if p.is_file())


def inspect_file(
    path: Path,
    expected_by_normalized: dict[str, str],
    check_expected: bool,
) -> tuple[list[TagOccurrence], list[Issue]]:
    text = read_text_with_fallback(path)
    occurrences: list[TagOccurrence] = []
    issues: list[Issue] = []

    for line_no, line in enumerate(text.splitlines(), start=1):
        candidate_spans: list[tuple[int, int]] = []

        for match in TAG_CANDIDATE_RE.finditer(line):
            open_brackets = match.group("open")
            close_brackets = match.group("close")
            body = match.group("body")
            token = match.group(0)
            start, end = match.span()

            open_count = len(open_brackets)
            close_count = len(close_brackets)

            # Ignora metadatos normales tipo [StartTime=... EndTime=...].
            if open_count == 1 and close_count == 1:
                continue

            if not looks_like_label_body(body):
                continue

            candidate_spans.append((start, end))
            inner = re.sub(r"\s+", " ", body.strip())
            label_name = extract_label_name(inner)
            normalized_name = normalize_label_name(label_name)
            context = make_context(line, start, end)
            col_no = start + 1

            if open_count != 2 or close_count != 2:
                suggestion_inner = inner
                issues.append(
                    Issue(
                        file=path,
                        line_no=line_no,
                        col_no=col_no,
                        kind="BRACKETS",
                        found=token,
                        message="Etiqueta con cantidad incorrecta de corchetes; debe usar exactamente [[...]].",
                        suggestion=f"[[{suggestion_inner}]]",
                        context=context,
                    )
                )

            # Aunque tenga error de corchetes, la recuperamos para revisar typo/case.
            occurrences.append(
                TagOccurrence(
                    file=path,
                    line_no=line_no,
                    col_no=col_no,
                    raw_token=token,
                    inner=inner,
                    label_name=label_name,
                    normalized_name=normalized_name,
                    context=context,
                )
            )

            expected_case = normalize_label_name(label_name)
            # Normaliza solo para sugerir mayúsculas; conserva atributos.
            if label_name != expected_case and LETTER_RE.search(label_name):
                fixed_inner = replace_label_name(inner, expected_case)
                issues.append(
                    Issue(
                        file=path,
                        line_no=line_no,
                        col_no=col_no,
                        kind="CASE",
                        found=token,
                        message="Etiqueta escrita con minúsculas, mezcla de mayúsculas o acentos/espacios inconsistentes.",
                        suggestion=f"[[{fixed_inner}]]",
                        context=context,
                    )
                )

            if check_expected and normalized_name not in expected_by_normalized:
                closest = find_closest_expected(normalized_name, expected_by_normalized)
                if closest:
                    fixed_inner = replace_label_name(inner, closest)
                    suggestion = f"[[{fixed_inner}]]"
                    message = "Etiqueta no está en la lista esperada; parece un typo de una etiqueta oficial."
                else:
                    suggestion = ""
                    message = "Etiqueta no está en la lista esperada; revisar si es código nuevo o error."
                issues.append(
                    Issue(
                        file=path,
                        line_no=line_no,
                        col_no=col_no,
                        kind="UNKNOWN_OR_TYPO",
                        found=token,
                        message=message,
                        suggestion=suggestion,
                        context=context,
                    )
                )

        # Casos más rotos: [[SIN CIERRE o CIERRE]] sin apertura.
        # No duplicamos lo que ya capturó TAG_CANDIDATE_RE.
        for open_match in re.finditer(r"\[\[", line):
            start = open_match.start()
            if not in_any_span(start, candidate_spans):
                issues.append(
                    Issue(
                        file=path,
                        line_no=line_no,
                        col_no=start + 1,
                        kind="BRACKETS",
                        found="[[",
                        message="Apertura [[ sin cierre ]] claro en la misma línea.",
                        suggestion="Revisar y cerrar como [[ETIQUETA]].",
                        context=make_context(line, start, start + 2),
                    )
                )

        for close_match in re.finditer(r"\]\]", line):
            end_pos = close_match.end()
            start = close_match.start()
            if not end_in_any_span(end_pos, candidate_spans):
                issues.append(
                    Issue(
                        file=path,
                        line_no=line_no,
                        col_no=start + 1,
                        kind="BRACKETS",
                        found="]]",
                        message="Cierre ]] sin apertura [[ clara en la misma línea.",
                        suggestion="Revisar y abrir como [[ETIQUETA]].",
                        context=make_context(line, start, end_pos),
                    )
                )

    return occurrences, issues


def add_variant_issues(occurrences: list[TagOccurrence]) -> list[Issue]:
    issues: list[Issue] = []

    by_file_and_norm: dict[tuple[Path, str], dict[str, list[TagOccurrence]]] = defaultdict(
        lambda: defaultdict(list)
    )
    by_file: dict[Path, dict[str, list[TagOccurrence]]] = defaultdict(lambda: defaultdict(list))

    for occ in occurrences:
        by_file_and_norm[(occ.file, occ.normalized_name)][occ.label_name].append(occ)
        by_file[occ.file][occ.normalized_name].append(occ)

    # Misma etiqueta normalizada escrita con variantes: [[sin respuesta]] vs [[SIN RESPUESTA]].
    for (path, normalized_name), variants in sorted(
        by_file_and_norm.items(), key=lambda item: (str(item[0][0]), item[0][1])
    ):
        if len(variants) <= 1:
            continue
        first_occ = min(
            (items[0] for items in variants.values()),
            key=lambda occ: (occ.line_no, occ.col_no),
        )
        variant_list = ", ".join(sorted(repr(v) for v in variants.keys()))
        issues.append(
            Issue(
                file=path,
                line_no=first_occ.line_no,
                col_no=first_occ.col_no,
                kind="VARIANT_CASE_SPACE",
                found=first_occ.raw_token,
                message=f"Misma etiqueta aparece con variantes de mayúsculas/espacios: {variant_list}.",
                suggestion=f"Usar una sola forma, por ejemplo [[{normalized_name}]].",
                context=first_occ.context,
            )
        )

    # Posibles typos dentro del mismo archivo: SIN RESPUESTA vs SIN REPUESTA.
    for path, labels in sorted(by_file.items(), key=lambda item: str(item[0])):
        normalized_names = sorted(labels.keys())
        for i, a in enumerate(normalized_names):
            for b in normalized_names[i + 1 :]:
                if min(len(a), len(b)) < 5:
                    continue
                if a[0] != b[0]:
                    continue

                distance = levenshtein_distance(a, b, limit=2)
                ratio = difflib.SequenceMatcher(None, a, b).ratio()

                if distance <= 2 or ratio >= 0.88:
                    occ_a = labels[a][0]
                    occ_b = labels[b][0]
                    first_occ = min((occ_a, occ_b), key=lambda occ: (occ.line_no, occ.col_no))
                    issues.append(
                        Issue(
                            file=path,
                            line_no=first_occ.line_no,
                            col_no=first_occ.col_no,
                            kind="POSSIBLE_TYPO_PAIR",
                            found=f"[[{a}]] / [[{b}]]",
                            message="Dos etiquetas muy parecidas aparecen en el mismo archivo; puede ser typo o duplicado inconsistente.",
                            suggestion=f"Revisar si una debe reemplazar a la otra: [[{a}]] vs [[{b}]].",
                            context=first_occ.context,
                        )
                    )

    return issues


def write_csv(path: Path, issues: Iterable[Issue], root: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "file",
                "line",
                "column",
                "kind",
                "found",
                "message",
                "suggestion",
                "context",
            ],
        )
        writer.writeheader()
        for issue in issues:
            writer.writerow(
                {
                    "file": str(issue.file.relative_to(root) if issue.file.is_relative_to(root) else issue.file),
                    "line": issue.line_no,
                    "column": issue.col_no,
                    "kind": issue.kind,
                    "found": issue.found,
                    "message": issue.message,
                    "suggestion": issue.suggestion,
                    "context": issue.context,
                }
            )


def print_report(files: list[Path], issues: list[Issue], root: Path) -> None:
    print("\n=== INSPECCIÓN DE ETIQUETAS [[...]] ===")
    print(f"Archivos analizados: {len(files)}")

    if not issues:
        print("Resultado: no se encontraron inconsistencias de etiquetas.")
        return

    files_with_issues = sorted({issue.file for issue in issues})
    print(f"Archivos con posibles inconsistencias: {len(files_with_issues)}")
    print(f"Hallazgos totales: {len(issues)}")

    by_file: dict[Path, list[Issue]] = defaultdict(list)
    for issue in issues:
        by_file[issue.file].append(issue)

    for path in files_with_issues:
        rel = path.relative_to(root) if path.is_relative_to(root) else path
        print(f"\n--- {rel} ---")
        for issue in sorted(by_file[path], key=lambda x: (x.line_no, x.col_no, x.kind)):
            loc = f"L{issue.line_no}:C{issue.col_no}"
            print(f"[{issue.kind}] {loc} {issue.message}")
            print(f"    encontrado: {issue.found}")
            if issue.suggestion:
                print(f"    sugerencia: {issue.suggestion}")
            if issue.context:
                print(f"    contexto: {issue.context}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detecta inconsistencias en etiquetas de transcripciones con formato [[ETIQUETA]]."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"Carpeta con .txt. Por defecto: {DEFAULT_ROOT}",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Busca .txt también en subcarpetas.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Ruta opcional para guardar un CSV con los hallazgos.",
    )
    parser.add_argument(
        "--expected-label",
        action="append",
        default=None,
        help=(
            "Etiqueta oficial esperada. Puedes repetirlo varias veces. "
            "Si no se usa, se toma la lista DEFAULT_EXPECTED_LABELS del script."
        ),
    )
    parser.add_argument(
        "--no-expected-check",
        action="store_true",
        help="Desactiva la revisión contra la lista de etiquetas esperadas.",
    )
    parser.add_argument(
        "--fail-on-issues",
        action="store_true",
        help="Devuelve código de salida 1 si encuentra hallazgos.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    root: Path = args.root

    if not root.exists() or not root.is_dir():
        print(f"ERROR: no existe la carpeta: {root}", file=sys.stderr)
        return 2

    expected_labels = set(args.expected_label or DEFAULT_EXPECTED_LABELS)
    expected_by_normalized = {normalize_label_name(label): label for label in expected_labels}
    check_expected = not args.no_expected_check

    files = iter_txt_files(root, recursive=args.recursive)
    if not files:
        print(f"No se encontraron archivos .txt en: {root}", file=sys.stderr)
        return 2

    all_occurrences: list[TagOccurrence] = []
    all_issues: list[Issue] = []

    for file_path in files:
        occurrences, issues = inspect_file(
            file_path,
            expected_by_normalized=expected_by_normalized,
            check_expected=check_expected,
        )
        all_occurrences.extend(occurrences)
        all_issues.extend(issues)

    all_issues.extend(add_variant_issues(all_occurrences))
    all_issues = sorted(all_issues, key=lambda x: (str(x.file), x.line_no, x.col_no, x.kind))

    print_report(files, all_issues, root)

    if args.csv:
        write_csv(args.csv, all_issues, root)
        print(f"\nCSV guardado en: {args.csv}")

    if all_issues and args.fail_on_issues:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
