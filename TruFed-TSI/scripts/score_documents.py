#!/usr/bin/env python3
"""Score Fed documents for the TruFed Text Sentiment Index.

Input is a CSV document inventory. The scorer reads text from a selected text
column or from raw_text_path, counts Fed-specific dovish/hawkish/neutral policy
terms, and writes a scored CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


Match = Tuple[int, int, str, float, str]


def load_dictionary(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        dictionary = json.load(f)
    for key in ("dovish", "hawkish", "neutral_policy"):
        if key not in dictionary:
            raise ValueError(f"Dictionary missing required key: {key}")
    return dictionary


def normalize_text(text: str) -> str:
    text = text.lower()
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def term_pattern(term: str) -> re.Pattern[str]:
    escaped = re.escape(term.lower())
    escaped = escaped.replace(r"\ ", r"\s+")
    return re.compile(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])")


def iter_dictionary_terms(dictionary: dict) -> Iterable[Tuple[str, str, float]]:
    for category in ("dovish", "hawkish", "neutral_policy"):
        for entry in dictionary.get(category, []):
            term = str(entry["term"]).strip().lower()
            weight = float(entry.get("weight", 1))
            if term:
                yield category, term, weight


def find_matches(text: str, dictionary: dict) -> List[Match]:
    matches: List[Match] = []
    normalized = normalize_text(text)
    for category, term, weight in iter_dictionary_terms(dictionary):
        pattern = term_pattern(term)
        for match in pattern.finditer(normalized):
            matches.append((match.start(), match.end(), category, weight, term))

    priority = {"dovish": 0, "hawkish": 0, "neutral_policy": 1}
    matches.sort(key=lambda m: (-(m[1] - m[0]), priority.get(m[2], 9), m[0]))

    accepted: List[Match] = []
    occupied: List[Tuple[int, int]] = []
    for match in matches:
        start, end = match[0], match[1]
        if any(start < used_end and end > used_start for used_start, used_end in occupied):
            continue
        accepted.append(match)
        occupied.append((start, end))

    accepted.sort(key=lambda m: m[0])
    return accepted


def score_text(text: str, dictionary: dict) -> dict:
    matches = find_matches(text, dictionary)
    counts = Counter()
    term_hits = Counter()
    for _start, _end, category, weight, term in matches:
        counts[category] += weight
        term_hits[f"{category}:{term}"] += weight

    dovish = counts["dovish"]
    hawkish = counts["hawkish"]
    neutral = counts["neutral_policy"]
    eligible = dovish + hawkish + neutral
    document_score = "" if eligible == 0 else (dovish - hawkish) / eligible

    return {
        "dovish_terms": clean_number(dovish),
        "hawkish_terms": clean_number(hawkish),
        "neutral_policy_terms": clean_number(neutral),
        "eligible_policy_terms": clean_number(eligible),
        "document_score": "" if document_score == "" else f"{document_score:.8f}",
        "term_hits_json": json.dumps(dict(sorted(term_hits.items())), sort_keys=True),
    }


def clean_number(value: float) -> str:
    if math.isclose(value, round(value)):
        return str(int(round(value)))
    return f"{value:.6f}".rstrip("0").rstrip(".")


def read_text(row: dict, text_column: str | None) -> str:
    if text_column and row.get(text_column):
        return row[text_column]
    for candidate in ("clean_text", "raw_text", "text"):
        if row.get(candidate):
            return row[candidate]
    raw_text_path = row.get("raw_text_path", "")
    if raw_text_path:
        path = Path(raw_text_path)
        if path.exists():
            return path.read_text(encoding="utf-8", errors="ignore")
    return ""


def score_csv(input_path: Path, dictionary_path: Path, output_path: Path, text_column: str | None) -> None:
    dictionary = load_dictionary(dictionary_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8-sig", newline="") as src:
        reader = csv.DictReader(src)
        if reader.fieldnames is None:
            raise ValueError("Input CSV has no header row")

        added_fields = [
            "dovish_terms",
            "hawkish_terms",
            "neutral_policy_terms",
            "eligible_policy_terms",
            "document_score",
            "term_hits_json",
            "scoring_dictionary_version",
        ]
        fieldnames = list(reader.fieldnames)
        for field in added_fields:
            if field not in fieldnames:
                fieldnames.append(field)

        with output_path.open("w", encoding="utf-8", newline="") as dst:
            writer = csv.DictWriter(dst, fieldnames=fieldnames)
            writer.writeheader()
            for row in reader:
                text = read_text(row, text_column)
                row.update(score_text(text, dictionary))
                row["scoring_dictionary_version"] = dictionary.get("version", "")
                writer.writerow(row)


def self_test(dictionary_path: Path) -> None:
    dictionary = load_dictionary(dictionary_path)
    sample = (
        "Inflation remains elevated and price pressures persistent. "
        "However, the labor market has softened and inflation has eased."
    )
    result = score_text(sample, dictionary)
    print(json.dumps(result, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score Fed documents for TruFed TSI.")
    parser.add_argument("--input", type=Path, help="Input document inventory CSV.")
    parser.add_argument("--dictionary", type=Path, required=True, help="Fed policy dictionary JSON.")
    parser.add_argument("--output", type=Path, help="Output scored CSV.")
    parser.add_argument("--text-column", default=None, help="Optional text column to score.")
    parser.add_argument("--self-test", action="store_true", help="Run a small scoring smoke test.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test(args.dictionary)
        return
    if not args.input or not args.output:
        raise SystemExit("--input and --output are required unless --self-test is used")
    score_csv(args.input, args.dictionary, args.output, args.text_column)


if __name__ == "__main__":
    main()
