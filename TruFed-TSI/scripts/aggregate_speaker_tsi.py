#!/usr/bin/env python3
"""Aggregate scored Fed documents into speaker-level TSI values."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Optional


SPEAKER_DOCUMENT_TYPES = {
    "chair_press_conference",
    "board_speech",
    "board_testimony",
    "reserve_bank_speech",
    "voting_member_commentary",
}


def parse_date(value: str) -> Optional[date]:
    value = (value or "").strip()
    if not value:
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%B %d, %Y", "%b %d, %Y"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            pass
    return None


def quarter_label(d: date) -> str:
    quarter = ((d.month - 1) // 3) + 1
    return f"{d.year}Q{quarter}"


def number(row: dict, field: str) -> float:
    raw = (row.get(field) or "").strip()
    if not raw:
        return 0.0
    return float(raw)


def clean_number(value: float) -> str:
    if value == int(value):
        return str(int(value))
    return f"{value:.6f}".rstrip("0").rstrip(".")


def interpretation(tsi: Optional[float]) -> str:
    if tsi is None:
        return "missing"
    if tsi > 0.01:
        return "dovish"
    if tsi < -0.01:
        return "hawkish"
    return "neutral"


def speaker_name(row: dict) -> str:
    speaker = (row.get("speaker") or "").strip()
    if speaker:
        return speaker
    if row.get("document_type") == "chair_press_conference":
        return "Chair"
    return ""


def first_nonempty(values: list[str]) -> str:
    for value in values:
        if value:
            return value
    return ""


def aggregate(input_path: Path, output_path: Path, min_documents: int, include_committee: bool) -> None:
    with input_path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    groups = defaultdict(list)
    for row in rows:
        document_type = row.get("document_type", "")
        if not include_committee and document_type not in SPEAKER_DOCUMENT_TYPES:
            continue
        speaker = speaker_name(row)
        published_at = parse_date(row.get("published_at", ""))
        if not speaker or published_at is None:
            continue
        groups[(quarter_label(published_at), speaker)].append(row)

    output_rows = []
    for (period_label, speaker), group_rows in sorted(groups.items()):
        if len(group_rows) < min_documents:
            continue

        dates = [parse_date(row.get("published_at", "")) for row in group_rows]
        dates = [d for d in dates if d is not None]
        dovish = sum(number(row, "dovish_terms") for row in group_rows)
        hawkish = sum(number(row, "hawkish_terms") for row in group_rows)
        neutral = sum(number(row, "neutral_policy_terms") for row in group_rows)
        eligible = sum(number(row, "eligible_policy_terms") for row in group_rows)
        tsi = None if eligible == 0 else (dovish - hawkish) / eligible

        roles = [row.get("speaker_role", "") for row in group_rows]
        voter_flags = [row.get("is_fomc_voter", "") for row in group_rows]
        chair_flags = [row.get("is_chair", "") for row in group_rows]
        doc_types = sorted({row.get("document_type", "") for row in group_rows if row.get("document_type", "")})
        source_urls = [row.get("source_url", "") for row in group_rows if row.get("source_url", "")]

        output_rows.append(
            {
                "period_label": period_label,
                "speaker": speaker,
                "speaker_role": first_nonempty(roles),
                "is_fomc_voter": first_nonempty(voter_flags),
                "is_chair": first_nonempty(chair_flags),
                "document_count": len(group_rows),
                "document_types": ";".join(doc_types),
                "first_published_at": min(dates).isoformat() if dates else "",
                "last_published_at": max(dates).isoformat() if dates else "",
                "dovish_terms": clean_number(dovish),
                "hawkish_terms": clean_number(hawkish),
                "neutral_policy_terms": clean_number(neutral),
                "eligible_policy_terms": clean_number(eligible),
                "speaker_tsi": "" if tsi is None else f"{tsi:.8f}",
                "interpretation": interpretation(tsi),
                "source_urls": " ".join(source_urls),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "period_label",
        "speaker",
        "speaker_role",
        "is_fomc_voter",
        "is_chair",
        "document_count",
        "document_types",
        "first_published_at",
        "last_published_at",
        "dovish_terms",
        "hawkish_terms",
        "neutral_policy_terms",
        "eligible_policy_terms",
        "speaker_tsi",
        "interpretation",
        "source_urls",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate scored Fed documents into speaker-level TSI.")
    parser.add_argument("--input", type=Path, required=True, help="Scored document CSV.")
    parser.add_argument("--output", type=Path, required=True, help="Speaker-level TSI output CSV.")
    parser.add_argument("--min-documents", type=int, default=1, help="Minimum documents per speaker-period.")
    parser.add_argument(
        "--include-committee",
        action="store_true",
        help="Include committee-level documents if a speaker value is present.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    aggregate(args.input, args.output, args.min_documents, args.include_committee)


if __name__ == "__main__":
    main()
