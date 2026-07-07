#!/usr/bin/env python3
"""Aggregate scored Fed documents into a quarterly TruFed TSI series."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterable, List, Optional


@dataclass
class Window:
    period_label: str
    asof_date: date
    window_start: Optional[date]
    window_end: date


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


def load_asof_windows(path: Path) -> List[Window]:
    windows: List[Window] = []
    previous: Optional[date] = None
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("As-of CSV has no header row")
        for row in reader:
            raw_date = row.get("asof_date") or row.get("release_date") or row.get("date")
            asof = parse_date(raw_date or "")
            if asof is None:
                continue
            label = row.get("period_label") or row.get("quarter") or quarter_label(asof)
            windows.append(Window(label, asof, previous, asof))
            previous = asof
    return windows


def calendar_quarter_windows(rows: Iterable[dict]) -> List[Window]:
    dates = sorted({parse_date(row.get("published_at", "")) for row in rows})
    dates = [d for d in dates if d is not None]
    labels = sorted({quarter_label(d) for d in dates})
    windows: List[Window] = []
    for label in labels:
        year = int(label[:4])
        quarter = int(label[-1])
        start_month = (quarter - 1) * 3 + 1
        end_month = start_month + 2
        start = date(year, start_month, 1)
        if end_month == 12:
            end = date(year, 12, 31)
        else:
            end = date(year, end_month + 1, 1).replace(day=1)
            end = date.fromordinal(end.toordinal() - 1)
        windows.append(Window(label, end, date.fromordinal(start.toordinal() - 1), end))
    return windows


def number(row: dict, field: str) -> float:
    raw = (row.get(field) or "").strip()
    if not raw:
        return 0.0
    return float(raw)


def interpretation(tsi: Optional[float]) -> str:
    if tsi is None:
        return "missing"
    if tsi > 0.01:
        return "dovish"
    if tsi < -0.01:
        return "hawkish"
    return "neutral"


def aggregate(input_path: Path, output_path: Path, asof_dates_path: Path | None) -> None:
    with input_path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    if asof_dates_path:
        windows = load_asof_windows(asof_dates_path)
    else:
        windows = calendar_quarter_windows(rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "period_label",
        "asof_date",
        "window_start",
        "window_end",
        "document_count",
        "dovish_terms",
        "hawkish_terms",
        "neutral_policy_terms",
        "eligible_policy_terms",
        "tsi",
        "interpretation",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for window in windows:
            selected = []
            for row in rows:
                published_at = parse_date(row.get("published_at", ""))
                if published_at is None:
                    continue
                after_start = window.window_start is None or published_at > window.window_start
                before_end = published_at <= window.window_end
                if after_start and before_end:
                    selected.append(row)

            dovish = sum(number(row, "dovish_terms") for row in selected)
            hawkish = sum(number(row, "hawkish_terms") for row in selected)
            neutral = sum(number(row, "neutral_policy_terms") for row in selected)
            eligible = sum(number(row, "eligible_policy_terms") for row in selected)
            tsi = None if eligible == 0 else (dovish - hawkish) / eligible

            writer.writerow(
                {
                    "period_label": window.period_label,
                    "asof_date": window.asof_date.isoformat(),
                    "window_start": "" if window.window_start is None else window.window_start.isoformat(),
                    "window_end": window.window_end.isoformat(),
                    "document_count": len(selected),
                    "dovish_terms": f"{dovish:.6f}".rstrip("0").rstrip("."),
                    "hawkish_terms": f"{hawkish:.6f}".rstrip("0").rstrip("."),
                    "neutral_policy_terms": f"{neutral:.6f}".rstrip("0").rstrip("."),
                    "eligible_policy_terms": f"{eligible:.6f}".rstrip("0").rstrip("."),
                    "tsi": "" if tsi is None else f"{tsi:.8f}",
                    "interpretation": interpretation(tsi),
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate scored Fed documents into quarterly TSI.")
    parser.add_argument("--input", type=Path, required=True, help="Scored document CSV.")
    parser.add_argument("--output", type=Path, required=True, help="Quarterly TSI output CSV.")
    parser.add_argument(
        "--asof-dates",
        type=Path,
        default=None,
        help="Optional CSV with asof_date or release_date and optional period_label.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    aggregate(args.input, args.output, args.asof_dates)


if __name__ == "__main__":
    main()
