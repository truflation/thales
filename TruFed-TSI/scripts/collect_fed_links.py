#!/usr/bin/env python3
"""Collect official Fed document links for the TruFed TSI reference pipeline.

This script creates a document inventory from Federal Reserve HTML pages. It is
intentionally lightweight and uses only the Python standard library.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import re
import time
from datetime import date, datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from urllib.parse import urljoin
from urllib.request import Request, urlopen


FED_BASE = "https://www.federalreserve.gov"
RECENT_FOMC_URL = f"{FED_BASE}/monetarypolicy/fomccalendars.htm"
HISTORICAL_FOMC_URL = f"{FED_BASE}/monetarypolicy/fomchistorical{{year}}.htm"


class LinkExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: List[dict] = []
        self._current_href: Optional[str] = None
        self._current_text: List[str] = []

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag.lower() != "a":
            return
        attrs_dict = dict(attrs)
        href = attrs_dict.get("href")
        if href:
            self._current_href = href
            self._current_text = []

    def handle_data(self, data: str) -> None:
        if self._current_href is not None:
            self._current_text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "a" and self._current_href is not None:
            text = " ".join(part.strip() for part in self._current_text if part.strip())
            self.links.append({"href": self._current_href, "text": html.unescape(text)})
            self._current_href = None
            self._current_text = []


class TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: List[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag.lower() in {"script", "style", "noscript", "svg"}:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style", "noscript", "svg"} and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        text = html.unescape(data).strip()
        if text:
            self.parts.append(text)

    def text(self) -> str:
        joined = " ".join(self.parts)
        return re.sub(r"\s+", " ", joined).strip()


def fetch(url: str) -> str:
    request = Request(url, headers={"User-Agent": "TruFed-TSI-reference/0.1"})
    with urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8", errors="ignore")


def extract_links(page_html: str, source_url: str) -> List[dict]:
    parser = LinkExtractor()
    parser.feed(page_html)
    links = []
    for link in parser.links:
        absolute = urljoin(source_url, link["href"])
        links.append({"url": absolute, "text": link["text"]})
    return links


def extract_text(page_html: str) -> str:
    parser = TextExtractor()
    parser.feed(page_html)
    return parser.text()


def date_from_yyyymmdd(value: str) -> Optional[date]:
    try:
        return datetime.strptime(value, "%Y%m%d").date()
    except ValueError:
        return None


def parse_release_dates(page_html: str) -> Dict[str, date]:
    release_dates: Dict[str, date] = {}
    pattern = re.compile(
        r"(?:fomcminutes|minutes/)(?P<meeting>\d{8})\.htm",
        flags=re.IGNORECASE,
    )
    for match in pattern.finditer(page_html):
        window_start = max(0, match.start() - 400)
        window_end = min(len(page_html), match.end() + 400)
        window = page_html[window_start:window_end]
        release_match = re.search(
            r"Released\s+(?P<released>[A-Za-z]+\s+\d{1,2},\s+\d{4})",
            window,
            flags=re.IGNORECASE,
        )
        if release_match is None:
            continue
        released = None
        for fmt in ("%B %d, %Y", "%b %d, %Y"):
            try:
                released = datetime.strptime(release_match.group("released"), fmt).date()
                break
            except ValueError:
                pass
        if released is None:
            continue
        release_dates.setdefault(match.group("meeting"), released)
    return release_dates


def infer_date_from_url(url: str) -> Optional[date]:
    match = re.search(r"(20\d{2})(\d{2})(\d{2})", url)
    if not match:
        return None
    return date_from_yyyymmdd("".join(match.groups()))


def document_id(document_type: str, source_url: str) -> str:
    digest = hashlib.sha1(source_url.encode("utf-8")).hexdigest()[:12]
    url_date = infer_date_from_url(source_url)
    date_part = "unknown" if url_date is None else url_date.strftime("%Y%m%d")
    return f"fed_{date_part}_{document_type}_{digest}"


def empty_row() -> dict:
    return {
        "document_id": "",
        "document_type": "",
        "source_url": "",
        "published_at": "",
        "retrieved_at": "",
        "meeting_date": "",
        "speaker": "",
        "speaker_role": "",
        "is_fomc_voter": "",
        "is_chair": "",
        "title": "",
        "raw_text_path": "",
        "clean_text": "",
        "source_page": "",
        "date_quality": "",
        "include_by_default": "true",
    }


def fomc_document_type(url: str) -> Optional[str]:
    if re.search(r"/newsevents/pressreleases/monetary\d{8}a\.htm$", url):
        return "fomc_statement"
    if re.search(r"/newsevents/press/monetary/\d{8}[a-z]\.htm$", url):
        return "fomc_statement"
    if re.search(r"/monetarypolicy/fomcminutes\d{8}\.htm$", url):
        return "fomc_minutes"
    if re.search(r"/fomc/minutes/\d{8}\.htm$", url):
        return "fomc_minutes"
    if re.search(r"/monetarypolicy/fomc(?:press|pres)conf\d{8}\.htm$", url):
        return "chair_press_conference"
    return None


def press_conference_transcript_url(press_page_url: str) -> Optional[str]:
    try:
        page_html = fetch(press_page_url)
    except Exception:
        return None
    for link in extract_links(page_html, press_page_url):
        url = link["url"]
        text = link["text"].lower()
        if url.lower().endswith(".pdf") and ("transcript" in text or "presconf" in url.lower()):
            return url
    return None


def collect_fomc_year(year: int, retrieved_at: str) -> List[dict]:
    source_url = HISTORICAL_FOMC_URL.format(year=year) if year <= 2020 else RECENT_FOMC_URL
    page_html = fetch(source_url)
    links = extract_links(page_html, source_url)
    release_dates = parse_release_dates(page_html)
    rows: List[dict] = []

    for link in links:
        url = link["url"]
        doc_type = fomc_document_type(url)
        if doc_type is None:
            continue
        meeting_date = infer_date_from_url(url)
        if meeting_date is None or meeting_date.year != year:
            continue

        row = empty_row()
        source_document_url = url
        title = link["text"] or doc_type.replace("_", " ").title()
        if doc_type == "chair_press_conference":
            transcript_url = press_conference_transcript_url(url)
            if transcript_url:
                source_document_url = transcript_url
                title = "Press Conference Transcript (PDF)"

        row["document_type"] = doc_type
        row["source_url"] = source_document_url
        row["document_id"] = document_id(doc_type, source_document_url)
        row["retrieved_at"] = retrieved_at
        row["meeting_date"] = meeting_date.isoformat()
        row["source_page"] = source_url
        row["title"] = title

        if doc_type == "fomc_minutes":
            key = meeting_date.strftime("%Y%m%d")
            released = release_dates.get(key)
            if released is not None:
                row["published_at"] = released.isoformat()
                row["date_quality"] = "minutes_release_date"
            else:
                row["published_at"] = ""
                row["date_quality"] = "missing"
        else:
            row["published_at"] = meeting_date.isoformat()
            row["date_quality"] = "url_date"
            if doc_type == "chair_press_conference":
                row["speaker"] = "Chair"
                row["speaker_role"] = "Federal Reserve Chair"
                row["is_chair"] = "true"
                if source_document_url != url:
                    row["source_page"] = url

        rows.append(row)

    return rows


def speech_archive_url(year: int) -> Optional[str]:
    if year >= 2011:
        return f"{FED_BASE}/newsevents/{year}-speeches.htm"
    return None


def testimony_archive_url(year: int) -> str:
    if year >= 2017:
        return f"{FED_BASE}/newsevents/{year}-testimony.htm"
    return f"{FED_BASE}/newsevents/{year}testimony.htm"


def collect_board_archive(year: int, doc_kind: str, retrieved_at: str) -> List[dict]:
    if doc_kind == "board_speech":
        source_url = speech_archive_url(year)
        link_pattern = re.compile(r"/newsevents/speech/[a-z-]+20\d{6}[a-z]?\.htm")
    elif doc_kind == "board_testimony":
        source_url = testimony_archive_url(year)
        link_pattern = re.compile(r"/newsevents/testimony/[a-z-]+20\d{6}[a-z]?\.htm")
    else:
        raise ValueError(f"Unsupported doc_kind: {doc_kind}")

    if source_url is None:
        return []

    try:
        page_html = fetch(source_url)
    except Exception:
        return []

    links = extract_links(page_html, source_url)
    rows: List[dict] = []
    seen = set()
    for link in links:
        url = link["url"]
        if not link_pattern.search(url):
            continue
        published_at = infer_date_from_url(url)
        if published_at is None or published_at.year != year:
            continue
        if url in seen:
            continue
        seen.add(url)

        speaker_match = re.search(r"/(?:speech|testimony)/([a-z-]+)20\d{6}", url)
        speaker = "" if speaker_match is None else speaker_match.group(1).replace("-", " ").title()

        row = empty_row()
        row["document_type"] = doc_kind
        row["source_url"] = url
        row["document_id"] = document_id(doc_kind, url)
        row["published_at"] = published_at.isoformat()
        row["retrieved_at"] = retrieved_at
        row["speaker"] = speaker
        row["speaker_role"] = "Federal Reserve Board"
        row["is_chair"] = "true" if speaker.lower() in {"bernanke", "yellen", "powell"} else ""
        row["title"] = link["text"]
        row["source_page"] = source_url
        row["date_quality"] = "url_date"
        rows.append(row)

    return rows


def maybe_download_text(rows: List[dict], raw_text_dir: Path, sleep_seconds: float) -> None:
    raw_text_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        url = row["source_url"]
        if not url.endswith(".htm"):
            continue
        output_path = raw_text_dir / f"{row['document_id']}.txt"
        if output_path.exists():
            row["raw_text_path"] = str(output_path)
            continue
        try:
            page_html = fetch(url)
            text = extract_text(page_html)
            output_path.write_text(text, encoding="utf-8")
            row["raw_text_path"] = str(output_path)
            if sleep_seconds:
                time.sleep(sleep_seconds)
        except Exception as exc:
            row["raw_text_path"] = ""
            row["clean_text"] = f"DOWNLOAD_ERROR: {exc}"


def dedupe_rows(rows: Iterable[dict]) -> List[dict]:
    deduped: Dict[str, dict] = {}
    for row in rows:
        key = row["source_url"]
        if key not in deduped:
            deduped[key] = row
    return list(deduped.values())


def collect(start_year: int, end_year: int, sources: List[str], raw_text_dir: Path | None, sleep_seconds: float) -> List[dict]:
    retrieved_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    rows: List[dict] = []

    for year in range(start_year, end_year + 1):
        if "fomc" in sources:
            rows.extend(collect_fomc_year(year, retrieved_at))
        if "board_speeches" in sources:
            rows.extend(collect_board_archive(year, "board_speech", retrieved_at))
        if "board_testimony" in sources:
            rows.extend(collect_board_archive(year, "board_testimony", retrieved_at))
        if sleep_seconds:
            time.sleep(sleep_seconds)

    rows = dedupe_rows(rows)
    rows.sort(key=lambda r: (r.get("published_at") or "9999-99-99", r.get("document_type", ""), r.get("source_url", "")))

    if raw_text_dir is not None:
        maybe_download_text(rows, raw_text_dir, sleep_seconds)

    return rows


def write_csv(rows: List[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(empty_row().keys())
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect official Fed document links for TruFed TSI.")
    parser.add_argument("--start-year", type=int, default=2007)
    parser.add_argument("--end-year", type=int, default=date.today().year)
    parser.add_argument("--out", type=Path, required=True, help="Output document inventory CSV.")
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["fomc", "board_speeches", "board_testimony"],
        choices=["fomc", "board_speeches", "board_testimony"],
        help="Official Fed source groups to collect.",
    )
    parser.add_argument("--raw-text-dir", type=Path, default=None, help="Optional directory for downloaded HTML text.")
    parser.add_argument("--sleep-seconds", type=float, default=0.2, help="Delay between requests.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = collect(args.start_year, args.end_year, args.sources, args.raw_text_dir, args.sleep_seconds)
    write_csv(rows, args.out)
    print(f"Wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
