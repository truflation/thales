#!/usr/bin/env python3
"""Extract text for PDF rows in a TruFed TSI document inventory.

This helper uses the local `pdftotext` binary when available. It leaves rows
unchanged if no PDF extractor is installed.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
from pathlib import Path
from urllib.request import Request, urlopen


def download(url: str, output_path: Path) -> None:
    request = Request(url, headers={"User-Agent": "TruFed-TSI-reference/0.1"})
    with urlopen(request, timeout=60) as response:
        output_path.write_bytes(response.read())


def extract_with_pdftotext(pdf_path: Path, text_path: Path) -> bool:
    pdftotext = shutil.which("pdftotext")
    if not pdftotext:
        return False
    result = subprocess.run(
        [pdftotext, "-layout", str(pdf_path), str(text_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0 and text_path.exists()


def process_inventory(input_path: Path, output_path: Path, raw_text_dir: Path) -> None:
    raw_text_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir = raw_text_dir / "pdf"
    text_dir = raw_text_dir / "pdf_text"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8-sig", newline="") as src:
        reader = csv.DictReader(src)
        if reader.fieldnames is None:
            raise ValueError("Input CSV has no header row")
        rows = list(reader)
        fieldnames = list(reader.fieldnames)

    if "raw_text_path" not in fieldnames:
        fieldnames.append("raw_text_path")

    for row in rows:
        url = row.get("source_url", "")
        if not url.lower().endswith(".pdf"):
            continue

        document_id = row.get("document_id") or url.rsplit("/", 1)[-1].replace(".pdf", "")
        pdf_path = pdf_dir / f"{document_id}.pdf"
        text_path = text_dir / f"{document_id}.txt"

        if not pdf_path.exists():
            try:
                download(url, pdf_path)
            except Exception as exc:
                row["clean_text"] = f"PDF_DOWNLOAD_ERROR: {exc}"
                continue

        if text_path.exists() or extract_with_pdftotext(pdf_path, text_path):
            row["raw_text_path"] = str(text_path)
        else:
            row["clean_text"] = "PDF_TEXT_EXTRACTION_MISSING: install pdftotext or use the production PDF parser"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as dst:
        writer = csv.DictWriter(dst, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract PDF text for TruFed TSI inventory rows.")
    parser.add_argument("--input", type=Path, required=True, help="Input document inventory CSV.")
    parser.add_argument("--output", type=Path, required=True, help="Output document inventory CSV.")
    parser.add_argument("--raw-text-dir", type=Path, required=True, help="Directory for downloaded PDFs and text.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    process_inventory(args.input, args.output, args.raw_text_dir)


if __name__ == "__main__":
    main()
