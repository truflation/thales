"""Build the LLM Token Price Index snapshot.

Usage:
    python llm_token_price_index/scripts/build_token_price_index.py
    python llm_token_price_index/scripts/build_token_price_index.py --sources openrouter litellm simon
    python llm_token_price_index/scripts/build_token_price_index.py --portkey-providers openai anthropic google
    python llm_token_price_index/scripts/build_token_price_index.py --historical-backfill
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.token_price_index import build_token_price_index, parse_snapshot_date


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an LLM token price index snapshot.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "csv",
        help="Directory for normalized price and index outputs.",
    )
    parser.add_argument(
        "--snapshot-date",
        type=parse_snapshot_date,
        default=None,
        help="Snapshot date in YYYY-MM-DD format. Defaults to current UTC date.",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["openrouter", "portkey", "litellm", "simon"],
        choices=["openrouter", "portkey", "litellm", "simon", "simon_llm_prices"],
        help="Sources to collect.",
    )
    parser.add_argument(
        "--portkey-providers",
        nargs="+",
        default=None,
        help="Optional Portkey provider allowlist. Defaults to all Portkey provider files.",
    )
    parser.add_argument(
        "--portkey-core-providers",
        action="store_true",
        help="Use the built-in core Portkey provider list instead of all provider files.",
    )
    parser.add_argument(
        "--historical-backfill",
        action="store_true",
        help="Also collect Simon llm-prices historical interval data and backfill tables.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_token_price_index(
        output_dir=args.output_dir,
        snapshot_date=args.snapshot_date,
        sources=args.sources,
        portkey_providers=args.portkey_providers,
        portkey_all_providers=not args.portkey_core_providers,
        historical_backfill=args.historical_backfill,
    )
    print(f"Snapshot date: {summary['snapshot_date']}")
    print(f"Current observations: {summary['current_observations']:,}")
    print(f"Positive observations: {summary['positive_current_observations']:,}")
    print(f"Core headline observations: {summary['core_current_observations']:,}")
    print(f"Headline universe: {summary['headline_universe']}")
    print(f"Sources: {', '.join(summary['sources'])}")
    print(f"Token types: {', '.join(summary['token_types'])}")
    print(f"Latest prices: {summary['output_files']['latest']}")
    print(f"Latest core prices: {summary['output_files']['core_latest']}")
    print(f"Latest index: {summary['output_files']['latest_index']}")


if __name__ == "__main__":
    main()
