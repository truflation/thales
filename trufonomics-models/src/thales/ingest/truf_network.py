"""TRUF Network component-stream ingest (via subprocess worker).

Pulls component-level daily time-series from TRUF Network for each stream in
``data/truflation/streams_catalog.csv`` and writes them to the vintage store
with ``source='truf_network'``.

Why subprocess: the TRUF Python SDK's Go-backed native extension segfaults
(SIGSEGV) under pure-arm64 Python on macOS 15. The augustus venv-bridge
Python is a universal2 build that loads the bindings cleanly, so we shell
out to that interpreter for the SDK call and keep our main venv clean.

Override the worker Python with ``TRUF_WORKER_PYTHON`` env var (e.g. when
deploying to Linux, point it at the in-venv python where the Linux wheel
installs without issues).

Usage:
    python -m thales.ingest.truf_network               # ingest all streams
    python -m thales.ingest.truf_network --limit 5     # first 5 only (smoke)
    python -m thales.ingest.truf_network --streams utilities_natural_gas food_at_home_dairy
    python -m thales.ingest.truf_network --start 2022-01-01

Env vars read:
    TRUFLATION_API_KEY   Ethereum private key (with or without 0x)
    TRUF_WORKER_PYTHON   path to a python with working SDK install
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from thales.vintage import IngestResult, VintageStore

ROOT = Path(__file__).resolve().parents[3]
VINTAGE_DB = ROOT / "data" / "vintage_store" / "thales.duckdb"
STREAMS_CSV = ROOT / "data" / "truflation" / "streams_catalog.csv"
WORKER_SCRIPT = ROOT / "scripts" / "truf_worker.py"

DEFAULT_WORKER_PYTHON = "/Users/kluless/augustus/aggretor/backend/.venv-bridge/bin/python"
DEFAULT_PROVIDER = "0x4710a8d8f0d845da110086812a32de6d90d7ff5c"
DEFAULT_GATEWAY = "https://gateway.mainnet.truf.network"
DEFAULT_START = "2020-01-01"

SOURCE = "truf_network"


@dataclass
class StreamSpec:
    raw_name: str          # short id, used as series_id in our store
    humanized_name: str    # full descriptive name (e.g. us_utilities_natural_gas_frozen_index)
    tn_stream_id: str      # the TN blockchain stream id (st...)


def load_stream_catalog(path: Path = STREAMS_CSV) -> list[StreamSpec]:
    df = pd.read_csv(path)
    return [
        StreamSpec(r["raw_name"], r["humanized_name"], r["tn_stream_id"])
        for _, r in df.iterrows()
    ]


def _load_env(*names: str) -> str | None:
    """Read the first of `names` found in either .env or the shell env.

    Prefers .env over os.environ so a local project config can't be silently
    overridden by an unrelated shell variable with a colliding name.
    """
    env_file = ROOT / ".env"
    if env_file.exists():
        kv = {}
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            kv[k.strip()] = v.strip()
        for name in names:
            if name in kv and kv[name]:
                return kv[name]
    for name in names:
        v = os.environ.get(name)
        if v:
            return v
    return None


def _load_private_key() -> str | None:
    """Prefer TRUF_PRIVATE_KEY (augustus convention); fall back to
    TRUFLATION_API_KEY for back-compat with earlier Thales .env files."""
    return _load_env("TRUF_PRIVATE_KEY", "TRUFLATION_API_KEY")


def call_worker(
    stream_id: str,
    date_from: int,
    date_to: int,
    *,
    provider: str = DEFAULT_PROVIDER,
    gateway: str = DEFAULT_GATEWAY,
    private_key: str | None = None,
    worker_python: str | None = None,
    timeout_s: int = 120,
) -> list[tuple[int, float]]:
    """Invoke the SDK worker subprocess and return [(event_time, value), ...]."""
    private_key = private_key or _load_private_key()
    if not private_key:
        raise RuntimeError(
            "TRUF_PRIVATE_KEY (or legacy TRUFLATION_API_KEY) not set"
        )

    worker_python = (worker_python
                      or os.environ.get("TRUF_WORKER_PYTHON")
                      or DEFAULT_WORKER_PYTHON)

    if not Path(worker_python).exists():
        raise RuntimeError(
            f"TRUF worker Python not found: {worker_python}. "
            f"Set TRUF_WORKER_PYTHON to a python with a working "
            f"trufnetwork-sdk-py install."
        )

    cmd = [
        worker_python, str(WORKER_SCRIPT),
        "--stream-id", stream_id,
        "--provider", provider,
        "--date-from", str(date_from),
        "--date-to", str(date_to),
        "--gateway", gateway,
        "--private-key", private_key,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=timeout_s,
                               text=True, check=False)
    except subprocess.TimeoutExpired:
        raise TimeoutError(f"truf worker timed out after {timeout_s}s on {stream_id}")

    if proc.returncode != 0:
        err_tail = (proc.stderr or "")[-500:]
        raise RuntimeError(
            f"truf worker failed (exit {proc.returncode}) on {stream_id}: "
            f"{err_tail}"
        )
    try:
        body = json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"truf worker returned non-JSON on {stream_id}: "
            f"{proc.stdout[:300]!r}"
        ) from e
    return [(r["event_time"], r["value"]) for r in body.get("records", [])]


def ingest_stream(store: VintageStore, spec: StreamSpec, *,
                   start_date: str = DEFAULT_START,
                   as_of_date: date | None = None,
                   **worker_kwargs) -> IngestResult:
    """Fetch one stream and write to the vintage store."""
    as_of = as_of_date or date.today()
    start_ts = int(datetime.fromisoformat(start_date).replace(
        tzinfo=timezone.utc).timestamp())
    end_ts = int(datetime.now(timezone.utc).timestamp())

    records = call_worker(spec.tn_stream_id, start_ts, end_ts, **worker_kwargs)
    obs = []
    for event_time, value in records:
        ref_date = datetime.fromtimestamp(event_time, tz=timezone.utc).date()
        obs.append((ref_date, value))
    return store.ingest(
        series_id=spec.raw_name,
        observations=obs,
        as_of_date=as_of,
        source=SOURCE,
    )


def ingest_all(store: VintageStore, specs: list[StreamSpec], *,
                start_date: str = DEFAULT_START,
                sleep_between: float = 0.2) -> dict[str, IngestResult | str]:
    """Sequentially ingest every spec, tolerating per-stream failures."""
    as_of = date.today()
    results: dict[str, IngestResult | str] = {}
    n = len(specs)
    for i, spec in enumerate(specs, 1):
        t0 = time.monotonic()
        try:
            res = ingest_stream(store, spec, start_date=start_date,
                                 as_of_date=as_of)
            dt = time.monotonic() - t0
            results[spec.raw_name] = res
            print(f"  [{i:3d}/{n}] {spec.raw_name:70s} "
                  f"{res.rows_inserted:>5d} rows  ({dt:.1f}s)")
        except Exception as e:  # noqa: BLE001
            dt = time.monotonic() - t0
            msg = f"{type(e).__name__}: {e}"
            results[spec.raw_name] = msg
            print(f"  [{i:3d}/{n}] {spec.raw_name:70s} FAILED "
                  f"({dt:.1f}s): {msg[:100]}")
        time.sleep(sleep_between)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="ingest only first N streams (smoke test)")
    parser.add_argument("--streams", nargs="*",
                        help="subset of raw_names to ingest")
    parser.add_argument("--start", default=DEFAULT_START,
                        help=f"earliest date to fetch (default {DEFAULT_START})")
    args = parser.parse_args()

    catalog = load_stream_catalog()
    if args.streams:
        wanted = set(args.streams)
        catalog = [s for s in catalog if s.raw_name in wanted]
        missing = wanted - {s.raw_name for s in catalog}
        if missing:
            print(f"WARN: streams not in catalog: {sorted(missing)}")
    if args.limit:
        catalog = catalog[: args.limit]

    print(f"ingest target: {len(catalog)} stream(s) from {args.start}")
    print(f"worker python: "
          f"{os.environ.get('TRUF_WORKER_PYTHON', DEFAULT_WORKER_PYTHON)}")
    print(f"vintage store: {VINTAGE_DB}")
    print()

    with VintageStore(VINTAGE_DB) as store:
        results = ingest_all(store, catalog, start_date=args.start)

    ok = [k for k, v in results.items() if isinstance(v, IngestResult)]
    failed = [k for k, v in results.items() if isinstance(v, str)]
    total_rows = sum(v.rows_inserted for v in results.values()
                     if isinstance(v, IngestResult))
    print(f"\n=== done. {len(ok)} ok, {len(failed)} failed. "
          f"{total_rows:,} new rows. ===")
    if failed:
        print("failures:")
        for k in failed:
            print(f"  {k}: {results[k]}")


if __name__ == "__main__":
    main()
