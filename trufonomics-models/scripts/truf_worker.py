"""Stand-alone TRUF Network SDK worker.

Invoked as a subprocess by ``thales.ingest.truf_network``. Must be runnable by
a Python interpreter that has a working ``trufnetwork_sdk_py`` install (see
README — on macOS arm64 we use the augustus venv-bridge Python because its
universal2 build loads the Go bindings cleanly; pure-arm64 Python segfaults
the SDK).

Takes arguments:
    --stream-id     st...
    --provider      0x... (data provider address)
    --date-from     unix timestamp (inclusive)
    --date-to       unix timestamp (inclusive)
    --gateway       default https://gateway.mainnet.truf.network
    --private-key   hex (with or without 0x)

Writes JSON to stdout:
    {"n": <int>, "records": [{"event_time": <unix>, "value": <float>}, ...]}

Warnings / logs go to stderr so they don't corrupt the stdout payload.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stream-id", required=True)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--date-from", required=True, type=int)
    parser.add_argument("--date-to", required=True, type=int)
    parser.add_argument("--gateway", default="https://gateway.mainnet.truf.network")
    parser.add_argument("--private-key", required=True)
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    from trufnetwork_sdk_py.client import TNClient  # type: ignore

    key = args.private_key[2:] if args.private_key.startswith("0x") else args.private_key
    client = TNClient(args.gateway, key)

    # Legacy return format = direct list of StreamRecord objects.
    # use_cache=True on v0.6.6 may switch to a dict-wrapped response; we keep
    # the legacy form for simplicity.
    records = client.get_records(
        stream_id=args.stream_id,
        data_provider=args.provider,
        date_from=args.date_from,
        date_to=args.date_to,
    ) or []
    # Unwrap if the response is a dict (future SDK versions).
    if isinstance(records, dict):
        records = records.get("records") or records.get("data") or []

    out = []
    for rec in records:
        if hasattr(rec, "Value"):
            value = float(rec.Value)
            event_time = int(rec.EventTime)
        elif isinstance(rec, dict):
            value = float(rec.get("value", rec.get("Value", 0)))
            event_time = int(rec.get("event_time", rec.get("EventTime", 0)))
        else:
            continue
        out.append({"event_time": event_time, "value": value})
    payload = {"stream_id": args.stream_id, "n": len(out), "records": out}
    sys.stdout.write(json.dumps(payload))
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
