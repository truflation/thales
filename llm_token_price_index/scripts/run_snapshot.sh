#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/logs"
LOG_FILE="${LOG_DIR}/snapshot.log"
SNAPSHOT_DATE="$(date -u +%F)"

mkdir -p "${LOG_DIR}"
cd "${ROOT_DIR}"

{
  echo "[$(date -u +%FT%TZ)] starting token price index snapshot ${SNAPSHOT_DATE}"
  python scripts/build_token_price_index.py \
    --snapshot-date "${SNAPSHOT_DATE}" \
    --historical-backfill
  echo "[$(date -u +%FT%TZ)] finished token price index snapshot ${SNAPSHOT_DATE}"
} >> "${LOG_FILE}" 2>&1
