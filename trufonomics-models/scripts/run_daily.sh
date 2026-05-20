#!/bin/zsh
# Daily wrapper for the Thales day-ahead LIVE pilot.
#
# 1) Score yesterday's prediction against today's realized published YoY.
#    No-op (clean exit) if no prediction exists for that target date.
# 2) Generate today's forecast for tomorrow + persist JSON + vintage row.
#
# Idempotent: re-running on the same day overwrites today's prediction
# JSON and the same scoring row. Safe under launchd RunAtLoad replay.
#
# Wired by: ~/Library/LaunchAgents/com.thales.dailyforecast.plist

set -e
set -o pipefail

REPO="/Users/kluless/kairos/trufonomics-models"
UV="/opt/homebrew/bin/uv"
LOG_DIR="$HOME/Library/Logs/thales"
mkdir -p "$LOG_DIR"

cd "$REPO"
echo "── $(date '+%Y-%m-%d %H:%M:%S')  Daily Thales run starting ──"

echo "[1/2] score_yesterday.py"
"$UV" run python scripts/score_yesterday.py

echo "[2/2] forecast_live_tomorrow.py"
"$UV" run python scripts/forecast_live_tomorrow.py

echo "── $(date '+%Y-%m-%d %H:%M:%S')  Daily Thales run done ──"
