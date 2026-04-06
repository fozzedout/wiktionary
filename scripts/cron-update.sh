#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT"

STATE_DB="${STATE_DB:-$ROOT/out/stage1.sqlite3}"
STATUS_JSON="${STATUS_JSON:-$ROOT/out/pipeline-status.json}"
UPDATE_PLAN="${UPDATE_PLAN:-$ROOT/out/update-plan.json}"
BUILD_REPORT="${BUILD_REPORT:-$ROOT/out/build-report.json}"
LOCK_FILE="${LOCK_FILE:-$ROOT/out/cron-update.lock}"
LOG_DIR="${LOG_DIR:-$ROOT/out/logs}"
DEPLOY_DB="${DEPLOY_DB:-1}"
DEPLOY_TRIE="${DEPLOY_TRIE:-1}"
MARK_TARGET="${MARK_TARGET:-full}"

mkdir -p "$ROOT/out" "$LOG_DIR"

if command -v flock >/dev/null 2>&1; then
  exec 9>"$LOCK_FILE"
  if ! flock -n 9; then
    echo "Another cron update run is already in progress."
    exit 0
  fi
fi

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_FILE="$LOG_DIR/cron-update-$STAMP.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=== Stage-1 cron update started at $STAMP ==="
echo "State DB: $STATE_DB"

bash "$SCRIPT_DIR/build-all.sh"

python scripts/make_small_dictionary.py \
  --mode pipeline-status \
  --state "$STATE_DB" \
  --output "$STATUS_JSON" >/dev/null

LATEST_EXTRACT_ID="$(python3 - <<'PY' "$STATUS_JSON"
import json, sys
with open(sys.argv[1], 'r', encoding='utf-8') as f:
    data = json.load(f)
latest = data.get('latest_extract') or {}
print(latest.get('id') or '')
PY
)"

PUBLISHED_FULL_EXTRACT_ID="$(python3 - <<'PY' "$STATUS_JSON"
import json, sys
with open(sys.argv[1], 'r', encoding='utf-8') as f:
    data = json.load(f)
published = data.get('published') or {}
full = published.get('full') or {}
print(full.get('extract_id') or '')
PY
)"

if [[ -z "$LATEST_EXTRACT_ID" ]]; then
  echo "No completed extract found after build."
  exit 1
fi

echo "Latest extract id: $LATEST_EXTRACT_ID"
echo "Previously published full extract id: ${PUBLISHED_FULL_EXTRACT_ID:-none}"

if [[ "${PUBLISHED_FULL_EXTRACT_ID:-}" == "$LATEST_EXTRACT_ID" ]]; then
  echo "Latest extract is already published. No deploy needed."
  exit 0
fi

if [[ "$DEPLOY_DB" == "1" ]]; then
  echo "=== Deploying D1 snapshot ==="
  bash "$SCRIPT_DIR/deploy-db.sh"
  python scripts/make_small_dictionary.py \
    --mode mark-published \
    --state "$STATE_DB" \
    --target d1 \
    --extract-id "$LATEST_EXTRACT_ID" \
    --artifact-version "extract-$LATEST_EXTRACT_ID" >/dev/null
fi

if [[ "$DEPLOY_TRIE" == "1" ]]; then
  echo "=== Deploying trie artifact ==="
  bash "$SCRIPT_DIR/deploy-trie.sh"
  python scripts/make_small_dictionary.py \
    --mode mark-published \
    --state "$STATE_DB" \
    --target trie \
    --extract-id "$LATEST_EXTRACT_ID" \
    --artifact-version "extract-$LATEST_EXTRACT_ID" >/dev/null
fi

if [[ "$DEPLOY_DB" == "1" && "$DEPLOY_TRIE" == "1" ]]; then
  python scripts/make_small_dictionary.py \
    --mode mark-published \
    --state "$STATE_DB" \
    --target "$MARK_TARGET" \
    --extract-id "$LATEST_EXTRACT_ID" \
    --artifact-version "extract-$LATEST_EXTRACT_ID" >/dev/null
fi

echo "=== Stage-1 cron update finished successfully ==="
echo "Update plan: $UPDATE_PLAN"
echo "Build report: $BUILD_REPORT"
echo "Status file: $STATUS_JSON"
echo "Log file: $LOG_FILE"
