#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT"

mkdir -p out

INPUT="${INPUT:-raw-wiktextract-data.jsonl.gz}"
URL="${URL:-https://kaikki.org/dictionary/raw-wiktextract-data.jsonl.gz}"
STATE_DB="${STATE_DB:-out/stage1.sqlite3}"
FULL_DB="${FULL_DB:-out/current_dictionary.sqlite3}"
PURE_DB="${PURE_DB:-out/pure_dictionary.sqlite3}"
PURE_WORDS="${PURE_WORDS:-out/pure-words.txt}"
UPDATE_PLAN="${UPDATE_PLAN:-out/update-plan.json}"
BUILD_REPORT="${BUILD_REPORT:-out/build-report.json}"
TRIE_BASE="${TRIE_BASE:-out/words-trie}"

echo "=== Step 0: Ensure latest Wiktionary extract ==="
if [ -f "$INPUT" ]; then
  echo "Local file found, checking for updates..."
  HTTP_CODE=$(curl -sSL -o "$INPUT.tmp" -w "%{http_code}" -z "$INPUT" "$URL")
  if [ "$HTTP_CODE" = "200" ]; then
    mv "$INPUT.tmp" "$INPUT"
    echo "Updated to latest version."
  else
    rm -f "$INPUT.tmp"
    echo "Already up to date."
  fi
else
  echo "Downloading Wiktionary extract..."
  curl -sSL -o "$INPUT" "$URL"
  echo "Download complete."
fi

echo "=== Step 1: Ingest extract (versioned) ==="
python scripts/make_small_dictionary.py \
  --input "$INPUT" \
  --state "$STATE_DB" \
  --mode ingest-extract

echo "=== Step 2: Plan update (diff against prior) ==="
python scripts/make_small_dictionary.py \
  --state "$STATE_DB" \
  --mode plan-update \
  --output "$UPDATE_PLAN"

echo "=== Step 3: Enhance changed definitions ==="
python scripts/make_small_dictionary.py \
  --state "$STATE_DB" \
  --mode enhance-changed

echo "=== Step 4: Export current snapshot ==="
python scripts/make_small_dictionary.py \
  --state "$STATE_DB" \
  --mode export-current \
  --output "$FULL_DB" \
  --pure-db "$PURE_DB" \
  --pure-words "$PURE_WORDS"

echo "=== Step 5: Compile trie ==="
node scripts/build-trie.mjs \
  --input "$PURE_WORDS" \
  --out "$TRIE_BASE" \
  --format bin

echo "=== Step 6: Build stage-1 report ==="
python scripts/make_small_dictionary.py \
  --state "$STATE_DB" \
  --mode build-report \
  --output "$BUILD_REPORT" \
  --export-db "$FULL_DB" \
  --pure-db "$PURE_DB" \
  --trie-manifest "${TRIE_BASE}.manifest.json"

echo ""
echo "=== Build complete ==="
echo "Artifacts:"
echo "  $FULL_DB (full dictionary DB)"
echo "  $PURE_DB (pure words DB for D1)"
echo "  ${TRIE_BASE}.bin (compiled trie)"
echo "  ${TRIE_BASE}.json (trie metadata)"
echo "  ${TRIE_BASE}.manifest.json (trie manifest)"
echo "  $PURE_WORDS (word list)"
echo "  $UPDATE_PLAN (diff report)"
echo "  $BUILD_REPORT (stage-1 report)"

if [[ "${SKIP_DEPLOY:-}" == "1" ]]; then
  echo ""
  echo "=== Skipping deployment (SKIP_DEPLOY=1) ==="
  exit 0
fi

echo "=== Step 7: Generate deploy SQL ==="
python scripts/make_small_dictionary.py \
  --state "$STATE_DB" \
  --mode generate-deploy-sql \
  --pure-db "$PURE_DB" \
  --output out

# Check if DB needs deploying
DEPLOY_OPS=$(python3 -c "import json; print(json.load(open('out/_deploy_meta.json'))['total_operations'])")

if [[ "$DEPLOY_OPS" -gt 0 ]]; then
  echo "=== Step 8: Deploy DB to D1 ($DEPLOY_OPS operations) ==="
  bash "$SCRIPT_DIR/deploy-db.sh"

  echo "Marking D1 as published..."
  python scripts/make_small_dictionary.py \
    --state "$STATE_DB" \
    --mode mark-published \
    --target d1
else
  echo "=== Step 8: DB already up to date — skipping ==="
fi

echo "=== Step 9: Deploy API Worker ==="
cd "$ROOT/worker"
npx wrangler deploy
cd "$ROOT"

# Only publish trie if words were added or removed
WORDS_ADDED=$(python3 -c "import json; m=json.load(open('out/_deploy_meta.json')); print(m.get('words_inserted',0) + m.get('words_deleted',0))")
if [[ "$WORDS_ADDED" -gt 0 ]]; then
  echo "=== Step 10: Publish trie ==="
  bash "$SCRIPT_DIR/deploy-trie.sh"
else
  echo "=== Step 10: No words added/removed — trie unchanged ==="
fi

echo ""
echo "=== Pipeline complete ==="
