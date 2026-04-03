#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT"

mkdir -p out

INPUT="raw-wiktextract-data.jsonl.gz"
URL="https://kaikki.org/dictionary/raw-wiktextract-data.jsonl.gz"

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

echo "=== Step 1: Baseline (JSONL.gz -> state DB) ==="
python scripts/make_small_dictionary.py \
  --input "$INPUT" \
  --state out/state.sqlite3 \
  --mode baseline

echo "=== Step 2: Generate pure DB ==="
python scripts/make_small_dictionary.py \
  --state out/state.sqlite3 \
  --mode generate-pure-db \
  --output out/pure_dictionary.sqlite3

echo "=== Step 3: Extract word list from pure DB ==="
python scripts/make_small_dictionary.py \
  --state out/state.sqlite3 \
  --mode extract-pure-words > out/pure-words.txt

echo "=== Step 4: Compile trie ==="
node scripts/build-trie.mjs \
  --input out/pure-words.txt \
  --out out/words-trie \
  --format bin

echo ""
echo "=== Build complete ==="
echo "Artifacts:"
echo "  out/pure_dictionary.sqlite3  (definitions DB)"
echo "  out/words-trie.bin           (compiled trie)"
echo "  out/words-trie.json          (trie metadata)"
echo "  out/pure-words.txt           (word list)"
