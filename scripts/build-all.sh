#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT"

mkdir -p out

echo "=== Step 1: Baseline (JSONL.gz -> state DB) ==="
python scripts/make_small_dictionary.py \
  --input raw-wiktextract-data.jsonl.gz \
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
