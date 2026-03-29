#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DB_FILE="$ROOT/out/pure_dictionary.sqlite3"
DB_NAME="dictionary-db"
WORKER_DIR="$ROOT/worker"

if [[ ! -f "$DB_FILE" ]]; then
  echo "Error: $DB_FILE not found. Run 'bash scripts/build-all.sh' first." >&2
  exit 1
fi

cd "$WORKER_DIR"

echo "=== Deploying dictionary to D1: $DB_NAME ==="

# Create tables (idempotent)
npx wrangler d1 execute "$DB_NAME" --remote --command \
  "CREATE TABLE IF NOT EXISTS words (word TEXT PRIMARY KEY, raw TEXT);"
npx wrangler d1 execute "$DB_NAME" --remote --command \
  "CREATE TABLE IF NOT EXISTS definitions (word TEXT NOT NULL, idx INTEGER NOT NULL, pos TEXT, definition TEXT NOT NULL, PRIMARY KEY (word, idx));"

# Clear existing data
echo "Clearing existing data..."
npx wrangler d1 execute "$DB_NAME" --remote --command "DELETE FROM definitions;"
npx wrangler d1 execute "$DB_NAME" --remote --command "DELETE FROM words;"

# Export to SQL using Python (sqlite3 CLI may not be installed)
echo "Exporting SQL..."
python3 "$ROOT/scripts/dump-pure-db.py" "$DB_FILE" "$ROOT/out"

WORDS_COUNT=$(wc -l < "$ROOT/out/_words.sql")
DEFS_COUNT=$(wc -l < "$ROOT/out/_definitions.sql")
echo "  $WORDS_COUNT word rows, $DEFS_COUNT definition rows"

# Import in chunks
CHUNK_SIZE=5000

echo "Importing words..."
split -l "$CHUNK_SIZE" "$ROOT/out/_words.sql" "$ROOT/out/_w_chunk_"
TOTAL_W=$(ls "$ROOT"/out/_w_chunk_* | wc -l)
I=0
for chunk in "$ROOT"/out/_w_chunk_*; do
  I=$((I + 1))
  npx wrangler d1 execute "$DB_NAME" --remote --file "$chunk" 2>/dev/null
  echo "  words: $I/$TOTAL_W"
done

echo "Importing definitions..."
split -l "$CHUNK_SIZE" "$ROOT/out/_definitions.sql" "$ROOT/out/_d_chunk_"
TOTAL_D=$(ls "$ROOT"/out/_d_chunk_* | wc -l)
I=0
for chunk in "$ROOT"/out/_d_chunk_*; do
  I=$((I + 1))
  npx wrangler d1 execute "$DB_NAME" --remote --file "$chunk" 2>/dev/null
  echo "  defs: $I/$TOTAL_D"
done

# Cleanup temp files
rm -f "$ROOT"/out/_words.sql "$ROOT"/out/_definitions.sql \
      "$ROOT"/out/_w_chunk_* "$ROOT"/out/_d_chunk_*

echo "=== D1 deployment complete ==="
