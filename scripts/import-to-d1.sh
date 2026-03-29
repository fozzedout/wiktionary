#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/worker"
DB_NAME="dictionary-db"

echo "=== Importing words to D1 ==="
TOTAL_W=$(ls "$ROOT"/out/_w_chunk_* | wc -l)
I=0
for chunk in "$ROOT"/out/_w_chunk_*; do
  I=$((I + 1))
  npx wrangler d1 execute "$DB_NAME" --remote --file "$chunk" 2>/dev/null
  echo "  words: $I/$TOTAL_W"
done

echo "=== Importing definitions to D1 ==="
TOTAL_D=$(ls "$ROOT"/out/_d_chunk_* | wc -l)
I=0
for chunk in "$ROOT"/out/_d_chunk_*; do
  I=$((I + 1))
  npx wrangler d1 execute "$DB_NAME" --remote --file "$chunk" 2>/dev/null
  echo "  defs: $I/$TOTAL_D"
done

echo "=== Import complete ==="

# Cleanup
rm -f "$ROOT"/out/_words.sql "$ROOT"/out/_definitions.sql \
      "$ROOT"/out/_w_chunk_* "$ROOT"/out/_d_chunk_*
echo "Cleaned up temp files"
