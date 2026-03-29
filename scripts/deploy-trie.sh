#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TRIE_BIN="$ROOT/out/words-trie.bin"

if [[ ! -f "$TRIE_BIN" ]]; then
  echo "Error: $TRIE_BIN not found. Run 'bash scripts/build-all.sh' first." >&2
  exit 1
fi

TRIE_SIZE=$(stat -c%s "$TRIE_BIN" 2>/dev/null || stat -f%z "$TRIE_BIN")
echo "Trie binary: $TRIE_BIN ($TRIE_SIZE bytes)"

# Deploy to Lexfall
LEXFALL_DIR="${LEXFALL_DIR:-$ROOT/../lexfall}"
if [[ -d "$LEXFALL_DIR/public" ]]; then
  cp "$TRIE_BIN" "$LEXFALL_DIR/public/words-trie.bin"

  TRIE_VER_FILE="$LEXFALL_DIR/src/trie-version.txt"
  CURRENT=$(cat "$TRIE_VER_FILE" 2>/dev/null || echo "0")
  NEXT=$((CURRENT + 1))
  echo "$NEXT" > "$TRIE_VER_FILE"
  echo "Lexfall: copied trie, version $CURRENT -> $NEXT"
else
  echo "Lexfall dir not found at $LEXFALL_DIR — skipping"
fi

# Future: upload to R2 for CDN distribution
# npx wrangler r2 object put "dictionary-assets/words-trie/v${VERSION}.bin" --file "$TRIE_BIN"

echo "=== Trie deployment complete ==="
