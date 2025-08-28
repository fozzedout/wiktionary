#!/usr/bin/env bash
set -euo pipefail

# Tiny smoke test for make_small_dictionary.py against the provided snippet.
# Usage:
#   scripts/smoke_test_small_dictionary.sh [INPUT_JSONL] [fresh]
# Defaults:
#   INPUT_JSONL = snippet-dict.jsonl
#   If second arg is 'fresh', clears previous state/output.

INPUT=${1:-snippet-dict.jsonl}
OUT_DIR=${OUT_DIR:-out}
OUT="$OUT_DIR/small_dictionary.snippet.jsonl"
STATE="$OUT_DIR/state.snippet.sqlite3"
OUT_LM="$OUT_DIR/small_dictionary.snippet.lm.jsonl"

mkdir -p "$OUT_DIR"

if [[ "${2:-}" == "fresh" ]]; then
  echo "Clearing previous state/output in $OUT_DIR"
  rm -f "$OUT" "$STATE"
fi

echo "Baseline load into DB from: $INPUT"

# Optional LM Studio integration via env vars:
#   export LMSTUDIO_URL=http://localhost:1234/v1/chat/completions
#   export LMSTUDIO_MODEL=YourModelName
LM_ARGS=()
if [[ -n "${LMSTUDIO_URL:-}" && -n "${LMSTUDIO_MODEL:-}" ]]; then
  LM_ARGS=(--lmstudio-url "$LMSTUDIO_URL" --lmstudio-model "$LMSTUDIO_MODEL")
fi

python scripts/make_small_dictionary.py \
  --input "$INPUT" \
  --state "$STATE" \
  --mode baseline \
  --max-defs 5 \
  --checkpoint-interval 50

echo "Exporting consolidated JSONL to: $OUT"
python scripts/make_small_dictionary.py \
  --output "$OUT" \
  --state "$STATE" \
  --mode export

echo
echo "Sample output (first 10 words, up to 3 defs each):"
python - "$OUT" <<'PY'
import json, sys, itertools
path = sys.argv[1]
try:
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in zip(range(10), f):
            obj = json.loads(line)
            print(f"- {obj.get('word','?')}:")
            for d in obj.get('definitions', [])[:3]:
                print(f"  {d}")
            print()
except FileNotFoundError:
    print("No output found:", path)
PY

echo "Done. Full output at: $OUT"

if [[ -n "${LMSTUDIO_URL:-}" && -n "${LMSTUDIO_MODEL:-}" ]]; then
  echo
  echo "Enhancing pending words with LM Studio..."
  python scripts/make_small_dictionary.py \
    --state "$STATE" \
    --mode enhance \
    --lmstudio-url "$LMSTUDIO_URL" \
    --lmstudio-model "$LMSTUDIO_MODEL" \
    --max-defs 5 \
    --checkpoint-interval 20

  echo "Exporting LM-enhanced JSONL to: $OUT_LM"
  python scripts/make_small_dictionary.py \
    --output "$OUT_LM" \
    --state "$STATE" \
    --mode export
  echo "LM-enhanced output at: $OUT_LM"
fi
