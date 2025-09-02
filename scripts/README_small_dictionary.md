Small Dictionary Builder (Resumable)

Overview
- Converts a large JSONL of dictionary entries into a compact dictionary stored in SQLite, then optionally exports JSONL.
- Each definition line is <= 25 words and starts with the part of speech in underscores when available (e.g., "_adjective_ Of the color of the raven, jet-black.").
- Two-phase workflow: baseline (fast, no LLM) and enhance (LLM only for words that need it), both resumable.
- English-only: If a language is specified on an entry, only English (`lang == "English"` or `lang_code in {"en", "eng"}`) is accepted; entries without a language tag are allowed.
- Safe to stop/start; state and statuses are stored in SQLite. A single `definitions` table is updated in place when enhanced.
- Optional LM Studio integration to rephrase/condense glosses.

Input Expectations
- Source is JSONL: one JSON object per line.
- The script tries to infer fields across common Wiktionary-like exports:
  - word keys: "word", "title", "headword", "lemma", "entry"
  - pos keys: "pos", "partOfSpeech", "part_of_speech", "part-of-speech"
  - sense containers: "senses", "definitions", "meanings", "glosses", "entries"
  - gloss keys: "gloss", "definition", "def", "sense", "meaning", "desc", "text"

Phases & Run
- No external dependencies required; uses only Python standard library.
- Example commands:

  - Baseline (fast, no LLM):
    python scripts/make_small_dictionary.py \
      --input data/snippet.jsonl \
      --state out/state.sqlite3 \
      --mode baseline \
      --checkpoint-interval 100

- Enhance with LM Studio (OpenAI-compatible API):
  python scripts/make_small_dictionary.py \
    --state out/state.sqlite3 \
    --mode enhance \
    --lmstudio-url http://localhost:1234/v1/chat/completions \
    --lmstudio-model qwen/qwen3-4b-2507
  # To only enhance items explicitly marked for reprocessing (e.g., by verify-lm):
  python scripts/make_small_dictionary.py \
    --state out/state.sqlite3 \
    --mode enhance \
    --only-reprocess \
    --lmstudio-url http://localhost:1234/v1/chat/completions \
    --lmstudio-model qwen/qwen3-4b-2507

  - Report progress (counts only):
    python scripts/make_small_dictionary.py --state out/state.sqlite3 --mode report --input dummy --output dummy

  - Export consolidated JSONL from DB:
    python scripts/make_small_dictionary.py \
      --output out/small_dictionary.final.jsonl \
      --state out/state.sqlite3 \
      --mode export

- Progress output: Periodic progress lines are printed during baseline and enhance runs (done/pending counts, line number).
- Stop anytime with Ctrl-C; the script will checkpoint the current line.
- Resume by rerunning with the same --state path.
- In baseline mode, LM status is set per word:
  - status "done" if all chosen first-sentence glosses are already ≤ max words (no LLM needed)
  - status "pending" otherwise (eligible for later enhancement)
  - Enhance mode only processes words with status "pending" and marks them "done".

Database Schema
- words(word, status, updated_at)
- definitions(word, idx, pos, source_first_sentence, current_line)

Exported JSONL
- One object per unique word:
  {"word": "raven", "definitions": ["_adjective_ Of the color of the raven, jet-black.", "_noun_ A large, glossy-black passerine bird."]}

Notes
- Deduplication is case-insensitive (e.g., "Raven" and "raven" are considered the same word key).
- In baseline mode, no LM calls are made.
- In enhance mode, only words marked "pending" are sent to LM; results are updated in-place in the `definitions` table.
- Use `--only-reprocess` to restrict enhance to words that were explicitly flagged for reprocessing (tracked via an internal `reprocess` flag). The `verify-lm` mode sets this flag when it finds issues.
- Checkpointing is by input line number; on resume, the script skips that many lines before continuing.

Verification
- Rule-based verify summary (no LLM):
  python scripts/make_small_dictionary.py --state out/state.sqlite3 --mode verify --max-words 25 --verify-sample 10 --output out/verify_report.json
- LLM-based verify (uses a more accurate model via LM Studio):
  export LMSTUDIO_URL=http://localhost:1234/v1/chat/completions
  export LMSTUDIO_MODEL=YourAccurateModel
  python scripts/make_small_dictionary.py --state out/state.sqlite3 --mode verify-lm --max-words 25 --verify-sample 10 --lmstudio-url $LMSTUDIO_URL --lmstudio-model $LMSTUDIO_MODEL --output out/verify_lm_report.json
- The LLM judge checks that each line is faithful and concise relative to the stored source sentence and reports failures with reasons.

Compact Final DB (space-optimized)
- To create a minimal SQLite for deployment (no statuses or source sentences), run:
  python scripts/make_small_dictionary.py \
    --state out/state.sqlite3 \
    --output out/dictionary.compact.sqlite3 \
    --mode compact
- Minimal schema:
  - terms(id INTEGER PRIMARY KEY, word TEXT UNIQUE)
  - defs(term_id INTEGER, idx INTEGER, line TEXT, PRIMARY KEY(term_id, idx)) WITHOUT ROWID
- Query pattern for API:
  - SELECT id FROM terms WHERE word=?;
  - SELECT line FROM defs WHERE term_id=? ORDER BY idx;
