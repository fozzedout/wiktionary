# Wiktionary Dictionary Service

Central dictionary service for word games. Processes raw Wiktionary data into two artifacts:

- **Compiled trie** (`words-trie.bin`) — downloadable binary for client-side word validation
- **Definition API** — Cloudflare Worker + D1 serving short definitions via authenticated API

## Setup

```bash
npm install
npx wrangler login
```

## Build Pipeline

Processes `raw-wiktextract-data.jsonl.gz` (download from https://kaikki.org/dictionary/raw-wiktextract-data.jsonl.gz) through four steps:

```bash
bash scripts/build-all.sh
```

This runs:
1. **Baseline** — extracts English words and definitions from the raw JSONL.gz into a state DB (~15 min)
2. **Generate pure DB** — filters to pure alpha words, produces `out/pure_dictionary.sqlite3`
3. **Extract word list** — dumps word list to `out/pure-words.txt`
4. **Compile trie** — builds `out/words-trie.bin` (radix trie, WTRI v1 format)

Individual steps can be run separately:

```bash
# Baseline only (resumable)
python scripts/make_small_dictionary.py --input raw-wiktextract-data.jsonl.gz --state out/state.sqlite3 --mode baseline

# LLM enhancement (requires LM Studio running locally)
python scripts/make_small_dictionary.py --state out/state.sqlite3 --mode enhance

# Rebuild pure DB + trie from existing state
python scripts/make_small_dictionary.py --state out/state.sqlite3 --mode generate-pure-db --output out/pure_dictionary.sqlite3
python scripts/make_small_dictionary.py --state out/state.sqlite3 --mode extract-pure-words > out/pure-words.txt
node scripts/build-trie.mjs --input out/pure-words.txt --out out/words-trie --format bin
```

## Deploy

### First-time D1 setup

```bash
npx wrangler d1 create dictionary-db
# Paste the database_id into worker/wrangler.toml
```

### Set API keys

```bash
cd worker
npx wrangler secret put API_KEYS
# Enter comma-separated API keys when prompted
```

### Deploy everything

```bash
bash scripts/deploy-all.sh
```

Or individually:

```bash
# Upload definitions to D1 (~20 min for 848K words)
bash scripts/deploy-db.sh

# Deploy the API worker
cd worker && npx wrangler deploy

# Copy trie to consumer repos (e.g. Lexfall)
bash scripts/deploy-trie.sh
```

## API Endpoints

Base URL: `https://dictionary-api.xefig.workers.dev`

All `/api/*` routes require an `X-API-Key` header.

| Endpoint | Description |
|----------|-------------|
| `GET /` | Front-end dictionary tester |
| `GET /health` | Health check (no auth) |
| `GET /api/definitions?word=test` | Look up word definitions |
| `GET /api/validate?word=test` | Check if word exists in dictionary |

### Example

```bash
curl -H "X-API-Key: YOUR_KEY" "https://dictionary-api.xefig.workers.dev/api/definitions?word=hello"
```

```json
{
  "word": "hello",
  "found": true,
  "definitions": [
    { "definition": "A greeting used when meeting someone", "pos": "interjection" }
  ]
}
```

## Artifacts

| File | Size | Description |
|------|------|-------------|
| `out/state.sqlite3` | ~288MB | Working DB with processing state, all words |
| `out/pure_dictionary.sqlite3` | ~139MB | Deployment DB (alpha-only words + definitions) |
| `out/words-trie.bin` | ~21MB | Compiled radix trie (WTRI v1 binary) |
| `out/pure-words.txt` | ~9MB | Plain text word list (one per line) |

## Consumers

- **Lexfall** — first consumer. `deploy-trie.sh` copies the trie to `../lexfall/public/` and bumps the trie version.
- Future word games consume the trie binary for validation and call the API for definitions.
