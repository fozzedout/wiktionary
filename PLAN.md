# Wiktionary Service Implementation Plan

## Purpose

This plan turns the current repository into a production-capable dictionary build and delivery pipeline in three stages:

1. Process each new Wiktionary extract, update only changed definitions, and publish an updated D1 database and trie.
2. Harden the API so real client projects can consume it safely and predictably.
3. Ship a simple public-facing dictionary search demo that exercises the same backend used by real customers.

The sequencing matters. Stage 1 is the core business system. Stage 2 makes it safe to expose. Stage 3 is presentation and customer acquisition.

## Current State Summary

The repository already contains:

- A resumable SQLite-backed builder in [scripts/make_small_dictionary.py](/home/paul/source/wiktionary/scripts/make_small_dictionary.py).
- A trie compiler in [scripts/build-trie.mjs](/home/paul/source/wiktionary/scripts/build-trie.mjs).
- A D1 deploy script in [scripts/deploy-db.sh](/home/paul/source/wiktionary/scripts/deploy-db.sh).
- A Cloudflare Worker API in [worker/src/index.js](/home/paul/source/wiktionary/worker/src/index.js).
- A simple demo frontend in [worker/src/frontend.html](/home/paul/source/wiktionary/worker/src/frontend.html).

The main gaps relative to the desired roadmap are:

- The builder produces multiple definitions per word instead of a single pocket-dictionary definition.
- The build pipeline does not model extracts as versions.
- There is no durable change-detection layer between one extract and the next.
- LLM work is tied to the current mutable state DB rather than to stable source fingerprints.
- D1 deployment is full-reload and non-atomic.
- The API is not yet ready for external client use.
- The demo frontend exists before the service layer is hardened.

## Guiding Decisions

These decisions should be treated as constraints unless a later design review explicitly changes them.

### 1. Source-of-truth model

The state database should become the canonical build ledger. It should remember:

- Which extract was processed.
- Which words existed in that extract.
- Which source definitions existed for each word.
- Which source definitions changed between extracts.
- Which LLM outputs are still valid for unchanged source definitions.

### 2. Stable change detection

Incremental processing should use normalized hashes, not timestamps or row positions.

Use hashes for:

- Whole extract identity.
- Word-level source identity.
- Definition-level source identity.
- Optional pure-word-set identity for trie versioning.

### 3. Single pocket-dictionary definition per word

The final output for each word should be a single short definition, not a list of numbered senses. Think of what you would find in a pocket dictionary: one concise line that captures the word's core meaning(s). When a word has multiple distinct senses, the LLM should synthesize them into a single compact summary rather than preserving each sense as a separate entry.

The source pipeline still ingests and stores all source definitions individually (they are needed for change detection), but the enhancement step combines them into one definition per word.

### 4. Separation of concerns

Keep these concerns separate in the data model and code:

- Raw source ingestion.
- Canonical normalized definitions.
- LLM-synthesized pocket definitions.
- Deployment snapshots.

Do not continue folding all of that into a single mutable `definitions.current_line` field.

### 5. Incremental first, not “fast full rebuild”

The stage-1 goal is not merely “rerun faster”. It is:

- Reuse prior work for unchanged definitions.
- Re-run LLM only where source changed or the enhancement policy changed.
- Publish only the net changes needed for D1.

Trie generation can remain full rebuild initially, because it is simpler and likely cheap relative to the definition pipeline.

### 6. Demo follows service readiness

The public demo should consume the same API behavior, auth rules, and data freshness guarantees as first-party clients. It must not depend on special-case shortcuts that bypass production behavior.

### 7. API-First Startup, Trie In Background

Client applications should be usable before the trie finishes downloading.

The intended behavior is:

- the API is available immediately for definition lookup
- the API can act as the initial validation fallback
- the trie downloads in the background after the app becomes usable
- once the trie is ready, the client can switch to local validation for speed and offline-friendliness
- the trie should be cached locally so it is not re-downloaded unnecessarily

This means the trie is a performance optimization, not a hard startup dependency.

## Stage 1: Incremental Extract Processing and Artifact Publishing

### Outcome

Given a new Wiktionary extract, the system should:

- Identify whether the extract is new.
- Parse it into a canonical source snapshot.
- Detect new, changed, unchanged, and removed words/definitions.
- Reuse prior enhanced definitions where the source is unchanged.
- Re-run LLM enhancement only for changed definitions or when policy/version changes.
- Produce an updated pure-word SQLite DB.
- Produce an updated trie binary.
- Publish the DB to D1 safely.

### Deliverables

- Versioned extract tracking in the state DB.
- Canonical source definition hashing.
- Incremental enhancement planner.
- Incremental deployment planner for D1.
- Build report summarizing changes between extracts.
- New CLI modes and scripts for stage-1 operations.

### Stage-1 Data Model Changes

Extend the state DB with explicit versioned tables instead of continuing to overload the current `words` and `definitions` tables.

#### New tables

`extract_runs`

- `id INTEGER PRIMARY KEY`
- `extract_key TEXT UNIQUE NOT NULL`
- `source_path TEXT NOT NULL`
- `source_url TEXT`
- `file_size INTEGER`
- `sha256 TEXT NOT NULL`
- `etag TEXT`
- `last_modified TEXT`
- `started_at INTEGER NOT NULL`
- `completed_at INTEGER`
- `status TEXT NOT NULL CHECK(status IN ('running','complete','failed'))`
- `builder_version TEXT NOT NULL`
- `notes TEXT`

`source_words`

- `extract_id INTEGER NOT NULL`
- `word_key TEXT NOT NULL`
- `display_word TEXT`
- `language_code TEXT`
- `word_hash TEXT NOT NULL`
- `is_pure_alpha INTEGER NOT NULL`
- `PRIMARY KEY (extract_id, word_key)`

`source_definitions`

- `extract_id INTEGER NOT NULL`
- `word_key TEXT NOT NULL`
- `definition_key TEXT NOT NULL`
- `source_idx INTEGER NOT NULL`
- `pos TEXT`
- `source_first_sentence TEXT NOT NULL`
- `source_hash TEXT NOT NULL`
- `normalized_source TEXT NOT NULL`
- `PRIMARY KEY (extract_id, word_key, definition_key)`

`enhanced_words`

- `word_key TEXT NOT NULL`
- `word_hash TEXT NOT NULL` — hash over all source definitions for this word
- `enhancement_version TEXT NOT NULL`
- `pocket_definition TEXT NOT NULL` — single synthesized definition for the word
- `source_pos_list TEXT` — JSON array of distinct parts of speech from source
- `source_definition_count INTEGER NOT NULL` — how many source defs were combined
- `enhanced_source TEXT NOT NULL` — which LLM produced this
- `quality_status TEXT NOT NULL CHECK(quality_status IN ('accepted','needs_review','failed'))`
- `updated_at INTEGER NOT NULL`
- `PRIMARY KEY (word_key, word_hash, enhancement_version)`

`word_snapshots`

- `extract_id INTEGER NOT NULL`
- `word_key TEXT NOT NULL`
- `display_word TEXT`
- `word_status TEXT NOT NULL CHECK(word_status IN ('new','changed','unchanged','deleted'))`
- `source_definition_count INTEGER NOT NULL`
- `enhancement_action TEXT CHECK(enhancement_action IN ('reused','regenerate','pending'))` — whether the pocket definition was reused or needs regeneration
- `PRIMARY KEY (extract_id, word_key)`

`deploy_runs`

- `id INTEGER PRIMARY KEY`
- `extract_id INTEGER NOT NULL`
- `target TEXT NOT NULL CHECK(target IN ('d1','trie','worker','full'))`
- `started_at INTEGER NOT NULL`
- `completed_at INTEGER`
- `status TEXT NOT NULL CHECK(status IN ('running','complete','failed'))`
- `details_json TEXT`

`published_state`

- `target TEXT PRIMARY KEY`
- `extract_id INTEGER NOT NULL`
- `published_at INTEGER NOT NULL`
- `artifact_version TEXT NOT NULL`

#### Existing tables

Retain current `words`, `definitions`, and `progress` only as transitional compatibility tables during migration. Long term:

- `words` and `definitions` should either become derived views or be removed after the new pipeline is stable.
- `progress` should be scoped by extract and mode, not kept as one global offset.

### Stage-1 Hashing Rules

Define hash behavior up front so the system does not churn unnecessarily.

#### Extract hash

Compute SHA-256 over the raw compressed file contents. This is the primary “is this extract new?” signal.

#### Word hash

For each `word_key`, compute a deterministic hash over:

- normalized `word_key`
- preferred `display_word`
- the ordered list of definition hashes for that word

This is the primary “did anything about this word change?” signal.

#### Definition hash

For each source definition, hash:

- normalized `word_key`
- normalized `pos`
- normalized `source_first_sentence`

Normalization rules should be explicit and stable:

- trim whitespace
- collapse internal whitespace
- lowercase `word_key`
- lowercase and normalize `pos`
- strip trivial markup artifacts before hashing

Do not include ephemeral ordering from the raw extract unless the order is semantically meaningful and stable.

### Stage-1 Code Changes

#### 1. Introduce extract-aware ingestion

Refactor [scripts/make_small_dictionary.py](/home/paul/source/wiktionary/scripts/make_small_dictionary.py) so baseline ingestion becomes extract-versioned.

Add a new mode:

- `ingest-extract`

Responsibilities:

- Validate input file presence.
- Compute file hash and metadata.
- Create an `extract_runs` row.
- Stream the JSONL.gz once.
- Materialize `source_words` and `source_definitions` for that extract.
- Mark the run complete or failed.

Important implementation detail:

- Do not mutate prior extract data in place.
- Every extract should produce a complete logical snapshot.

#### 2. Build a diff phase

Add a new mode:

- `plan-update`

Responsibilities:

- Compare the new extract against the most recent completed prior extract.
- For each word, classify status as `new`, `changed`, `unchanged`, or `deleted`.
- For each definition under a changed word, classify whether the source hash is new, changed, unchanged, or removed.
- Populate `word_snapshots`.
- Emit a machine-readable report such as `out/update-plan.json`.

Report contents should include:

- extract id and prior extract id
- total words in new extract
- count of unchanged words (pocket definition can be reused)
- count of changed words (pocket definition must be regenerated)
- count of new words (pocket definition must be generated)
- count of deleted words
- total words requiring LLM work
- pure-word count delta

#### 3. Split LLM enhancement from source ingestion

The current pipeline mixes source and current output in the same rows. Replace that with a source-to-enhancement join model.

Add a new mode:

- `enhance-changed`

Responsibilities:

- For each word with changed or new source definitions, gather all source definitions for that word.
- Synthesize a single pocket-dictionary definition from the full set of source definitions. The LLM prompt should instruct the model to produce one concise line — the kind of definition you would find in a pocket dictionary — covering the word's core meaning(s).
- Reuse prior `pocket_definition` from `enhanced_words` when the word's `word_hash` (derived from all its source definitions) is unchanged and the `enhancement_version` matches.
- Generate fresh LLM output only for words whose source definitions changed or whose enhancement version is outdated.
- Record quality status and enhancement source metadata.

The LLM prompt should:

- Receive all source definitions for the word as input context.
- Produce a single definition line, not a numbered list.
- Capture the most important sense(s) in natural language.
- Stay under a target word count (e.g. 25-30 words).
- Not include examples, etymologies, or cross-references.

This mode should support:

- dry run
- bounded batch size
- resume after interruption
- optional verification after enhancement

#### 4. Introduce explicit enhancement versioning

Add a constant in the builder such as:

- `ENHANCEMENT_VERSION = "pocket_v1"`

This version should change when:

- prompts materially change
- normalization rules change
- post-processing logic changes
- validation rules change in a way that should invalidate old output

This allows safe reuse of old LLM results until the policy changes.

#### 5. Build a canonical export phase

Add a new mode:

- `export-current`

Responsibilities:

- Resolve the latest completed extract.
- Join `source_words` and `enhanced_words`.
- Each word produces exactly one row in the output: the word and its single pocket definition.
- Prefer the `pocket_definition` from `enhanced_words` when available and accepted.
- Fall back to the first source definition when no enhancement exists yet.
- Produce:
  - `out/current_dictionary.sqlite3` — one definition per word
  - `out/pure_dictionary.sqlite3` — same, filtered to pure-alpha words
  - `out/pure-words.txt`

The output schema should be simplified to reflect one definition per word:

```sql
CREATE TABLE words (word TEXT PRIMARY KEY, definition TEXT NOT NULL);
```

This replaces the current multi-definition model and the ambiguous coupling between `definitions.current_line` and the final export.

#### 6. Keep trie generation simple in stage 1

Continue using [scripts/build-trie.mjs](/home/paul/source/wiktionary/scripts/build-trie.mjs) against `out/pure-words.txt`.

Changes needed:

- add metadata output that includes the extract id and source hash
- write a version manifest alongside the trie
- make the output artifact name deterministic per extract if desired

Example metadata file:

- `out/words-trie.manifest.json`

Containing:

- extract id
- source extract hash
- pure-word count
- trie build timestamp
- trie binary filename

#### 7. Replace the current monolithic build script

Retire the current linear behavior in [scripts/build-all.sh](/home/paul/source/wiktionary/scripts/build-all.sh) and replace it with an extract-aware sequence:

1. fetch or validate extract
2. ingest extract
3. diff against prior extract
4. enhance changed definitions
5. export current DBs
6. build trie
7. write build report

The script should fail fast and log enough metadata that reruns are explainable.

### Stage-1 Deployment Changes

#### D1 deployment target

Stage 1 still needs a safer D1 update path than the current truncate-and-reload behavior.

Short-term target:

- continue publishing a full logical snapshot to D1
- avoid deleting the live dataset before the replacement is ready

Recommended implementation:

1. Create a staging table in D1:
   - `words_next` (word TEXT PRIMARY KEY, definition TEXT NOT NULL)
2. Import the new snapshot into the staging table.
3. Validate row counts and spot-check key queries.
4. Swap staging into the live name in a controlled sequence.
5. Keep the previous live table until swap succeeds, then clean up.

If D1 table renames are awkward, use versioned tables:

- `words_v{extract_id}`

Then maintain a view:

- `words`

that points to the current version. The plan should choose one of these patterns after a small D1 capability spike.

#### D1 delta deployment

Full staging-and-swap is sufficient for the first stage-1 milestone. After that is stable, add optional delta deployment.

Delta deployment design:

- compute inserts, updates, and deletes between the last published extract and the new extract
- emit SQL patch files:
  - `_upsert_words.sql`
  - `_delete_words.sql`
- apply patches inside bounded batches

Do not start with delta deployment if it risks delaying stable extract-diffing. The durable value is incremental LLM reuse, not SQL cleverness.

### Stage-1 CLI and Script Plan

Add or update the following commands:

- `python scripts/make_small_dictionary.py --mode ingest-extract --input raw-wiktextract-data.jsonl.gz --state out/state.sqlite3`
- `python scripts/make_small_dictionary.py --mode plan-update --state out/state.sqlite3 --output out/update-plan.json`
- `python scripts/make_small_dictionary.py --mode enhance-changed --state out/state.sqlite3 --lmstudio-url ... --lmstudio-model ...`
- `python scripts/make_small_dictionary.py --mode export-current --state out/state.sqlite3 --output out/current_dictionary.sqlite3`
- `bash scripts/build-all.sh`
- `bash scripts/deploy-db.sh`
- `bash scripts/deploy-all.sh`

Update [package.json](/home/paul/source/wiktionary/package.json) scripts to match the new lifecycle.

### Stage-1 Testing Plan

#### Unit-level tests

Add tests for:

- word normalization
- definition normalization
- hash generation stability
- extract diff classification
- pocket definition reuse logic (word_hash unchanged = reuse)
- export fallback behavior when no pocket definition exists yet
- output contains exactly one definition per word

#### Fixture-based integration tests

Create tiny fixture extracts representing:

- unchanged extract rerun
- one changed definition
- one added word
- one removed word
- casing-only changes
- punctuation-only changes that should not trigger re-enhancement

Each fixture test should assert:

- expected diff counts
- expected LLM work count (one call per changed/new word, not per definition)
- expected export rows (one per word)
- each exported word has exactly one definition
- expected pure-word outputs

#### Deploy tests

Before touching remote D1, add a local SQL patch verification step:

- generate export DB
- dump SQL
- verify row counts and uniqueness
- validate sample lookups

### Stage-1 Observability

Every build should emit a summary report, for example `out/build-report.json`, containing:

- extract identity
- prior extract identity
- words new/changed/unchanged/deleted
- pocket definitions reused/regenerated
- total LLM requests
- total LLM failures
- pure-word count
- trie metadata
- export DB row counts
- deploy result status

Also print a human-readable summary to stdout/stderr for operator use.

### Stage-1 Migration Plan

Implement the transition in these steps:

1. Add new tables and hash helpers without removing current behavior.
2. Add `ingest-extract` writing the new tables.
3. Add `plan-update`.
4. Add `enhance-changed` with reusable enhancement cache.
5. Add `export-current`.
6. Switch `build-all.sh` to the new modes.
7. Switch `deploy-db.sh` to staged deployment.
8. Remove or deprecate old baseline/export assumptions once the new path is verified.

### Stage-1 Acceptance Criteria

Stage 1 is done when all of the following are true:

- Running the pipeline twice on the same extract performs zero LLM work on the second run.
- A small fixture extract with one changed definition only re-generates the pocket definition for that word.
- Each word in the output DB has exactly one pocket-dictionary-style definition.
- Pure-word DB and trie are produced from the latest accepted snapshot.
- D1 can be updated without clearing the live dataset first.
- The build report clearly shows the delta between extracts.
- A failed enhancement or deploy can be retried without rebuilding everything from scratch.

## Stage 2: API Readiness for Real Clients

### Outcome

The API should become a reliable shared service that first-party projects can depend on.

### Deliverables

- secure auth model
- rate limiting
- versioned API behavior
- operational health endpoints
- customer-safe error responses
- deployment and rollback procedure

### Client Startup Flow

Stage 2 should explicitly support this runtime flow:

1. app loads
2. API-backed lookup is immediately usable
3. trie download begins in the background
4. client switches to local trie validation when ready

That makes the API the availability path and the trie the fast path.

### Stage-2 Work Items

#### 1. Remove committed secrets and demo backdoors

Change [worker/wrangler.toml](/home/paul/source/wiktionary/worker/wrangler.toml) so it no longer contains live API keys.

Required changes:

- remove `API_KEYS` from checked-in vars
- remove `FRONTEND_TEST_KEY` from checked-in vars
- treat all real keys as Wrangler secrets
- stop embedding a broadly usable key into the public HTML

#### 2. Tighten auth behavior

Update [worker/src/auth.js](/home/paul/source/wiktionary/worker/src/auth.js):

- require header-based auth only
- remove `api_key` query parameter support
- normalize error codes and messages
- add optional key metadata support later if per-customer limits are needed

Future-ready schema:

- `api_clients(id, name, key_hash, status, tier, created_at)`

Even if this is not implemented in D1 immediately, the auth code should not block moving in that direction.

#### 3. Implement actual rate limiting

The current `RATE_LIMIT_RPM` variable is configuration without enforcement.

Choose a Cloudflare-native approach:

- Durable Object token bucket
- per-key sliding window in KV or D1
- Cloudflare native product if available and appropriate

For first-party clients, start with:

- per-key requests per minute
- optional burst allowance
- explicit `429` response contract

#### 4. Define response contracts

Lock down the response shapes for:

- `GET /health`
- `GET /api/definitions`
- `GET /api/validate`

Add consistency for:

- content type
- cache headers
- not-found responses
- invalid-input responses
- auth failures
- rate-limit failures

Add a lightweight API version strategy:

- response header like `X-Dictionary-Version`
- optional extract id / publish version in health and response metadata

#### 5. Add service metadata endpoints

Add endpoints such as:

- `GET /health`
- `GET /version`
- `GET /ready`

Suggested payloads:

- current extract id
- published at timestamp
- trie version
- D1 row counts
- worker version

#### 6. Add operational safeguards

Implement:

- structured request logging
- build/deploy version headers
- correlation id support
- better timeout/error handling around D1 queries

#### 7. Add client-focused tests

Add integration tests covering:

- valid definition lookup
- not-found lookup
- invalid word parameter
- missing auth
- invalid auth
- rate limit exceeded
- CORS behavior for the approved frontend surface

### Stage-2 Acceptance Criteria

Stage 2 is done when:

- no live keys are committed to the repository
- the public demo no longer exposes reusable API credentials
- client projects can authenticate with stable header-based auth
- rate limiting is enforced
- the API publishes version/build metadata
- failures are predictable and documented

## Stage 3: Public Demo and Customer Enticement

### Outcome

Expose a simple, attractive dictionary search experience that demonstrates the product without undermining the production API model.

### Deliverables

- a clean search page
- a constrained public lookup mode
- product messaging
- usage analytics

The demo should follow the same pattern:

- page is usable immediately through the API
- any large client validation asset should be optional, background-loaded, and cached

### Stage-3 Work Items

#### 1. Rework the demo frontend

Replace the current barebones tester in [worker/src/frontend.html](/home/paul/source/wiktionary/worker/src/frontend.html) with a deliberate demo experience.

Functional requirements:

- search by word
- show the pocket definition
- show not-found state
- mobile-friendly layout
- fast loading

Non-functional requirements:

- no exposed private customer key
- no special privileged route that bypasses service rules
- clear branding and positioning

#### 2. Introduce a public demo access model

The public demo should not use the same auth path as real customer integrations.

Recommended options:

- a dedicated demo route with a separate rate limit bucket
- a server-side internal call path inside the same worker
- a signed demo session or captcha gate if abuse becomes a problem

The simplest initial design is:

- public route handled inside the worker
- stricter rate limit than customer API keys
- demo-only response surface

#### 3. Add customer-oriented messaging

Content should communicate:

- what the service offers
- who it is for
- what is live now
- what clients can integrate against

Keep the product promise aligned with stage-2 reality.

#### 4. Add demo analytics

Track:

- searches per day
- top lookups
- not-found rate
- abuse rate
- conversion actions if relevant

Do not overbuild this before the service itself is ready.

### Stage-3 Acceptance Criteria

Stage 3 is done when:

- the public can search words without exposing customer credentials
- the demo is visually deliberate and usable on mobile and desktop
- the demo uses the same published data version as clients
- abuse can be controlled operationally

## Recommended Execution Order

Implement in this order:

1. Stage-1 schema additions and extract ledger
2. Stage-1 diff planner
3. Stage-1 enhancement reuse cache
4. Stage-1 export rebuild
5. Stage-1 safer D1 deployment
6. Stage-1 reporting and tests
7. Stage-2 auth hardening
8. Stage-2 rate limiting and API contracts
9. Stage-2 observability and rollout support
10. Stage-3 frontend and demo path

## Proposed Milestones

### Milestone 1

Goal: new extract can be ingested and diffed against the prior extract.

Exit criteria:

- extract identity recorded
- per-word and per-definition hashes recorded
- diff report generated

### Milestone 2

Goal: unchanged definitions reuse prior LLM work.

Exit criteria:

- no re-enhancement on unchanged rerun
- changed-only enhancement verified on fixture tests

### Milestone 3

Goal: publish current snapshot artifacts.

Exit criteria:

- export-current produces deployment DB
- pure-word DB and trie are rebuilt from current snapshot

### Milestone 4

Goal: D1 deployment is safe to use continuously.

Exit criteria:

- no destructive truncate-before-import path
- deployment is restartable or recoverable

### Milestone 5

Goal: first-party client usage is safe.

Exit criteria:

- auth hardened
- rate limiting active
- version metadata exposed

### Milestone 6

Goal: demo is presentable and safe.

Exit criteria:

- public search page live
- no customer secret exposure

## Risks and Mitigations

### Risk: overloading the existing state DB migration

Mitigation:

- add new tables first
- preserve current pipeline until the new path is proven
- migrate by addition, then cut over

### Risk: false-positive definition changes due to unstable normalization

Mitigation:

- keep normalization rules small and explicit
- lock them with fixture tests
- do not include volatile formatting in hashes

### Risk: D1 deployment complexity delays delivery

Mitigation:

- ship staged full-snapshot deployment before delta deployment
- treat SQL delta patches as a second optimization

### Risk: LLM prompt changes invalidate too much cached work

Mitigation:

- explicit enhancement versioning
- change the version only when behavior materially changes

### Risk: stage 3 distracts from stage 1 and 2

Mitigation:

- treat the demo as downstream of service readiness
- do not add more frontend surface until auth and deploy are stable

## Immediate Next Tasks

The next concrete implementation tasks should be:

1. Add the new extract/version tables to the state DB initialization code.
2. Implement hash helpers and normalization helpers.
3. Add `ingest-extract`.
4. Add `plan-update`.
5. Define the enhancement cache lookup and `ENHANCEMENT_VERSION`.
6. Replace the current build script with the new staged flow.
7. Redesign `deploy-db.sh` around staging tables or versioned tables.

## Definition of Done For The Overall Roadmap

The roadmap is complete when:

- a new extract can be processed incrementally with changed-only LLM work
- D1 and trie publishing are reliable and explainable
- first-party clients can use the API under stable auth and rate limits
- a public demo showcases the same underlying service without exposing customer credentials
