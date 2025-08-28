#!/usr/bin/env python3
"""
Resumable JSONL -> small dictionary builder.

Features:
- Reads a large JSONL of dictionary entries (use your snippet to test).
- Emits compact definitions per unique word, e.g.: "_adjective_ Of the color of the raven, jet-black."
- Limits each definition line to <= 25 words.
- Two-phase workflow:
  - baseline: stream the large JSONL once, store compact baseline defs in SQLite, mark LM status per word.
  - enhance: without scanning the large file, read pending words from SQLite, use LM to refine into <= 25 words, store enhanced defs, mark done.
- Optional export mode to write a final JSONL from the DB.
- Extract pure words: filter words that are single whole words (no spaces, hyphens, apostrophes, etc.)
- Generate pure words DB: create new database with only pure words and their definitions.

Example (baseline only):
  python scripts/make_small_dictionary.py \
    --input data/snippet.jsonl \
    --state out/state.sqlite3 \
    --max-defs 5 \
    --checkpoint-interval 100

Enhance with LM Studio summarization (OpenAI-compatible API):
  python scripts/make_small_dictionary.py \
    --input data/snippet.jsonl \
    --state out/state.sqlite3 \
    --mode enhance \
    --lmstudio-url http://localhost:1234/v1/chat/completions \
    --lmstudio-model qwen/qwen3-4b-2507 \
    --max-defs 5

Export consolidated JSONL from DB (prefers enhanced when present):
  python scripts/make_small_dictionary.py \
    --output out/small_dictionary.final.jsonl \
    --state out/state.sqlite3 \
    --mode export

Extract pure words list:
  python scripts/make_small_dictionary.py \
    --state out/state.sqlite3 \
    --mode extract-pure-words

Generate pure words database:
  python scripts/make_small_dictionary.py \
    --state out/state.sqlite3 \
    --mode generate-pure-db \
    --output out/pure_dictionary.sqlite3

Cleanup tags from definitions:
  python scripts/make_small_dictionary.py \
    --state out/state.sqlite3 \
    --mode cleanup-tags

Stop anytime with Ctrl-C. Re-run with the same --state and --output to resume.
"""

from __future__ import annotations

import argparse
import dataclasses
import io
import json
import os
import re
import signal
import sqlite3
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple, cast

try:
    # Use stdlib if requests isn't available.
    import urllib.request as _urllib
except Exception:  # pragma: no cover
    _urllib = None





def ensure_parent_dir(path: str) -> None:
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def connect_state(db_path: str) -> sqlite3.Connection:
    ensure_parent_dir(db_path)
    # Increase default busy timeout to reduce 'database is locked' errors
    conn = sqlite3.connect(db_path, timeout=60.0)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA busy_timeout=60000;")
    # Core tables only
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS words (
            word TEXT PRIMARY KEY,
            status TEXT NOT NULL CHECK (status IN ('pending','done')),
            updated_at INTEGER DEFAULT (strftime('%s','now'))
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS definitions (
            word TEXT NOT NULL,
            idx INTEGER NOT NULL,
            pos TEXT,
            source_first_sentence TEXT NOT NULL,
            current_line TEXT NOT NULL,
            enhanced_source TEXT,
            PRIMARY KEY (word, idx)
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS progress (
            id INTEGER PRIMARY KEY CHECK (id=1),
            line_offset INTEGER NOT NULL DEFAULT 0
        );
        """
    )
    # Lightweight, backwards-compatible migration: ensure a 'reprocess' column exists on words.
    # This avoids changing the CHECK constraint, and lets us target only words flagged for reprocessing.
    try:
        cur = conn.execute("PRAGMA table_info(words);")
        cols = [r[1] for r in cur.fetchall()]
        if "reprocess" not in cols:
            conn.execute("ALTER TABLE words ADD COLUMN reprocess INTEGER NOT NULL DEFAULT 0;")
        if "display_word" not in cols:
            conn.execute("ALTER TABLE words ADD COLUMN display_word TEXT;")
    except Exception:
        # If ALTER fails for any reason, continue without the column; callers must handle gracefully.
        pass
    
    # Migration for enhanced_source column in definitions table
    try:
        cur = conn.execute("PRAGMA table_info(definitions);")
        def_cols = [r[1] for r in cur.fetchall()]
        if "enhanced_source" not in def_cols:
            conn.execute("ALTER TABLE definitions ADD COLUMN enhanced_source TEXT;")
    except Exception:
        pass

    # Create search results table for storing online search data
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS search_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            word TEXT NOT NULL,
            definition_idx INTEGER NOT NULL,
            source TEXT NOT NULL,
            title TEXT NOT NULL,
            snippet TEXT NOT NULL,
            url TEXT,
            rank INTEGER DEFAULT 0,
            created_at INTEGER DEFAULT (strftime('%s','now')),
            FOREIGN KEY (word, definition_idx) REFERENCES definitions(word, idx) ON DELETE CASCADE
        );
        """
    )

    # Create search metadata table for tracking search operations
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS search_metadata (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            word TEXT NOT NULL,
            definition_idx INTEGER NOT NULL,
            search_term TEXT NOT NULL,
            search_engines TEXT NOT NULL,  -- JSON array of search engines used
            total_results INTEGER DEFAULT 0,
            helpful_results INTEGER DEFAULT 0,
            created_at INTEGER DEFAULT (strftime('%s','now')),
            FOREIGN KEY (word, definition_idx) REFERENCES definitions(word, idx) ON DELETE CASCADE
        );
        """
    )
    # Ensure a single row exists
    conn.execute("INSERT OR IGNORE INTO progress (id, line_offset) VALUES (1, 0);")
    conn.commit()
    return conn


def get_line_offset(conn: sqlite3.Connection) -> int:
    cur = conn.execute("SELECT line_offset FROM progress WHERE id=1;")
    row = cur.fetchone()
    return int(row[0]) if row else 0


def set_line_offset(conn: sqlite3.Connection, line_offset: int) -> None:
    conn.execute("UPDATE progress SET line_offset=? WHERE id=1;", (line_offset,))
    conn.commit()


def is_processed(conn: sqlite3.Connection, word: str) -> bool:
    # Consider word processed in baseline if it already exists in 'words'.
    cur = conn.execute("SELECT 1 FROM words WHERE word=?;", (word.lower(),))
    return cur.fetchone() is not None


def mark_processed(conn: sqlite3.Connection, word: str) -> None:
    # No-op retained for compatibility in baseline flow.
    return


def get_lm_status(conn: sqlite3.Connection, word: str) -> Optional[str]:
    cur = conn.execute("SELECT status FROM words WHERE word=?;", (word.lower(),))
    row = cur.fetchone()
    return row[0] if row else None


def set_lm_status(conn: sqlite3.Connection, word: str, status: str, *, display_word: Optional[str] = None) -> None:
    wkey = word.lower()
    d = display_word
    # Insert or update status; keep existing display_word if present, otherwise set from provided value
    conn.execute(
        "INSERT INTO words(word, status, updated_at, display_word) VALUES(?, ?, strftime('%s','now'), ?)\n"
        "ON CONFLICT(word) DO UPDATE SET status=excluded.status, updated_at=strftime('%s','now'),\n"
        " display_word=COALESCE(words.display_word, excluded.display_word)",
        (wkey, status, d),
    )


def _casing_rank(s: str) -> int:
    """Return preference rank for display casing: lower is better.

    0 = all lowercase (best)
    1 = name/mixed case (e.g., Titlecase, camel, mixed)
    2 = all uppercase (worst; acronym)
    """
    if not s:
        return 1
    if s.islower():
        return 0
    if s.isupper():
        return 2
    return 1


def maybe_upgrade_display_word(conn: sqlite3.Connection, word_key: str, candidate_display: Optional[str]) -> None:
    """Upgrade display_word if candidate has a better casing rank.

    word_key should be the lowercase canonical key. candidate_display is the new
    original-cased form encountered. If no existing display_word, set it. If the
    existing one has a worse rank (e.g., AMP) and candidate is better (Amp or amp),
    update it.
    """
    if not candidate_display:
        return
    wkey = word_key.lower()
    try:
        row = conn.execute("SELECT display_word FROM words WHERE word=?;", (wkey,)).fetchone()
        current = row[0] if row else None
        if not current:
            conn.execute(
                "UPDATE words SET display_word=?, updated_at=strftime('%s','now') WHERE word=?;",
                (candidate_display, wkey),
            )
            return
        if _casing_rank(candidate_display) < _casing_rank(current):
            conn.execute(
                "UPDATE words SET display_word=?, updated_at=strftime('%s','now') WHERE word=?;",
                (candidate_display, wkey),
            )
    except Exception:
        return


def set_reprocess_flag(conn: sqlite3.Connection, word: str, flag: bool) -> None:
    try:
        conn.execute(
            "UPDATE words SET reprocess=?, updated_at=strftime('%s','now') WHERE word=?;",
            (1 if flag else 0, word.lower()),
        )
    except Exception:
        # Column may not exist if migration failed; ignore silently.
        pass


def store_search_results(conn: sqlite3.Connection, word: str, definition_idx: int, search_term: str, search_engines: List[str], search_results: List[Dict[str, Any]]) -> None:
    """Store search results and metadata in the database."""
    try:
        # Store search metadata
        conn.execute(
            """
            INSERT INTO search_metadata (word, definition_idx, search_term, search_engines, total_results, helpful_results)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                word.lower(),
                definition_idx,
                search_term,
                json.dumps(search_engines),
                len(search_results),
                len([r for r in search_results if r.get("snippet", "").strip()])  # Count results with actual content
            )
        )

        # Store individual search results
        for result in search_results:
            conn.execute(
                """
                INSERT INTO search_results (word, definition_idx, source, title, snippet, url, rank)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    word.lower(),
                    definition_idx,
                    result.get("source", "unknown"),
                    result.get("title", ""),
                    result.get("snippet", ""),
                    result.get("url", ""),
                    result.get("rank", 0)
                )
            )

    except Exception as e:
        # Fail silently if search result storage fails
        pass


def get_stored_search_results(conn: sqlite3.Connection, word: str, definition_idx: int) -> List[Dict[str, Any]]:
    """Retrieve stored search results for a word and definition."""
    try:
        cur = conn.execute(
            """
            SELECT source, title, snippet, url, rank
            FROM search_results
            WHERE word=? AND definition_idx=?
            ORDER BY rank ASC
            """,
            (word.lower(), definition_idx)
        )
        return [
            {
                "source": row[0],
                "title": row[1],
                "snippet": row[2],
                "url": row[3],
                "rank": row[4]
            }
            for row in cur.fetchall()
        ]
    except Exception:
        return []


def has_recent_search_results(conn: sqlite3.Connection, word: str, definition_idx: int, max_age_hours: int = 24) -> bool:
    """Check if we have recent search results for a word and definition."""
    try:
        cutoff_time = int(time.time()) - (max_age_hours * 3600)
        cur = conn.execute(
            """
            SELECT COUNT(*) FROM search_metadata
            WHERE word=? AND definition_idx=? AND created_at > ?
            """,
            (word.lower(), definition_idx, cutoff_time)
        )
        return cur.fetchone()[0] > 0
    except Exception:
        return False


# -------------- Parsing helpers --------------

WORD_KEYS = (
    "word",
    "title",
    "headword",
    "lemma",
    "entry",
)

POS_KEYS = (
    "pos",
    "partOfSpeech",
    "part_of_speech",
    "part-of-speech",
)

GLOSS_KEYS = (
    "gloss",
    "definition",
    "def",
    "sense",
    "meaning",
    "desc",
    "text",
)

SENSES_KEYS = (
    "senses",
    "definitions",
    "meanings",
    "glosses",
    "entries",
)

LANG_KEYS = (
    "lang",
    "language",
    "lang_name",
)

LANG_CODE_KEYS = (
    "lang_code",
    "langCode",
    "language_code",
)


def _first_present(d: Dict[str, Any], keys: Iterable[str]) -> Optional[Any]:
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return None


def extract_word(obj: Dict[str, Any]) -> Optional[str]:
    w = _first_present(obj, WORD_KEYS)
    if isinstance(w, str):
        w = w.strip()
        # Ignore entries that look like phrases if desired? Keep as-is for now.
        return w if w else None
    return None


def extract_language(obj: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    lang = _first_present(obj, LANG_KEYS)
    code = _first_present(obj, LANG_CODE_KEYS)
    if isinstance(lang, str):
        lang = lang.strip()
    if isinstance(code, str):
        code = code.strip()
    return (lang if isinstance(lang, str) and lang else None,
            code if isinstance(code, str) and code else None)


def normalize_pos(pos: Optional[str]) -> Optional[str]:
    if not pos:
        return None
    p = str(pos).strip().lower()
    # Normalize common variants
    mapping = {
        "n": "noun",
        "v": "verb",
        "adj": "adjective",
        "adv": "adverb",
        "prep": "preposition",
        "pron": "pronoun",
        "det": "determiner",
        "conj": "conjunction",
        "interj": "interjection",
        "part": "particle",
        "num": "numeral",
    }
    return mapping.get(p, p)


def _split_sentences(text: str) -> List[str]:
    # Very light sentence splitter.
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s for s in parts if s]


def _word_count(text: str) -> int:
    return len(re.findall(r"\b\W+\b", text))


def _limit_words(text: str, max_words: int) -> str:
    words = re.findall(r"\b\w+\b", text)
    if len(words) <= max_words:
        return text.strip()
    # Rebuild truncated string preserving original non-word chars roughly
    # Simple approach: join by space and add period if missing.
    truncated = " ".join(words[:max_words]).strip()
    if not re.search(r"[.!?]$", truncated):
        truncated += "."
    return truncated


def strip_llm_artifacts(text: str) -> str:
    """Remove LLM meta tags and any stray HTML/XML tags from text."""
    s = text or ""
    # Remove hidden reasoning blocks entirely (content and tags)
    s = re.sub(r"(?is)<\s*(think|reasoning|reflection|chain[-_ ]?of[-_ ]?thought)[^>]*>.*?<\s*/\s*\1\s*>", " ", s)
    # Remove markdown code blocks (```json ... ```)
    s = re.sub(r"(?s)```\s*json\s*(.*?)\s*```", r"\1", s)
    s = re.sub(r"(?s)```\s*(.*?)\s*```", r"\1", s)
    # Strip common inline tags (keep inner text already remaining)
    s = re.sub(r"(?is)</?\s*(br|b|i|u|em|strong|p|span|div|code|pre|blockquote|ul|ol|li|sup|sub)\b[^>]*>", " ", s)
    # Fallback: remove any remaining tags
    s = re.sub(r"(?s)<[^>]+>", " ", s)
    # Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


def clean_gloss(gloss: str) -> str:
    # Remove example brackets or wikilinks like [[word]]
    g = re.sub(r"\[\[(.*?)\]\]", r"\1", gloss)
    # Collapse whitespace
    g = re.sub(r"\s+", " ", g).strip()
    return g




def _stem(tok: str) -> str:
    """Very light stemmer to improve token overlap on paraphrases.

    Handles simple English variants: plurals, -ing/-ed verbs, -ly/-ally adverbs.
    Avoids heavy dependencies and keeps behavior conservative.
    """
    t = tok
    if len(t) > 4 and t.endswith("ally"):
        # pneumatically -> pneumatic
        return t[:-4]
    if len(t) > 4 and t.endswith("ly"):
        # quickly -> quick
        return t[:-2]
    if len(t) > 5 and t.endswith("ing"):
        # carving -> carv(e)
        return t[:-3]
    if len(t) > 4 and t.endswith("ed"):
        # pressurized -> pressuriz(e)
        return t[:-2]
    if len(t) > 3 and t.endswith("ies"):
        # candies -> candy
        return t[:-3] + "y"
    if len(t) > 3 and t.endswith("es"):
        # boxes -> box; cases -> cas(e)
        return t[:-2]
    if len(t) > 3 and t.endswith("s"):
        # rocks -> rock
        return t[:-1]
    return t


def _tokens(text: str) -> List[str]:
    # Normalize by removing punctuation and hyphens so source/candidate tokenization aligns.
    s = re.sub(r"[^\w]+", " ", text.lower())
    raw = re.findall(r"\w+", s)
    return [_stem(t) for t in raw]


def sanity_check_line_with_reason(source_sentence: str, candidate_line: str, max_words: int, expected_pos: Optional[str]) -> Tuple[bool, str]:
    # Reject if any residual tags remain
    if re.search(r"<[^>]+>", candidate_line or "", re.S):
        return False, "tags"
    # Check word count directly on the candidate line
    # Allow lines that are only 2-3 words over the limit
    if _word_count(candidate_line) > max_words + 3:
        return False, "too_long"
    # Remove POS prefix checking since we no longer mandate it
    cand_tokens = set(_tokens(candidate_line))
    if not cand_tokens:
        return False, "empty"
    if 'http://' in candidate_line or 'https://' in candidate_line:
        return False, "url"
    # Allow quotes in dictionary definitions - they're commonly used for proper names, examples, etc.
    # if '"' in candidate_line:
    #     return False, "quotes"
    return True, "ok"


def sanity_check_line(source_sentence: str, candidate_line: str, max_words: int, expected_pos: Optional[str]) -> bool:
    ok, _ = sanity_check_line_with_reason(source_sentence, candidate_line, max_words, expected_pos)
    return ok


def format_def_line(pos: Optional[str], gloss: str, max_words: int) -> Optional[str]:
    g = clean_gloss(gloss)
    if not g:
        return None
    # Prefer first sentence; fall back to whole gloss
    first = _split_sentences(g)[0] if _split_sentences(g) else g
    first = _limit_words(first, max_words)
    # Remove POS prefix requirement - just return the formatted definition
    return first


def first_sentence_word_count(gloss: str) -> int:
    g = clean_gloss(gloss)
    parts = _split_sentences(g)
    s = parts[0] if parts else g
    return _word_count(s)


def extract_definitions(obj: Dict[str, Any]) -> List[Tuple[Optional[str], str]]:
    """Return list of (pos, gloss) pairs from a heterogenous entry object."""
    out: List[Tuple[Optional[str], str]] = []

    # Shape: entries: { pos: [ {definition: ...}, ... ] }
    entries = obj.get("entries")
    if isinstance(entries, dict):
        for pos_key, items in entries.items():
            if not isinstance(items, list):
                continue
            for it in items:
                if isinstance(it, dict):
                    g = _first_present(it, GLOSS_KEYS)
                    if isinstance(g, str):
                        out.append((pos_key, g))
                elif isinstance(it, str):
                    out.append((pos_key, it))

    # If there is a flat 'definitions' list of strings
    flat = _first_present(obj, ("definitions", "glosses"))
    if isinstance(flat, list) and flat and all(isinstance(x, str) for x in flat):
        pos = _first_present(obj, POS_KEYS)
        for g in flat:
            out.append((pos, str(g)))

    # Common shape: senses: [{pos, gloss}, ...] or senses: [{gloss}, ...] with pos on parent
    senses = _first_present(obj, SENSES_KEYS)
    if isinstance(senses, list):
        parent_pos = _first_present(obj, POS_KEYS)
        for s in senses:
            if isinstance(s, dict):
                s_pos = _first_present(s, POS_KEYS) or parent_pos
                g = _first_present(s, GLOSS_KEYS)
                if isinstance(g, str):
                    out.append((s_pos, g))
                # Sometimes nested structures e.g., {"glosses": ["..."]}
                g_list = _first_present(s, ("glosses", "definitions"))
                if isinstance(g_list, list):
                    for gg in g_list:
                        if isinstance(gg, str):
                            out.append((s_pos, gg))

    # Fallback: try top-level gloss-like fields
    for k in GLOSS_KEYS:
        g = obj.get(k)
        if isinstance(g, str):
            out.append((_first_present(obj, POS_KEYS), g))

    # Deduplicate identical (pos, gloss)
    dedup = list(dict.fromkeys(out))
    return dedup


# -------------- LM Studio client (optional) --------------


@dataclasses.dataclass
class LMStudioConfig:
    url: str
    model: str
    timeout: float = 15.0


def _search_web_for_context(term: str, pos: Optional[str], search_engines: Optional[List[str]] = None) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    """Search for additional context about a term using multiple web sources.

    Returns (context_string, search_results_list) where search_results_list contains
    detailed information about each search result for storage in database.
    """
    if search_engines is None:
        search_engines = ["duckduckgo", "wikipedia"]

    if _urllib is None:
        return None, []

    context_parts = []
    search_results = []

    try:
        import urllib.parse

        # Construct search query
        query = f"{term} definition meaning"
        if pos:
            query += f" {pos}"

        # 1. DuckDuckGo Instant Answer API (no API key required)
        if "duckduckgo" in search_engines:
            try:
                encoded_query = urllib.parse.quote(query)
                url = f"https://api.duckduckgo.com/?q={encoded_query}&format=json&no_redirect=1"

                req = _urllib.Request(url, headers={"User-Agent": "Dictionary-Builder/1.0"})
                with _urllib.urlopen(req, timeout=10.0) as resp:
                    raw = resp.read().decode("utf-8")
                    data = json.loads(raw)

                    # Abstract (main definition)
                    if data.get("Abstract"):
                        context_parts.append(data["Abstract"])
                        search_results.append({
                            "source": "duckduckgo_abstract",
                            "title": data.get("Heading", term),
                            "snippet": data["Abstract"],
                            "url": data.get("AbstractURL", ""),
                            "rank": 1
                        })

                    # Related topics
                    if data.get("RelatedTopics"):
                        for i, topic in enumerate(data["RelatedTopics"][:3]):  # Limit to first 3
                            if isinstance(topic, dict) and topic.get("Text"):
                                context_parts.append(topic["Text"])
                                search_results.append({
                                    "source": "duckduckgo_related",
                                    "title": topic.get("FirstURL", "").split('/')[-1] if topic.get("FirstURL") else f"Related Topic {i+1}",
                                    "snippet": topic["Text"],
                                    "url": topic.get("FirstURL", ""),
                                    "rank": i + 2
                                })

                    # Answer (if available)
                    if data.get("Answer"):
                        context_parts.append(data["Answer"])
                        search_results.append({
                            "source": "duckduckgo_answer",
                            "title": f"{term} - Answer",
                            "snippet": data["Answer"],
                            "url": data.get("AnswerURL", ""),
                            "rank": 1
                        })

            except Exception as e:
                # Continue with other search engines if DuckDuckGo fails
                pass

        # 2. Wikipedia search
        if "wikipedia" in search_engines:
            try:
                # Search Wikipedia API
                search_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(term)}"

                req = _urllib.Request(search_url, headers={"User-Agent": "Dictionary-Builder/1.0"})
                with _urllib.urlopen(req, timeout=10.0) as resp:
                    raw = resp.read().decode("utf-8")
                    data = json.loads(raw)

                    if data.get("extract"):
                        # Get first few sentences from Wikipedia extract
                        extract = data["extract"]
                        # Limit to reasonable length
                        parts = re.split(r"(?<=[.!?])\s+", extract.strip())
                        wiki_snippet = " ".join(parts[:2]) if parts else extract
                        context_parts.append(wiki_snippet)
                        search_results.append({
                            "source": "wikipedia",
                            "title": data.get("title", term),
                            "snippet": wiki_snippet,
                            "url": data.get("content_urls", {}).get("desktop", {}).get("page", ""),
                            "rank": 1
                        })

            except Exception as e:
                # Continue if Wikipedia fails
                pass

        # 3. Google Custom Search API (if API key available)
        if "google" in search_engines:
            try:
                # This would require a Google Custom Search API key
                # For now, we'll skip this as it requires API keys
                pass
            except Exception:
                pass

        if context_parts:
            return " | ".join(context_parts), search_results

    except Exception:
        pass  # Fail silently and continue without web context

    return None, []


def _evaluate_definition_quality(gloss: str, cfg: LMStudioConfig) -> bool:
    """Ask LLM to evaluate if a definition is helpful or just describes notation/formula."""
    if _urllib is None or not gloss.strip():
        return True  # Assume it's fine if we can't check
    
    evaluation_prompt = (
        "Evaluate if this dictionary definition is helpful for understanding the concept, "
        "or if it merely describes chemical formulas, mathematical notation, or technical symbols "
        "without explaining what the concept actually means. "
        "Respond with only 'helpful' or 'unhelpful'."
    )
    
    payload = {
        "model": cfg.model,
        "messages": [
            {"role": "system", "content": evaluation_prompt},
            {"role": "user", "content": f"Definition: {gloss}"},
        ],
        "temperature": 0.0,
        "max_tokens": 10,
    }
    
    try:
        data = json.dumps(payload).encode("utf-8")
        req = _urllib.Request(cfg.url, data=data, headers={"Content-Type": "application/json"})
        with _urllib.urlopen(req, timeout=cfg.timeout) as resp:
            raw = resp.read().decode("utf-8")
            obj = json.loads(raw)
            choices = obj.get("choices")
            if isinstance(choices, list) and choices:
                msg = choices[0].get("message") or {}
                text = msg.get("content")
                if isinstance(text, str):
                    cleaned = strip_llm_artifacts(text.strip().lower())
                    return cleaned != "unhelpful"
    except Exception:
        pass
    
    return True  # Default to assuming it's helpful if evaluation fails


def summarize_with_lmstudio(
    cfg: LMStudioConfig,
    pos: Optional[str],
    gloss: str,
    max_words: int,
    *,
    temperature: Optional[float] = None,
    web_context_override: Optional[str] = None,
    search_results_override: Optional[List[Dict[str, Any]]] = None,
) -> Optional[Tuple[str, Optional[str], List[Dict[str, Any]]]]:
    if _urllib is None:
        return None

    # First, evaluate if the definition is helpful
    is_helpful = _evaluate_definition_quality(gloss, cfg)

    # Allow callers to provide a pre-fetched web context / search results. If not provided,
    # perform an on-demand search only when the evaluator deems the gloss unhelpful.
    web_context = web_context_override
    search_results: List[Dict[str, Any]] = search_results_override or []
    if not is_helpful and web_context is None and not search_results:
        # Extract potential term from gloss for web search
        term_match = re.search(r'^([A-Za-z][A-Za-z0-9\s-]+)', gloss)
        if term_match:
            search_term = term_match.group(1).strip()
            # Inform the user we're about to perform an external web search
            try:
                sys.stderr.write(f"[info] Performing web search for '{search_term}' (duckduckgo,wikipedia)...\n")
                sys.stderr.flush()
            except Exception:
                pass
            web_context, search_results = _search_web_for_context(search_term, pos)

    # Check if gloss contains chemical/technical notation
    technical_pattern = r'[0-9]{2,}|\([^)]+\)|[a-zA-Z][0-9]|[+\-=]|[∑∫παβγδεζηθικλμνξοπρστυφχψω]|[⁰¹²³⁴⁵⁶⁷⁸⁹]|[₀₁₂₃₄₅₆₇₈₉]|[→←↔⇒⇐⇔]|[∀∃∈∉⊂⊆⊃⊇]|[≠≈≡≪≫]|[∞∂∇∫∮∬∭∮∯∰∱∲∳]|[∧∨¬⇒⇔]|[∪∩∈∋⊆⊇⊂⊃⊄⊅⊈⊉]|[≤≥≮≯≰≱]|[∑∏∐∏]|[√∛∜]|[°′″‴⁗]]' 
    technical_indicators = len(re.findall(technical_pattern, gloss))

    if technical_indicators >= 3 or web_context:  # Source has significant technical notation or we found web context
        if web_context:
            prompt = (
                "Rewrite the following dictionary gloss into a single compact line, at most "
                f"{max_words} words. Use the additional context provided to create a more helpful definition. "
                "For chemical compounds, mathematical concepts, or technical terms with complex notation, "
                "provide a clear, descriptive explanation rather than copying the formula or symbols. "
                "Focus on what the concept is, its meaning, and its key properties. "
                "Keep it plain and accessible for general readers."
            )
            content = f"Gloss: {gloss}\nAdditional context: {web_context}\nOutput only the rewritten gloss, no quotes."
        else:
            prompt = (
                "Rewrite the following dictionary gloss into a single compact line, at most "
                f"{max_words} words. For chemical compounds, mathematical concepts, or technical terms with complex notation, "
                "provide a clear, descriptive explanation rather than copying the formula or symbols. "
                "Focus on what the concept is, its meaning, and its key properties, not the specific technical notation. "
                "Keep it plain and accessible for general readers who may not understand advanced mathematical or chemical notation."
            )
            content = f"Gloss: {gloss}\nOutput only the rewritten gloss, no quotes."
    else:
        prompt = (
            "Rewrite the following dictionary gloss into a single compact line, at most "
            f"{max_words} words, plain and clear. Do not include examples."
        )
        content = f"Gloss: {gloss}\nOutput only the rewritten gloss, no quotes."

    payload = {
        "model": cfg.model,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": content},
        ],
        "temperature": 0.2 if (temperature is None) else float(temperature),
        "max_tokens": 80,
    }
    data = json.dumps(payload).encode("utf-8")

    # Inform the user we're about to call the LLM
    try:
        sys.stderr.write(f"[info] Calling LLM at {cfg.url} (model={cfg.model})...\n")
        sys.stderr.flush()
    except Exception:
        pass
    req = _urllib.Request(cfg.url, data=data, headers={"Content-Type": "application/json"})
    try:
        with _urllib.urlopen(req, timeout=cfg.timeout) as resp:  # type: ignore[attr-defined]
            raw = resp.read().decode("utf-8")
            obj = json.loads(raw)
            # OpenAI-style response
            choices = obj.get("choices")
            if isinstance(choices, list) and choices:
                msg = choices[0].get("message") or {}
                text = msg.get("content")
                if isinstance(text, str):
                    # Clean artifacts; do NOT force-truncate here. Let sanity check enforce max_words.
                    cleaned = strip_llm_artifacts(text.strip().strip('"'))
                    if not cleaned:
                        return None
                    try:
                        sys.stderr.write("[info] Received LLM response.\n")
                        sys.stderr.flush()
                    except Exception:
                        pass
                    return (cleaned.strip(), web_context if web_context else None, search_results)
    except Exception:
        return None
    return None


def format_eta(elapsed: float, progress: float) -> str:
    """Format ETA based on elapsed time and progress (0.0 to 1.0)."""
    if progress <= 0:
        return "--:--"

    remaining = elapsed * (1.0 - progress) / progress
    if remaining < 60:
        return f"{int(remaining)}s"
    elif remaining < 3600:
        minutes = int(remaining // 60)
        seconds = int(remaining % 60)
        return f"{minutes}m {seconds}s"
    else:
        hours = int(remaining // 3600)
        minutes = int((remaining % 3600) // 60)
        return f"{hours}h {minutes}m"

def verify_with_lmstudio(cfg: LMStudioConfig, *,
                         source_sentence: str,
                         candidate_line: str,
                         expected_pos: Optional[str],
                         max_words: int,
                         temperature: Optional[float] = None) -> Tuple[bool, str]:
    """Ask an LM to verify candidate_line is a faithful, concise paraphrase.

    Returns (valid, reason). Reason is a short machine-friendly string.
    """
    if _urllib is None:
        return False, "no_urllib"
    # Maximum token efficiency - single tokens for valid, JSON only for invalid
    if "deepseek" in cfg.model.lower():
        # DeepSeek needs very explicit instructions to avoid reasoning
        sys_prompt = (
            "Task: Check if candidate restates source faithfully."
            " Rules: <= {max_words} words, POS prefix matches, no examples/URLs."
            " Output: 'valid' if valid, or {{\"valid\": false, \"reason\": \"brief reason\"}} if invalid."
            " No explanation, no thinking, no markdown.".format(max_words=max_words)
        )
    else:
        # Other models work fine with simpler prompt
        sys_prompt = (
            "Verify if the candidate is a faithful, concise restatement of the source."
            " Return: 'valid' if valid, or {{\"valid\": false, \"reason\": \"brief reason\"}} if invalid."
            " Check: <= {max_words} words, no examples/URLs.".format(max_words=max_words)
        )
    # Use a compact, JSON-only response schema.
    user = {
        "source": source_sentence,
        "candidate": candidate_line,
        "expected_pos": normalize_pos(expected_pos) if expected_pos else None,
        "rules": {
            "max_words": max_words,
            "require_pos_match": bool(expected_pos),
        },
        "respond_with": {"valid": "boolean", "reason": "string"},
    }
    payload = {
        "model": cfg.model,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
        ],
        "temperature": 0.0 if (temperature is None) else float(temperature),
    }
    # No max_tokens limit for any model - let them complete naturally
    data = json.dumps(payload).encode("utf-8")
    req = _urllib.Request(cfg.url, data=data, headers={"Content-Type": "application/json"})
    try:
        with _urllib.urlopen(req, timeout=cfg.timeout) as resp:  # type: ignore[attr-defined]
            raw = resp.read().decode("utf-8")
            obj = json.loads(raw)
            choices = obj.get("choices")
            if isinstance(choices, list) and choices:
                msg = choices[0].get("message") or {}
                text = msg.get("content")
                if isinstance(text, str):
                    # Clean LLM artifacts first
                    text = strip_llm_artifacts(text.strip())

                    # Check for single "valid" token (maximum efficiency)
                    if text.strip().lower() == "valid":
                        return True, "ok"

                    # Otherwise, parse as JSON for invalid cases
                    try:
                        parsed = json.loads(text)
                    except Exception:
                        # Fallback: look for JSON at the end of the response (DeepSeek pattern)
                        lines = text.strip().split('\n')
                        for line in reversed(lines[-10:]):  # Check last 10 lines
                            line = line.strip()
                            if line.startswith('{') and line.endswith('}'):
                                try:
                                    parsed = json.loads(line)
                                    break
                                except Exception:
                                    continue
                        else:
                            # Original fallback: find any JSON object
                            m = re.search(r"\{.*\}", text, re.S)
                            if m:
                                try:
                                    parsed = json.loads(m.group(0))
                                except Exception:
                                    return False, "parse_error"
                            else:
                                return False, "parse_error"
                    valid = bool(parsed.get("valid"))
                    reason = str(parsed.get("reason") or ("ok" if valid else "invalid"))
                    return valid, reason
    except Exception:
        return False, "request_error"
    return False, "no_choice"


def warm_up_model(cfg: LMStudioConfig, *, temperature: Optional[float] = None, max_wait_seconds: float = 180.0) -> Tuple[bool, float, Optional[str]]:
    """Trigger a minimal completion to auto-load the target model and wait until it's ready.

    Returns (ok, elapsed_seconds, reported_model).
    """
    if _urllib is None:
        return False, 0.0, None

    print(f'Warming-up {cfg.model}')

    import time as _time
    start = _time.time()
    prompt = "Respond with the single token: OK"
    content = "OK"
    payload = {
        "model": cfg.model,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": content},
        ],
        "temperature": (0.0 if temperature is None else float(temperature)),
        "max_tokens": 2,
    }
    reported_model: Optional[str] = None
    while True:
        data = json.dumps(payload).encode("utf-8")
        req = _urllib.Request(cfg.url, data=data, headers={"Content-Type": "application/json"})
        try:
            with _urllib.urlopen(req, timeout=cfg.timeout) as resp:  # type: ignore[attr-defined]
                raw = resp.read().decode("utf-8")
                obj = json.loads(raw)
                reported_model = obj.get("model") if isinstance(obj, dict) else None
                choices = obj.get("choices") if isinstance(obj, dict) else None
                text = None
                if isinstance(choices, list) and choices:
                    msg = choices[0].get("message") or {}
                    text = msg.get("content")
                ok = isinstance(text, str) and text.strip().upper().startswith("OK")
                elapsed = _time.time() - start
                return ok, elapsed, reported_model
        except Exception:
            # Keep waiting up to max_wait_seconds
            if (_time.time() - start) >= max_wait_seconds:
                return False, (_time.time() - start), reported_model
            # Brief backoff before retry
            _time.sleep(1.0)


# -------------- Main processing --------------


def process_baseline_mode(
    conn: sqlite3.Connection,
    input_path: str,
    max_defs: int,
    max_words_per_def: int,
    checkpoint_interval: int,
    line_offset: int,
) -> None:
    """Process dictionary entries in baseline mode."""
    # Open and stream input; store results in DB only.
    in_f = open(input_path, "r", encoding="utf-8", errors="replace")
    # Skip already processed lines
    for _ in range(line_offset):
        if not in_f.readline():
            break

    processed_since_checkpoint = 0
    total_read = line_offset

    sys.stderr.write(f"Resuming at line {line_offset}\n")
    sys.stderr.flush()

    # Progress counters
    last_progress_time = time.time()
    processed_words_total = 0
    duplicates_skipped = 0

    # Global deduplication cache - track all seen definitions across all words
    global_seen_definitions = set()

    # Surname deduplication - track surname types to avoid duplicates
    seen_surname_types = set()

    while True:
        line = in_f.readline()
        if not line:
            break
        total_read += 1
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        # Language gating: if entry specifies language and it is not English, skip
        lang, code = extract_language(obj)
        if lang or code:
            is_en = False
            if isinstance(lang, str) and lang.lower() == "english":
                is_en = True
            if isinstance(code, str) and code.lower() in ("en", "eng"):
                is_en = True
            if not is_en:
                continue
        word = extract_word(obj)
        if not word:
            continue
        if is_processed(conn, word):
            continue
        defs = extract_definitions(obj)
        if not defs:
            continue

        # Build and store baseline defs
        formatted: List[Tuple[int, Optional[str], str, str]] = []
        seen_lines = set()
        lm_needed = False
        def_index = 0
        for pos_tag, gloss in defs:
            base_line = format_def_line(pos_tag, gloss, max_words_per_def) or ""
            if first_sentence_word_count(gloss) > max_words_per_def:
                lm_needed = True
            base_line = base_line.strip()
            if not base_line:
                continue
            if base_line in seen_lines:
                continue
            # Surname deduplication: avoid storing duplicate surname definitions
            if "surname" in base_line.lower():
                surname_type = base_line.lower().strip()
                if surname_type in seen_surname_types:
                    duplicates_skipped += 1
                    continue
                seen_surname_types.add(surname_type)

            # Global deduplication: skip if this definition already exists anywhere
            if base_line in global_seen_definitions:
                duplicates_skipped += 1
                continue
            seen_lines.add(base_line)
            source_sentence = _split_sentences(clean_gloss(gloss))
            source_sentence = source_sentence[0] if source_sentence else clean_gloss(gloss)
            formatted.append((def_index, pos_tag, source_sentence, base_line))
            def_index += 1
            if len(formatted) >= max_defs:
                break

        if not formatted:
            continue

        # Persist word and defs (current_line initially baseline)
        set_lm_status(conn, word, "pending" if lm_needed else "done", display_word=word)
        # Upgrade display casing if a better form is seen later (amp > Amp > AMP)
        maybe_upgrade_display_word(conn, word, word)
        for idx, pos_tag, source_sentence, base_line in formatted:
            conn.execute(
                "INSERT OR REPLACE INTO definitions(word, idx, pos, source_first_sentence, current_line) VALUES(?,?,?,?,?)",
                (word.lower(), idx, normalize_pos(pos_tag) if pos_tag else None, source_sentence, base_line),
            )
            # Add to global deduplication cache after successful storage
            global_seen_definitions.add(base_line)
        mark_processed(conn, word)
        processed_since_checkpoint += 1
        processed_words_total += 1

        if processed_since_checkpoint >= checkpoint_interval:
            set_line_offset(conn, total_read)
            conn.commit()
            processed_since_checkpoint = 0
        # Periodic progress output once per second, overwrite same line
        try:
            now = time.time()
            if now - last_progress_time >= 1.0:
                curp = conn.execute("SELECT COUNT(*) FROM words WHERE status='pending';")
                pending = curp.fetchone()[0]
                curd = conn.execute("SELECT COUNT(*) FROM words WHERE status='done';")
                done = curd.fetchone()[0]
                sys.stderr.write("\r" + f"Baseline: line={total_read} words={done+pending} done={done} pending={pending} dups={duplicates_skipped}")
                sys.stderr.flush()
                last_progress_time = now
        except Exception:
            pass

    # Final checkpoint for baseline
    set_line_offset(conn, total_read)
    conn.commit()
    in_f.close()
    try:
        sys.stderr.write("\n")
        sys.stderr.write(f"Baseline complete: {duplicates_skipped} duplicates skipped during import\n")
        sys.stderr.flush()
    except Exception:
        pass


def process_enhance_mode(
    conn: sqlite3.Connection,
    use_lmstudio: LMStudioConfig,
    max_words_per_def: int,
    checkpoint_interval: int,
    only_reprocess: bool,
    temperature: Optional[float],
) -> None:
    """Process dictionary entries in enhance mode using LLM."""
    if not use_lmstudio:
        sys.stderr.write("Enhance mode requires LM Studio configuration.\n")
        return

    processed_since_checkpoint = 0
    last_progress_time = time.time()
    processed_words_total = 0

    if only_reprocess:
        # Snapshot once and process each exactly once
        cur = conn.execute(
            "SELECT word FROM words WHERE status='pending' AND reprocess=1 ORDER BY word ASC;"
        )
        words = [r[0] for r in cur.fetchall()]
        attempted = 0
        succeeded = 0
        failed = 0

        for w in words:
            enhanced_any = process_single_word_enhancement(
                conn, w, use_lmstudio, max_words_per_def, temperature
            )
            attempted += 1
            if enhanced_any:
                set_lm_status(conn, w, "done")
                set_reprocess_flag(conn, w, False)
                succeeded += 1
            else:
                set_lm_status(conn, w, "pending")
                set_reprocess_flag(conn, w, True)
                failed += 1

            processed_since_checkpoint += 1
            processed_words_total += 1

            if processed_since_checkpoint >= checkpoint_interval:
                conn.commit()
                processed_since_checkpoint = 0

            # Progress update
            try:
                now = time.time()
                if now - last_progress_time >= 1.0:
                    sys.stderr.write("\r" + f"Enhance (reprocess): attempted={attempted} succeeded={succeeded} failed={failed}")
                    sys.stderr.flush()
                    last_progress_time = now
            except Exception:
                pass

        conn.commit()
        try:
            sys.stderr.write("\n")
            sys.stderr.flush()
        except Exception:
            pass
        try:
            sys.stderr.write(
                f"Enhanced summary (reprocess-only): attempted={attempted} succeeded={succeeded} failed={failed}\n"
            )
            sys.stderr.flush()
        except Exception:
            pass
    else:
        # Process all pending words
        while True:
            cur = conn.execute(
                "SELECT word FROM words WHERE status='pending' LIMIT ?;",
                (checkpoint_interval,),
            )
            batch = [r[0] for r in cur.fetchall()]
            if not batch:
                break

            for w in batch:
                enhanced_any = process_single_word_enhancement(
                    conn, w, use_lmstudio, max_words_per_def, temperature
                )
                set_lm_status(conn, w, "done")
                set_reprocess_flag(conn, w, False)
                processed_since_checkpoint += 1
                processed_words_total += 1

                if processed_since_checkpoint >= checkpoint_interval:
                    conn.commit()
                    processed_since_checkpoint = 0

                try:
                    now = time.time()
                    if now - last_progress_time >= 1.0:
                        curp = conn.execute("SELECT COUNT(*) FROM words WHERE status='pending';")
                        pending = curp.fetchone()[0]
                        curd = conn.execute("SELECT COUNT(*) FROM words WHERE status='done';")
                        done = curd.fetchone()[0]
                        sys.stderr.write("\r" + f"Enhanced: processed={done} pending={pending}")
                        sys.stderr.flush()
                        last_progress_time = now
                except Exception:
                    pass

        conn.commit()
        try:
            sys.stderr.write("\n")
            sys.stderr.flush()
        except Exception:
            pass


def process_single_word_enhancement(
    conn: sqlite3.Connection,
    word: str,
    use_lmstudio: LMStudioConfig,
    max_words_per_def: int,
    temperature: Optional[float],
) -> bool:
    """Enhance a single word's definitions using LLM."""
    curd = conn.execute(
        "SELECT idx, pos, source_first_sentence, current_line FROM definitions WHERE word=? ORDER BY idx ASC;",
        (word,),
    )
    rows = curd.fetchall()
    enhanced_any = False

    for idx, pos_tag, source_sentence, current_line in rows:
        if _word_count(source_sentence) > max_words_per_def:
            # Check if we have recent search results
            if not has_recent_search_results(conn, word, idx):
                term_match_local = re.search(r'^([A-Za-z][A-Za-z0-9\s-]+)', source_sentence)
                search_term_local = term_match_local.group(1).strip() if term_match_local else word
                is_helpful_local = _evaluate_definition_quality(source_sentence, use_lmstudio)
                web_context_local = None
                search_results_local: List[Dict[str, Any]] = []
                if not is_helpful_local:
                    web_context_local, search_results_local = _search_web_for_context(search_term_local, pos_tag)
                    if search_results_local:
                        store_search_results(conn, word, idx, search_term_local, ["duckduckgo", "wikipedia"], search_results_local)

                result = summarize_with_lmstudio(
                    use_lmstudio,
                    pos_tag,
                    source_sentence,
                    max_words_per_def,
                    temperature=temperature,
                    web_context_override=web_context_local,
                    search_results_override=search_results_local,
                )
                new_line = None
                enhanced_source = None
                search_results = []

                if result:
                    new_line, enhanced_source, search_results = result
                    if new_line:
                        new_line = strip_llm_artifacts(new_line)

                # Store the enhanced result
                result_to_store = new_line if new_line else source_sentence
                conn.execute(
                    "UPDATE definitions SET current_line=?, enhanced_source=? WHERE word=? AND idx=?",
                    (result_to_store, enhanced_source, word, idx),
                )

                # Store search results if we have any
                if search_results:
                    term_match = re.search(r'^([A-Za-z][A-Za0-9\s-]+)', source_sentence)
                    search_term = term_match.group(1).strip() if term_match else word
                    store_search_results(conn, word, idx, search_term, ["duckduckgo", "wikipedia"], search_results)

                # Mark as enhanced if the result passed sanity checks
                if new_line and sanity_check_line(source_sentence, new_line, max_words_per_def, pos_tag):
                    enhanced_any = True
            else:
                # Use existing search results to enhance
                stored_results = get_stored_search_results(conn, word, idx)
                if stored_results:
                    # Combine stored search results into context
                    context_parts = [r["snippet"] for r in stored_results if r.get("snippet", "").strip()]
                    web_context = " | ".join(context_parts) if context_parts else None

                    if web_context:
                        # Re-run LLM with stored context
                        result = summarize_with_lmstudio(
                            use_lmstudio,
                            pos_tag,
                            source_sentence,
                            max_words_per_def,
                            temperature=temperature,
                            web_context_override=web_context,
                            search_results_override=stored_results,
                        )
                        new_line = None
                        enhanced_source = None
                        search_results = []

                        if result:
                            new_line, enhanced_source, search_results = result
                            if new_line:
                                new_line = strip_llm_artifacts(new_line)

                        result_to_store = new_line if new_line else source_sentence
                        conn.execute(
                            "UPDATE definitions SET current_line=?, enhanced_source=? WHERE word=? AND idx=?",
                            (result_to_store, enhanced_source, word, idx),
                        )

                        if new_line and sanity_check_line(source_sentence, new_line, max_words_per_def, pos_tag):
                            enhanced_any = True
        else:
            # Source is <= max_words_per_def, use as-is
            conn.execute(
                "UPDATE definitions SET current_line=?, enhanced_source=? WHERE word=? AND idx=?",
                (source_sentence, None, word, idx),
            )
            enhanced_any = True

    return enhanced_any


def process_file(
    input_path: str,
    state_path: str,
    max_defs: int,
    max_words_per_def: int,
    checkpoint_interval: int,
    use_lmstudio: Optional[LMStudioConfig] = None,
    mode: str = "baseline",
    only_reprocess: bool = False,
    temperature: Optional[float] = None,
    output_path: Optional[str] = None,
) -> None:

    conn = connect_state(state_path)
    line_offset = get_line_offset(conn)

    if mode == "baseline":
        process_baseline_mode(conn, input_path, max_defs, max_words_per_def, checkpoint_interval, line_offset)

    elif mode == "enhance":
        if use_lmstudio is None:
            sys.stderr.write("Enhance mode requires LM Studio configuration.\n")
            return
        process_enhance_mode(conn, use_lmstudio, max_words_per_def, checkpoint_interval, only_reprocess, temperature)

    elif mode == "export":
        # Export consolidated JSONL to stdout or to --output if provided
        # Note: output path is optional; if not provided, print to stdout.
        def emit(line: str) -> None:
            sys.stdout.write(line + "\n")
        # Choose output stream
        out_stream: io.TextIOBase
        if output_path:
            ensure_parent_dir(output_path)
            out_stream = open(output_path, "w", encoding="utf-8")
        else:
            out_stream = cast(io.TextIOBase, sys.stdout)

        # Detect schema type by checking if 'raw' column exists (simplified schema)
        cur = conn.execute("PRAGMA table_info(words);")
        columns = [row[1] for row in cur.fetchall()]
        is_simplified_schema = 'raw' in columns

        if is_simplified_schema:
            # Simplified schema: use 'raw' and 'definition' columns
            cur = conn.execute("SELECT word, COALESCE(raw, word) AS disp FROM words ORDER BY word ASC;")
            rows = cur.fetchall()
            for (w, disp) in rows:
                defs: List[str] = []
                cur2 = conn.execute(
                    "SELECT idx, definition FROM definitions WHERE word=? ORDER BY idx ASC;",
                    (w,),
                )
                for idx, line in cur2.fetchall():
                    defs.append(line)
                if defs:
                    rec = {"word": disp, "definitions": defs}
                    out_stream.write(json.dumps(rec, ensure_ascii=False) + "\n")
        else:
            # Original schema: use 'display_word' and 'current_line' columns
            cur = conn.execute("SELECT word, COALESCE(display_word, word) AS disp FROM words ORDER BY word ASC;")
            rows = cur.fetchall()
            for (w, disp) in rows:
                defs: List[str] = []
                cur2 = conn.execute(
                    "SELECT idx, current_line FROM definitions WHERE word=? ORDER BY idx ASC;",
                    (w,),
                )
                for idx, line in cur2.fetchall():
                    defs.append(line)
                if defs:
                    rec = {"word": disp, "definitions": defs}
                    out_stream.write(json.dumps(rec, ensure_ascii=False) + "\n")

        if out_stream is not sys.stdout:
            out_stream.flush()
            out_stream.close()
        return 0
        # Interactive TUI removed. Use CLI modes instead.


def _select_llm(cfg: Dict[str, Any], *, default_tier: str = "fast") -> Optional[Tuple[str, str]]:
    """Interactively select an LLM configuration from available tiers.

    Returns (url, model) tuple or None if cancelled.
    """
    tiers = {
        "fast": cfg.get("fast", {}),
        "balanced": cfg.get("balanced", {}),
        "accurate": cfg.get("verify", {}),
    }

    # Filter to only tiers with both url and model
    available_tiers = {}
    for tier_name, tier_cfg in tiers.items():
        if tier_cfg.get("url") and tier_cfg.get("model"):
            available_tiers[tier_name] = tier_cfg

    if not available_tiers:
        print("No LLM configurations available. Please configure in option 1.")
        return None

    # If only one tier available, use it
    if len(available_tiers) == 1:
        tier_name = list(available_tiers.keys())[0]
        tier_cfg = available_tiers[tier_name]
        print(f"Using {tier_name} LLM: {tier_cfg['model']}")
        return (tier_cfg["url"], tier_cfg["model"])

    # Multiple tiers available, let user choose
    print("\nAvailable LLM configurations:")
    tier_options = []
    default_idx = 0

    for i, (tier_name, tier_cfg) in enumerate(available_tiers.items(), 1):
        tier_options.append((tier_name, tier_cfg))
        marker = " (default)" if tier_name == default_tier else ""
        print(f"{i}) {tier_name}: {tier_cfg['model']}{marker}")
        if tier_name == default_tier:
            default_idx = i

    try:
        choice = input(f"Select LLM (1-{len(tier_options)}, blank = {default_idx}): ").strip()
        if not choice:
            choice = str(default_idx)
        choice_idx = int(choice) - 1
        if 0 <= choice_idx < len(tier_options):
            selected_tier, selected_cfg = tier_options[choice_idx]
            print(f"Selected {selected_tier} LLM: {selected_cfg['model']}")
            return (selected_cfg["url"], selected_cfg["model"])
        else:
            print("Invalid choice.")
            return None
    except (ValueError, KeyboardInterrupt):
        print("\nCancelled.")
        return None


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    # Config fallback: load defaults from TOML if present
    cfg = _load_simple_toml(getattr(args, "config", "")) if getattr(args, "config", None) else {}
    # If no CLI args provided, drop into the built-in TUI to configure and run
    no_args = (argv is None and len(sys.argv) <= 1) or (isinstance(argv, list) and len(argv) == 0)
    if no_args:
        return _tui(getattr(args, "config", "out/small_dictionary.toml"))

    # Fill defaults from config if missing
    if not args.state:
        args.state = str(cfg.get("state") or "out/state.sqlite3")
    if args.mode == "baseline" and not args.input and cfg.get("input"):
        args.input = str(cfg.get("input"))
    if args.mode == "export" and not args.output and cfg.get("output"):
        args.output = str(cfg.get("output"))
    # Populate LLM from config if not provided
    if args.mode == "enhance":
        if (not args.lmstudio_url or not args.lmstudio_model) and cfg.get("fast"):
            fast = cfg.get("fast") or {}
            args.lmstudio_url = str(fast.get("url") or args.lmstudio_url)
            args.lmstudio_model = str(fast.get("model") or args.lmstudio_model)
    if args.mode == "verify-lm":
        if (not args.lmstudio_url or not args.lmstudio_model) and cfg.get("verify"):
            ver = cfg.get("verify") or {}
            args.lmstudio_url = str(ver.get("url") or args.lmstudio_url)
            args.lmstudio_model = str(ver.get("model") or args.lmstudio_model)

    lm_cfg: Optional[LMStudioConfig] = None
    if args.mode == "enhance":
        if not args.lmstudio_url or not args.lmstudio_model:
            sys.stderr.write("Enhance mode requires --lmstudio-url and --lmstudio-model\n")
            return 2
        lm_cfg = LMStudioConfig(url=args.lmstudio_url, model=args.lmstudio_model, timeout=float(getattr(args, "lm_timeout", 120.0) or 120.0))
    elif args.lmstudio_url:
        # Allow specifying LM even in baseline; it will be ignored.
        lm_cfg = LMStudioConfig(url=args.lmstudio_url, model=args.lmstudio_model or "", timeout=float(getattr(args, "lm_timeout", 120.0) or 120.0))
    if args.mode == "baseline" and not args.input:
        sys.stderr.write("Baseline mode requires --input\n")
        return 2
    if not args.state:
        sys.stderr.write("State DB path is required (provide --state or set it in --config)\n")
        return 2

    # Dispatch modes
    if args.mode == "baseline":
        process_file(args.input, args.state, args.max_defs, args.max_words, args.checkpoint_interval, use_lmstudio=None, mode="baseline", output_path=args.output)
        return 0
    if args.mode == "enhance":
        process_file(args.input or "", args.state, args.max_defs, args.max_words, args.checkpoint_interval, use_lmstudio=lm_cfg, mode="enhance", only_reprocess=getattr(args, "only_reprocess", False), temperature=getattr(args, "temperature", None), output_path=args.output)
        return 0
    if args.mode == "export":
        process_file(args.input or "", args.state, args.max_defs, args.max_words, args.checkpoint_interval, use_lmstudio=None, mode="export", output_path=args.output)
        return 0
    if args.mode == "verify-lm":
        # LM-based verification: verify ALL definitions for complete coverage
        if not args.lmstudio_url or not args.lmstudio_model:
            sys.stderr.write("verify-lm mode requires --lmstudio-url and --lmstudio-model\n")
            return 2
        verifier = LMStudioConfig(url=args.lmstudio_url, model=args.lmstudio_model, timeout=float(getattr(args, "lm_timeout", 120.0) or 120.0))
        conn = connect_state(args.state)

        # Get ALL definitions from completed words (not just long ones)
        query = """
            SELECT d.word, d.idx, d.pos, d.source_first_sentence, d.current_line
            FROM definitions d
            JOIN words w ON d.word = w.word
            WHERE w.status='done'
            ORDER BY d.word, d.idx
        """

        if args.verify_sample > 0:
            query += f" LIMIT {args.verify_sample * 5}"  # Estimate 5 defs per word

        cur = conn.execute(query)
        definitions = cur.fetchall()

        total_definitions = len(definitions)
        processed_definitions = 0
        failed_definitions = 0
        short_definitions = 0
        llm_validated = 0
        words_to_reprocess = set()
        start_time = time.time()
        last_progress_time = start_time

        for word, idx, pos, src, line in definitions:
            src_text = (src or "").strip()
            line_text = (line or "").strip()
            word_count = _word_count(src_text)

            processed_definitions += 1

            # Handle short definitions (auto-valid)
            if word_count <= args.max_words:
                short_definitions += 1
                continue

            # LLM verification for longer definitions
            ok, reason = verify_with_lmstudio(
                verifier,
                source_sentence=src_text,
                candidate_line=line_text,
                expected_pos=pos,
                max_words=args.max_words,
                temperature=getattr(args, "temperature", None)
            )

            llm_validated += 1

            if not ok:
                failed_definitions += 1
                words_to_reprocess.add(word)
            # No tag annotation needed

            # Progress update
            now = time.time()
            if now - last_progress_time >= 1.0:
                elapsed = now - start_time
                progress = processed_definitions / total_definitions if total_definitions > 0 else 0
                eta = format_eta(elapsed, progress)
                progress_msg = f"Verify-LM: {processed_definitions}/{total_definitions} defs ({short_definitions} short, {llm_validated} LLM, {failed_definitions} failed), ETA:{eta}"
                sys.stderr.write("\r" + progress_msg)
                sys.stderr.flush()
                last_progress_time = now

        # Mark words for reprocessing
        for word in words_to_reprocess:
            set_lm_status(conn, word, "pending")
            set_reprocess_flag(conn, word, True)

        conn.commit()
        sys.stderr.write("\n")
        sys.stderr.flush()

        hit_rate = (llm_validated - failed_definitions) / llm_validated if llm_validated > 0 else 0
        print(f"Verify-LM: {total_definitions} definitions checked")
        print(f"  - {short_definitions} short definitions (auto-validated)")
        print(f"  - {llm_validated} definitions LLM-validated ({failed_definitions} failed)")
        print(f"  - {len(words_to_reprocess)} words marked for reprocessing")
        print(f"  - LLM hit rate: {hit_rate:.1%}")
        return 0
    if args.mode == "verify":
        # Rule-based verification: check ALL definitions for complete coverage
        conn = connect_state(args.state)

        # Get ALL definitions from completed words (not just long ones)
        query = """
            SELECT d.word, d.idx, d.pos, d.source_first_sentence, d.current_line
            FROM definitions d
            JOIN words w ON d.word = w.word
            WHERE w.status='done'
            ORDER BY d.word, d.idx
        """

        if args.verify_sample > 0:
            query += f" LIMIT {args.verify_sample * 5}"  # Estimate 5 defs per word

        cur = conn.execute(query)
        definitions = cur.fetchall()

        total_definitions = len(definitions)
        checked = 0
        failed = 0
        short_definitions = 0
        rule_validated = 0
        words_to_reprocess = set()

        for word, idx, pos, src, line in definitions:
            src_text = (src or "").strip()
            line_text = (line or "").strip()
            word_count = _word_count(src_text)

            checked += 1

            # Handle short definitions (auto-valid)
            if word_count <= args.max_words:
                short_definitions += 1
                continue

            # Rule-based validation for longer definitions
            ok, reason = sanity_check_line_with_reason(src_text, line_text, args.max_words, pos)
            rule_validated += 1

            if not ok:
                failed += 1
                words_to_reprocess.add(word)
            # No tag annotation needed

        # Mark words for reprocessing
        for word in words_to_reprocess:
            set_lm_status(conn, word, "pending")
            set_reprocess_flag(conn, word, True)

        conn.commit()
        hit_rate = (rule_validated - failed) / rule_validated if rule_validated > 0 else 0
        print(f"Verify (rule): {total_definitions} definitions checked")
        print(f"  - {short_definitions} short definitions (auto-validated)")
        print(f"  - {rule_validated} definitions rule-validated ({failed} failed)")
        print(f"  - {len(words_to_reprocess)} words marked for reprocessing")
        print(f"  - Rule hit rate: {hit_rate:.1%}")
        return 0

    if args.mode == "extract-pure-words":
        # Extract word list of pure words (no spaces, hyphens, apostrophes, etc.)
        import re
        conn = connect_state(args.state)
        cur = conn.execute("SELECT word FROM words ORDER BY word ASC;")
        words = [r[0] for r in cur.fetchall()]
        pure_words = [w for w in words if re.match(r'^[a-zA-Z]+$', w)]
        for word in pure_words:
            print(word)
        conn.close()
        return 0

    if args.mode == "generate-pure-db":
        # Generate new database file limited to pure words with simplified schema
        import re
        if not args.output:
            sys.stderr.write("generate-pure-db mode requires --output\n")
            return 2

        # Connect to source database (read-only)
        conn = connect_state(args.state)

        # Create new database with custom schema (not using connect_state to avoid default tables)
        ensure_parent_dir(args.output)
        new_conn = sqlite3.connect(args.output)
        new_conn.execute("PRAGMA journal_mode=WAL;")
        new_conn.execute("PRAGMA synchronous=NORMAL;")

        # Create simplified schema for pure words database
        new_conn.execute("""
            CREATE TABLE words (
                word TEXT PRIMARY KEY,
                raw TEXT
            );
        """)

        new_conn.execute("""
            CREATE TABLE definitions (
                word TEXT NOT NULL,
                idx INTEGER NOT NULL,
                pos TEXT,
                definition TEXT NOT NULL,
                PRIMARY KEY (word, idx)
            );
        """)

        # Ensure no extra columns exist (in case connect_state was called previously)
        try:
            new_conn.execute("ALTER TABLE words DROP COLUMN reprocess;")
        except:
            pass
        try:
            new_conn.execute("ALTER TABLE words DROP COLUMN display_word;")
        except:
            pass
        try:
            new_conn.execute("ALTER TABLE words DROP COLUMN status;")
        except:
            pass
        try:
            new_conn.execute("ALTER TABLE words DROP COLUMN updated_at;")
        except:
            pass
        try:
            new_conn.execute("ALTER TABLE definitions DROP COLUMN source_first_sentence;")
        except:
            pass
        try:
            new_conn.execute("ALTER TABLE definitions DROP COLUMN enhanced_source;")
        except:
            pass

        # Get pure words
        cur = conn.execute("SELECT word FROM words ORDER BY word ASC;")
        words = [r[0] for r in cur.fetchall()]
        pure_words = [w for w in words if re.match(r'^[a-zA-Z]+$', w)]

        print(f"Found {len(pure_words)} pure words to process...")

        # Copy pure words with simplified schema
        batch_size = 1000
        for i in range(0, len(pure_words), batch_size):
            batch = pure_words[i:i+batch_size]
            placeholders = ','.join('?' * len(batch))
            cur_words = conn.execute(f"SELECT word, display_word FROM words WHERE word IN ({placeholders})", batch)
            word_rows = cur_words.fetchall()

            # Insert words with 'raw' column (renamed from display_word)
            for word_row in word_rows:
                word, display_word = word_row
                new_conn.execute("INSERT INTO words(word, raw) VALUES(?, ?)", (word, display_word))

        # Copy definitions for pure words with simplified schema
        for i in range(0, len(pure_words), batch_size):
            batch = pure_words[i:i+batch_size]
            placeholders = ','.join('?' * len(batch))
            cur_defs = conn.execute(f"SELECT word, idx, pos, current_line FROM definitions WHERE word IN ({placeholders})", batch)
            def_rows = cur_defs.fetchall()

            # Insert definitions with 'definition' column (renamed from current_line)
            for def_row in def_rows:
                word, idx, pos, current_line = def_row
                new_conn.execute("INSERT INTO definitions(word, idx, pos, definition) VALUES(?, ?, ?, ?)",
                               (word, idx, pos, current_line))

        new_conn.commit()
        conn.close()
        new_conn.close()

        print(f"Generated pure words database at {args.output}")
        print(f"- Words: {len(pure_words)}")
        print(f"- Schema: words(word, raw), definitions(word, idx, pos, definition)")

        return 0

    if args.mode == "cleanup-tags":
        # Remove all tags from current_line in definitions
        import re
        if not args.state:
            sys.stderr.write("cleanup-tags mode requires --state\n")
            return 2

        conn = connect_state(args.state)

        # Find definitions with tags
        cur = conn.execute("SELECT word, idx, current_line FROM definitions WHERE current_line LIKE '%[%' AND current_line LIKE '%]%'")
        tagged_definitions = cur.fetchall()

        print(f"Found {len(tagged_definitions)} definitions with tags to clean...")

        cleaned_count = 0
        for word, idx, current_line in tagged_definitions:
            # Remove all tags like [VERIFIED SHORT], [FAILED VERIFICATION: reason], etc.
            cleaned_line = re.sub(r'\s*\[([^\]]+)\]', '', current_line).strip()

            if cleaned_line != current_line:
                conn.execute(
                    "UPDATE definitions SET current_line=? WHERE word=? AND idx=?",
                    (cleaned_line, word, idx)
                )
                cleaned_count += 1

        conn.commit()
        conn.close()

        print(f"Cleaned {cleaned_count} definitions by removing tags")
        return 0


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", help="Input JSONL file")
    p.add_argument("--state", help="SQLite state DB path")
    p.add_argument("--output", help="Output path for export")
    p.add_argument("--mode", choices=["baseline","enhance","export","verify","verify-lm","manual-review","extract-pure-words","generate-pure-db","cleanup-tags"], default="baseline")
    p.add_argument("--lmstudio-url", help="LM Studio URL")
    p.add_argument("--lmstudio-model", help="LM Studio model")
    p.add_argument("--lm-timeout", type=int, default=120)
    p.add_argument("--max-defs", type=int, default=5)
    p.add_argument("--max-words", type=int, default=25)
    p.add_argument("--checkpoint-interval", type=int, default=100)
    p.add_argument("--only-reprocess", action="store_true")
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--verify-sample", type=int, default=0, help="Only verify a sample of N words (0 = all)")
    p.add_argument("--config", help="Path to config toml", default="out/small_dictionary.toml")
    return p.parse_args(argv)


# ---- Minimal TUI and helpers (restored) ----
def _ask(prompt: str, default: Optional[str] = None) -> str:
    if default:
        resp = input(f"{prompt} [{default}]: ").strip()
        return resp or default
    return input(f"{prompt}: ").strip()


def _confirm(prompt: str, default: bool = True) -> bool:
    d = 'Y/n' if default else 'y/N'
    try:
        r = input(f"{prompt} [{d}]: ").strip().lower()
    except Exception:
        return default
    if not r:
        return default
    return r[0] == 'y'


def _hr_size(num: int) -> str:
    # human-readable bytes
    for unit in ['B','KB','MB','GB','TB']:
        if num < 1024.0:
            return f"{num:.1f}{unit}"
        num /= 1024.0
    return f"{num:.1f}PB"


def _save_simple_toml(path: str, cfg: Dict[str, Any]) -> None:
    # Minimal persistence: write JSON as a pragmatic fallback to avoid adding toml dependency.
    ensure_parent_dir(path)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _load_simple_toml(path: str) -> Dict[str, Any]:
    if not path:
        return {}
    try:
        # Prefer stdlib tomllib when available (Python 3.11+)
        try:
            import tomllib  # type: ignore
            with open(path, "rb") as f:
                return tomllib.load(f)  # type: ignore
        except Exception:
            # Fallback: try JSON (in case file is JSON) or simple key=value parsing
            with open(path, "r", encoding="utf-8") as f:
                txt = f.read()
            try:
                return json.loads(txt)
            except Exception:
                # Very small TOML-ish parser for key = "value" and simple tables [section]
                out: Dict[str, Any] = {}
                cur_section: Optional[Dict[str, Any]] = None
                for line in txt.splitlines():
                    line = line.split('#', 1)[0].strip()
                    if not line:
                        continue
                    if line.startswith('[') and line.endswith(']'):
                        section = line[1:-1].strip()
                        cur_section = out.setdefault(section, {})
                        continue
                    m = re.match(r"^([A-Za-z0-9_]+)\s*=\s*(.*)$", line)
                    if not m:
                        continue
                    k = m.group(1)
                    v = m.group(2).strip()
                    # Remove surrounding quotes
                    if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
                        v = v[1:-1]
                    else:
                        # Try to parse numbers / booleans
                        if v.lower() in ("true", "false"):
                            v = (v.lower() == "true")
                        else:
                            try:
                                if '.' in v:
                                    v = float(v)
                                else:
                                    v = int(v)
                            except Exception:
                                pass
                    if cur_section is not None:
                        cur_section[k] = v
                    else:
                        out[k] = v
                return out
    except Exception:
        return {}


def _press_enter() -> None:
    try:
        input("Press Enter to continue...")
    except Exception:
        pass


def _validate_cfg(cfg: Dict[str, Any]) -> Tuple[bool, List[str]]:
    # Minimal validation for the fields the TUI relies on.
    errors: List[str] = []
    if not cfg.get("state"):
        errors.append("state is not set")
    return (len(errors) == 0, errors)


def _db_counts(state_path: str) -> Tuple[int, int, int, int]:
    try:
        conn = connect_state(state_path)
        cur = conn.execute("SELECT COUNT(*) FROM words;")
        total = cur.fetchone()[0]
        cur = conn.execute("SELECT COUNT(*) FROM words WHERE status='pending';")
        pending = cur.fetchone()[0]
        cur = conn.execute("SELECT COUNT(*) FROM words WHERE reprocess=1;")
        reprocess = cur.fetchone()[0]
        cur = conn.execute("SELECT COUNT(*) FROM words WHERE status='done';")
        done = cur.fetchone()[0]
        conn.close()
        return int(total), int(pending), int(reprocess), int(done)
    except Exception:
        return 0, 0, 0, 0


def manual_review_entries(state_path: str, max_words: int, use_lm_validation: Optional[LMStudioConfig] = None) -> int:
    # Minimal manual review: iterate words that need review (reprocess=1 or failed verification), allow inline edits, or mark reprocess
    conn = connect_state(state_path)
    try:
        # Only select words that need manual review
        curw = conn.execute("SELECT word FROM words WHERE reprocess=1 OR status='pending' ORDER BY word ASC;")
        words = [r[0] for r in curw.fetchall()]
        for w in words:
            curd = conn.execute("SELECT idx, pos, source_first_sentence, current_line FROM definitions WHERE word=? ORDER BY idx ASC;", (w,))
            rows = curd.fetchall()
            # Try to read a nicer display_word if present
            try:
                disp_row = conn.execute("SELECT display_word FROM words WHERE word=?;", (w,)).fetchone()
                display = disp_row[0] if disp_row and disp_row[0] else w
            except Exception:
                display = w

            for idx, pos, src, line in rows:
                print('\n' + '=' * 72)
                print(f"Word: {display}  (key: {w})")
                print(f"Definition #{idx}    Part-of-speech: {pos or 'unknown'}")
                print('-' * 72)
                print("Source sentence:")
                print(f"  {src}")
                print()
                print("Current stored definition:")
                print(f"  {line}")
                print('-' * 72)
                if use_lm_validation:
                    ok, reason = verify_with_lmstudio(use_lm_validation, source_sentence=src or "", candidate_line=line or "", expected_pos=pos, max_words=max_words)
                    print(f"LLM validation: {'PASS' if ok else 'FAIL'}  (reason: {reason})")
                try:
                    choice = input("Choose: (e)dit / (k)eep / (r)eprocess / (q)uit: ").strip().lower()
                except KeyboardInterrupt:
                    choice = 'q'
                if choice == 'q':
                    conn.commit()
                    return 0
                if choice == 'e':
                    new = input("New definition (blank to cancel): ").strip()
                    if new:
                        conn.execute("UPDATE definitions SET current_line=?, enhanced_source=? WHERE word=? AND idx=?", (new, None, w, idx))
                        set_lm_status(conn, w, 'done')
                        set_reprocess_flag(conn, w, False)
                        print('Updated stored definition and marked as completed.')
                elif choice == 'k':
                    set_lm_status(conn, w, 'done')
                    set_reprocess_flag(conn, w, False)
                    print('Kept current definition and marked as completed.')
                elif choice == 'r':
                    set_lm_status(conn, w, 'pending')
                    set_reprocess_flag(conn, w, True)
                    print('Marked for reprocessing.')
        conn.commit()
    finally:
        conn.close()
    return 0


def _tui(config_path: str) -> int:
    cfg = _load_simple_toml(config_path) or {}
    # ensure defaults
    cfg.setdefault('state', cfg.get('state', 'out/state.sqlite3'))
    cfg.setdefault('input', cfg.get('input', ''))
    cfg.setdefault('output', cfg.get('output', 'out/small_dictionary.final.jsonl'))
    cfg.setdefault('max_defs', cfg.get('max_defs', 5))
    cfg.setdefault('max_words', cfg.get('max_words', 25))
    cfg.setdefault('checkpoint', cfg.get('checkpoint', 100))
    cfg.setdefault('lm_timeout', cfg.get('lm_timeout', 120))

    while True:
        print('\nSmall Dictionary Builder - TUI')
        t, p, r, d = _db_counts(cfg.get('state'))
        print(f"DB: total={t} done={d} pending={p} reprocess={r}")
        print('1) Configure')
        print('2) Baseline (build from input JSONL)')
        print('3) Enhance pending (LM)')
        print('4) Verify (LM)')
        print('5) Enhance (reprocess-only)')
        print('6) Manual review')
        print('7) Export (final JSONL)')
        print('8) Compact DB (create a compact copy)')
        print('9) Rule-based verify')
        print('10) Extract pure words')
        print('11) Generate pure words DB')
        print('12) Cleanup tags from definitions')
        print('13) Quit')
        try:
            choice = input('Select an option [1-13]: ').strip()
        except KeyboardInterrupt:
            return 0

        if choice == '1':
            # Full configure flow
            cfg['state'] = _ask('State DB path', cfg.get('state'))
            cfg['input'] = _ask('Input JSONL (baseline)', cfg.get('input'))
            cfg['output'] = _ask('Export JSONL path', cfg.get('output'))
            try:
                cfg['max_defs'] = int(_ask('Max defs per word', str(cfg.get('max_defs', 5))))
                cfg['max_words'] = int(_ask('Max words per def', str(cfg.get('max_words', 25))))
                cfg['checkpoint'] = int(_ask('Checkpoint interval', str(cfg.get('checkpoint', 100))))
                cfg['lm_timeout'] = int(_ask('LM HTTP timeout (seconds)', str(cfg.get('lm_timeout', 120))))
            except Exception:
                pass
            cfg.setdefault('fast', {})
            cfg['fast']['url'] = _ask('Fast LLM URL', cfg['fast'].get('url', ''))
            cfg['fast']['model'] = _ask('Fast LLM model', cfg['fast'].get('model', ''))
            cfg.setdefault('balanced', {})
            cfg['balanced']['url'] = _ask('Balanced LLM URL', cfg['balanced'].get('url', cfg['fast'].get('url', '')))
            cfg['balanced']['model'] = _ask('Balanced LLM model', cfg['balanced'].get('model', cfg['fast'].get('model', '')))
            cfg.setdefault('verify', {})
            cfg['verify']['url'] = _ask('Accurate LLM URL', cfg['verify'].get('url', cfg['balanced'].get('url', '')))
            cfg['verify']['model'] = _ask('Accurate LLM model', cfg['verify'].get('model', cfg['balanced'].get('model', '')))
            cfg['initialized'] = True
            _save_simple_toml(config_path, cfg)
            print('Saved configuration.')
            _press_enter()
            continue

        if not cfg.get('initialized'):
            print('Please configure first (option 1) — loading defaults from config or creating it now.')
            _press_enter()
            continue

        if choice == '2':
            # Baseline: ensure input exists
            if not cfg.get('input'):
                cfg['input'] = _ask('Input JSONL (baseline)', cfg.get('input', ''))
            inp = cfg.get('input')
            if not inp or not os.path.exists(inp):
                print(f"Input path '{inp}' does not exist. Please configure a valid input.")
                _press_enter(); continue
            print('\nStarting baseline...')
            print(f"Input: {inp}\nState DB: {cfg.get('state')}")
            try:
                rc = main(['--input', inp, '--state', cfg.get('state'), '--mode', 'baseline', '--max-defs', str(cfg.get('max_defs')), '--max-words', str(cfg.get('max_words')), '--checkpoint-interval', str(cfg.get('checkpoint'))])
                print(f'Baseline finished with code {rc}')
            except SystemExit as se:
                print(f'Baseline exited: {se}')
            _press_enter(); continue

        if choice == '3':
            sel = _select_llm(cfg)
            if not sel:
                _press_enter(); continue
            url, model = sel
            # Count pending words before running
            try:
                conn = connect_state(cfg.get('state'))
                pending = conn.execute("SELECT COUNT(*) FROM words WHERE status='pending'").fetchone()[0]
                conn.close()
            except Exception:
                pending = None
            print(f"LM: {model} @ {url}")
            if pending is not None:
                print(f"Pending words: {pending}")
                if pending == 0:
                    print('No pending words to enhance.')
                    _press_enter(); continue
            t_in = _ask('Temperature for LLM (blank = default)', '')
            try:
                temp = float(t_in) if t_in else None
            except Exception:
                temp = None
            # Warm up the chosen model so LM Studio loads/switches the model before the run
            ok, elapsed, rmodel = warm_up_model(LMStudioConfig(url=url, model=model), temperature=temp)
            print(f'Warm-up: ok={ok} elapsed={elapsed:.1f}s model={rmodel or model}')
            # proceed with enhancement
            try:
                args_list = ['--state', cfg.get('state'), '--mode', 'enhance', '--lmstudio-url', url, '--lmstudio-model', model, '--lm-timeout', str(cfg.get('lm_timeout')), '--max-defs', str(cfg.get('max_defs')), '--max-words', str(cfg.get('max_words')), '--checkpoint-interval', str(cfg.get('checkpoint'))]
                if temp is not None:
                    args_list += ['--temperature', str(temp)]
                rc = main(args_list)
                print(f'Enhance finished with code {rc}')
            except SystemExit as se:
                print(f'Enhance exited: {se}')
            _press_enter(); continue

        if choice == '4':
            sel = _select_llm(cfg, default_tier='accurate')
            if not sel:
                _press_enter(); continue
            url, model = sel
            sample = _ask('Verify sample size (0 = all)', '0')
            try:
                sample_n = int(sample)
            except Exception:
                sample_n = 0
            # Count done words
            try:
                conn = connect_state(cfg.get('state'))
                done = conn.execute("SELECT COUNT(*) FROM words WHERE status='done'").fetchone()[0]
                conn.close()
            except Exception:
                done = None
            if done is not None:
                if sample_n == 0:
                    print(f"Will verify all {done} words with the LM.")
                else:
                    print(f"Will verify up to {sample_n} words (out of {done})")
            # Warm up model prior to verification to ensure LM Studio is using the target model
            ok, elapsed, rmodel = warm_up_model(LMStudioConfig(url=url, model=model), temperature=None)
            print(f'Warm-up: ok={ok} elapsed={elapsed:.1f}s model={rmodel or model}')
            try:
                args_list = ['--state', cfg.get('state'), '--mode', 'verify-lm', '--lmstudio-url', url, '--lmstudio-model', model, '--lm-timeout', str(cfg.get('lm_timeout')), '--max-words', str(cfg.get('max_words'))]
                if sample_n:
                    args_list += ['--verify-sample', str(sample_n)]
                rc = main(args_list)
                print(f'Verify-LM finished with code {rc}')
            except SystemExit as se:
                print(f'Verify-LM exited: {se}')
            _press_enter(); continue

        if choice == '5':
            sel = _select_llm(cfg)
            if not sel: _press_enter(); continue
            url, model = sel
            # Warm up model before running reprocess-only enhancement
            ok, elapsed, rmodel = warm_up_model(LMStudioConfig(url=url, model=model), temperature=None)
            print(f'Warm-up: ok={ok} elapsed={elapsed:.1f}s model={rmodel or model}')
            try:
                rc = main(['--state', cfg.get('state'), '--mode', 'enhance', '--only-reprocess', '--lmstudio-url', url, '--lmstudio-model', model, '--lm-timeout', str(cfg.get('lm_timeout'))])
                print(f'Enhance (reprocess-only) finished with code {rc}')
            except SystemExit as se:
                print(f'Enhance exited: {se}')
            _press_enter(); continue

        if choice == '6':
            sel = _select_llm(cfg, default_tier='balanced')
            lm_cfg = None
            if sel:
                url, model = sel
                lm_cfg = LMStudioConfig(url=url, model=model)
            manual_review_entries(cfg.get('state'), max_words=cfg.get('max_words', 25), use_lm_validation=lm_cfg)
            _press_enter(); continue

        if choice == '7':
            outp = _ask('Export JSONL path', cfg.get('output'))
            try:
                rc = main(['--state', cfg.get('state'), '--mode', 'export', '--output', outp])
                print(f'Export finished with code {rc}')
            except SystemExit as se:
                print(f'Export exited: {se}')
            _press_enter(); continue

        if choice == '8':
            dest = _ask('Compact DB output path', os.path.join('out','dictionary.compact.sqlite3'))
            ensure_parent_dir(dest)
            print(f'Creating compact DB at {dest}...')
            try:
                # create compact copy by reading schema and inserting rows
                src = cfg.get('state')
                # create new empty db with schema
                new_conn = connect_state(dest)
                old_conn = connect_state(src)
                # copy words
                old_words = old_conn.execute('SELECT word,status,updated_at,reprocess,display_word FROM words').fetchall()
                new_conn.executemany('INSERT OR REPLACE INTO words(word,status,updated_at,reprocess,display_word) VALUES(?,?,?,?,?)', old_words)
                # copy definitions
                old_defs = old_conn.execute('SELECT word,idx,pos,source_first_sentence,current_line,enhanced_source FROM definitions').fetchall()
                new_conn.executemany('INSERT OR REPLACE INTO definitions(word,idx,pos,source_first_sentence,current_line,enhanced_source) VALUES(?,?,?,?,?,?)', old_defs)
                # copy search tables
                try:
                    sr = old_conn.execute('SELECT word,definition_idx,source,title,snippet,url,rank,created_at FROM search_results').fetchall()
                    if sr:
                        new_conn.executemany('INSERT INTO search_results(word,definition_idx,source,title,snippet,url,rank,created_at) VALUES(?,?,?,?,?,?,?,?)', sr)
                except Exception:
                    pass
                try:
                    sm = old_conn.execute('SELECT word,definition_idx,search_term,search_engines,total_results,helpful_results,created_at FROM search_metadata').fetchall()
                    if sm:
                        new_conn.executemany('INSERT INTO search_metadata(word,definition_idx,search_term,search_engines,total_results,helpful_results,created_at) VALUES(?,?,?,?,?,?,?)', sm)
                except Exception:
                    pass
                new_conn.commit()
                new_conn.execute('VACUUM')
                new_conn.commit()
                old_conn.close()
                new_conn.close()
                print('Compact DB created.')
            except Exception as e:
                print('Compact failed:', e)
            _press_enter(); continue

        if choice == '9':
            # Default to 0 (verify all) unless the user specifies a sample size
            sample = _ask('Verify sample size (0 = all)', '0')
            try:
                sample_n = int(sample)
            except Exception:
                sample_n = 0
            try:
                rc = main(['--state', cfg.get('state'), '--mode', 'verify', '--max-words', str(cfg.get('max_words')), '--verify-sample', str(sample_n) if sample_n else '0'])
                print(f'Rule-based verify finished with code {rc}')
            except SystemExit as se:
                print(f'Verify exited: {se}')
            _press_enter(); continue

        if choice == '10':
            outp = _ask('Pure words output file (blank = stdout)', '')
            try:
                if outp:
                    with open(outp, 'w', encoding='utf-8') as f:
                        import sys
                        old_stdout = sys.stdout
                        sys.stdout = f
                        rc = main(['--state', cfg.get('state'), '--mode', 'extract-pure-words'])
                        sys.stdout = old_stdout
                else:
                    rc = main(['--state', cfg.get('state'), '--mode', 'extract-pure-words'])
                print(f'Extract pure words finished with code {rc}')
            except SystemExit as se:
                print(f'Extract pure words exited: {se}')
            _press_enter(); continue

        if choice == '11':
            outp = _ask('Pure words DB output path', os.path.join('out','pure_dictionary.sqlite3'))
            try:
                rc = main(['--state', cfg.get('state'), '--mode', 'generate-pure-db', '--output', outp])
                print(f'Generate pure words DB finished with code {rc}')
            except SystemExit as se:
                print(f'Generate pure words DB exited: {se}')
            _press_enter(); continue

        if choice == '12':
            try:
                rc = main(['--state', cfg.get('state'), '--mode', 'cleanup-tags'])
                print(f'Cleanup tags finished with code {rc}')
            except SystemExit as se:
                print(f'Cleanup tags exited: {se}')
            _press_enter(); continue

        if choice == '13':
            print('Bye.')
            return 0

        print('Unknown option.')
        _press_enter()


if __name__ == "__main__":
    rc = main(None)
    sys.exit(rc)
