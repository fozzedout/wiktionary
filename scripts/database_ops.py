#!/usr/bin/env python3
"""
Database operations for the small dictionary builder.
"""

import os
import sqlite3
from typing import Optional, Tuple


def ensure_parent_dir(path: str) -> None:
    """Ensure the parent directory of a path exists."""
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def connect_state(db_path: str) -> sqlite3.Connection:
    """Connect to the SQLite state database and set up tables."""
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
    """Get the current line offset from the progress table."""
    cur = conn.execute("SELECT line_offset FROM progress WHERE id=1;")
    row = cur.fetchone()
    return int(row[0]) if row else 0


def set_line_offset(conn: sqlite3.Connection, line_offset: int) -> None:
    """Set the current line offset in the progress table."""
    conn.execute("UPDATE progress SET line_offset=? WHERE id=1;", (line_offset,))
    conn.commit()


def is_processed(conn: sqlite3.Connection, word: str) -> bool:
    """Check if a word has been processed in baseline mode."""
    # Consider word processed in baseline if it already exists in 'words'.
    cur = conn.execute("SELECT 1 FROM words WHERE word=?;", (word.lower(),))
    return cur.fetchone() is not None


def mark_processed(conn: sqlite3.Connection, word: str) -> None:
    """Mark a word as processed (no-op retained for compatibility)."""
    return


def get_lm_status(conn: sqlite3.Connection, word: str) -> Optional[str]:
    """Get the LM processing status for a word."""
    cur = conn.execute("SELECT status FROM words WHERE word=?;", (word.lower(),))
    row = cur.fetchone()
    return row[0] if row else None


def set_lm_status(conn: sqlite3.Connection, word: str, status: str, *, display_word: Optional[str] = None) -> None:
    """Set the LM processing status for a word."""
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
    """Set the reprocess flag for a word."""
    try:
        conn.execute(
            "UPDATE words SET reprocess=?, updated_at=strftime('%s','now') WHERE word=?;",
            (1 if flag else 0, word.lower()),
        )
    except Exception:
        # Column may not exist if migration failed; ignore silently.
        pass


def store_search_results(conn: sqlite3.Connection, word: str, definition_idx: int, search_term: str, search_engines: list, search_results: list) -> None:
    """Store search results and metadata in the database."""
    try:
        import json
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


def get_stored_search_results(conn: sqlite3.Connection, word: str, definition_idx: int) -> list:
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
        import time
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