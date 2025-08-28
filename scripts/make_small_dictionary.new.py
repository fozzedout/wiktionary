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


STOP_REQUESTED = False


def _install_sigint_handler():
    def handler(signum, frame):  # type: ignore[no-untyped-def]
        global STOP_REQUESTED
        STOP_REQUESTED = True
        sys.stderr.write("\nStop requested, finishing current item and checkpointing...\n")
        sys.stderr.flush()

    signal.signal(signal.SIGINT, handler)


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
    # For non-TUI runs, install the SIGINT handler to checkpoint gracefully
    _install_sigint_handler()
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

    if args.mode == "report":
        conn = connect_state(args.state)
        cur = conn.execute("SELECT COUNT(*) FROM processed_words;")
        processed_count = cur.fetchone()[0]
        cur = conn.execute("SELECT COUNT(*) FROM words WHERE status='done';")
        lm_done = cur.fetchone()[0]
        cur = conn.execute("SELECT COUNT(*) FROM words WHERE status='pending';")
        lm_pending = cur.fetchone()[0]
        line_offset = get_line_offset(conn)
        print(json.dumps({
            "baseline_words": processed_count,
            "lm_done": lm_done,
            "lm_pending": lm_pending,
            "line_offset": line_offset,
        }, indent=2))
        return 0

    if args.mode == "export":
        conn = connect_state(args.state)
        # Choose output stream
        out_stream: io.TextIOBase
        if args.output:
            ensure_parent_dir(args.output)
            out_stream = open(args.output, "w", encoding="utf-8")
        else:
            out_stream = cast(io.TextIOBase, sys.stdout)
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

    if args.mode == "compact":
        if not args.output:
            sys.stderr.write("Compact mode requires --output path to the new SQLite file.\n")
            return 2
        src = connect_state(args.state)
        dst_path = os.path.abspath(args.output)
        ensure_parent_dir(dst_path)
        # Create destination DB
        if os.path.exists(dst_path):
            os.remove(dst_path)
        dst = sqlite3.connect(dst_path)
        dst.execute("PRAGMA journal_mode=OFF;")
        dst.execute("PRAGMA synchronous=OFF;")
        dst.execute(
            "CREATE TABLE terms (id INTEGER PRIMARY KEY, word TEXT NOT NULL UNIQUE);"
        )
        dst.execute(
            "CREATE TABLE defs (term_id INTEGER NOT NULL, idx INTEGER NOT NULL, line TEXT NOT NULL, PRIMARY KEY(term_id, idx)) WITHOUT ROWID;"
        )
        # Copy data in batches
        cur = src.execute("SELECT word, COALESCE(display_word, word) AS disp FROM words ORDER BY word ASC;")
        count = 0
        for (w, disp) in cur.fetchall():
            dst.execute("INSERT INTO terms(word) VALUES(?);", (disp,))
            # Get term_id
            tid = dst.execute("SELECT id FROM terms WHERE word=?;", (disp,)).fetchone()[0]
            for idx, line in src.execute(
                "SELECT idx, current_line FROM definitions WHERE word=? ORDER BY idx ASC;",
                (w,),
            ):
                dst.execute(
                    "INSERT INTO defs(term_id, idx, line) VALUES(?,?,?);",
                    (tid, idx, line),
                )
            count += 1
            if count % 1000 == 0:
                dst.commit()
        dst.commit()
        try:
            dst.execute("PRAGMA optimize;")
        except Exception:
            pass
        dst.close()
        return 0

    if args.mode == "verify":
        conn = connect_state(args.state)
        # Iterate through definitions and check current_line vs. source_first_sentence
        total = 0
        checked = 0
        failed = 0
        failures: List[Dict[str, Any]] = []
        words_to_mark: set[str] = set()
        cur = conn.execute(
            "SELECT d.word, d.idx, d.pos, d.source_first_sentence, d.current_line FROM definitions d ORDER BY d.word, d.idx;"
        )
        for word, idx, pos, src, line in cur.fetchall():
            total += 1
            # Only consider entries that would be enhanced (long sources), to align with enhance behavior
            if _word_count(src or "") <= args.max_words:
                continue
            ok, reason = sanity_check_line_with_reason(src or "", line or "", args.max_words, pos)
            checked += 1
            if not ok:
                failed += 1
                words_to_mark.add(str(word))
                if len(failures) < max(0, args.verify_sample):
                    failures.append({
                        "word": word,
                        "idx": idx,
                        "pos": pos,
                        "reason": reason,
                        "source": src,
                        "line": line,
                    })
        # Mark words that failed as pending + reprocess
        marked_reprocess = 0
        for w in words_to_mark:
            set_lm_status(conn, w, "pending")
            set_reprocess_flag(conn, w, True)
            marked_reprocess += 1
        conn.commit()
        report = {
            "total_defs": total,
            "checked_defs": checked,
            "failed_defs": failed,
            "fail_rate": (failed / checked) if checked else 0.0,
            "marked_reprocess": marked_reprocess,
            "sample_failures": failures,
        }
        if args.output:
            ensure_parent_dir(args.output)
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(json.dumps(report, ensure_ascii=False, indent=2))
        else:
            sys.stdout.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        return 0

    if args.mode == "manual-review":
        # Support optional LLM validation in manual review
        lm_validation_cfg = None
        if args.lmstudio_url and args.lmstudio_model:
            lm_validation_cfg = LMStudioConfig(
                url=args.lmstudio_url, 
                model=args.lmstudio_model, 
                timeout=float(getattr(args, "lm_timeout", 120.0) or 120.0)
            )
        manual_review_entries(args.state, args.max_words, use_lm_validation=lm_validation_cfg)
        return 0

    if args.mode == "verify-lm":
        if not args.lmstudio_url or not args.lmstudio_model:
            sys.stderr.write("verify-lm mode requires --lmstudio-url and --lmstudio-model\n")
            return 2
        verifier = LMStudioConfig(url=args.lmstudio_url, model=args.lmstudio_model, timeout=float(getattr(args, "lm_timeout", 120.0) or 120.0))
        conn = connect_state(args.state)
        # Verify only words marked done; mark any failing words back to pending for reprocessing
        curw = conn.execute("SELECT word FROM words WHERE status='done' ORDER BY word ASC;")
        words = [r[0] for r in curw.fetchall()]
        total_words = len(words)
        checked_words = 0
        marked_reprocess = 0
        checked_defs = 0
        failed_defs = 0
        last_progress_time = time.time()
        start_time = time.time()
        batch_commit = 0
        for w in words:
            if STOP_REQUESTED:
                break
            # Pull all defs for the word
            curd = conn.execute(
                "SELECT idx, pos, source_first_sentence, current_line FROM definitions WHERE word=? ORDER BY idx ASC;",
                (w,),
            )
            rows = curd.fetchall()
            need_reprocess = False
            failed_reasons = {}  # Track reasons per definition index
            for idx, pos, src, line in rows:
                if STOP_REQUESTED:
                    break
                # Only verify those that would have been enhanced (long sources)
                if _word_count(src or "") <= args.max_words:
                    failed_reasons[idx] = "Skipped - too short"
                    continue

                # Perform verification and get result
                verification_result = verify_with_lmstudio(
                    verifier,
                    source_sentence=src or "",
                    candidate_line=line or "",
                    expected_pos=pos,
                    max_words=args.max_words,
                    temperature=getattr(args, "temperature", None),
                )

                # Safely extract results
                if isinstance(verification_result, (list, tuple)) and len(verification_result) >= 2:
                    ok = verification_result[0]
                    current_reason = verification_result[1]
                else:
                    ok = False
                    current_reason = "Invalid result format"

                checked_defs += 1
                if not ok:
                    failed_defs += 1
                    need_reprocess = True

                # Always track the reason for manual review
                failed_reasons[idx] = current_reason
            if STOP_REQUESTED:
                break
            if need_reprocess:
                set_lm_status(conn, w, "pending")
                # Mark explicitly for reprocessing so enhance can target only these if requested
                set_reprocess_flag(conn, w, True)

                # Update current_line with verification failure reasons for manual review
                for idx, pos, src, line in rows:
                    if idx in failed_reasons:
                        current_reason = failed_reasons[idx]
                        updated_line = line + f" [FAILED VERIFICATION: {current_reason}]"
                        conn.execute(
                            "UPDATE definitions SET current_line=?, enhanced_source=? WHERE word=? AND idx=?",
                            (updated_line, None, w, idx),
                        )

                marked_reprocess += 1
                batch_commit += 1
            checked_words += 1
            if batch_commit >= max(1, args.checkpoint_interval):
                conn.commit()
                batch_commit = 0
            # Time-based progress line overwrite
            try:
                now = time.time()
                if now - last_progress_time >= 1.0:
                    # Calculate progress and failure rate
                    progress_pct = (checked_words / total_words * 100) if total_words > 0 else 0
                    fail_rate = (failed_defs / checked_defs * 100) if checked_defs > 0 else 0

                    # Calculate ETA if we have enough data
                    eta_str = ""
                    if checked_words > 5:
                        elapsed = now - start_time
                        rate = checked_words / elapsed  # words per second
                        remaining = total_words - checked_words
                        eta_seconds = remaining / rate if rate > 0 else 0
                        if eta_seconds < 60:
                            eta_str = f" ETA: {eta_seconds:.0f}s"
                        elif eta_seconds < 3600:
                            eta_str = f" ETA: {eta_seconds/60:.1f}m"
                        else:
                            eta_str = f" ETA: {eta_seconds/3600:.1f}h"

                    # Format progress line
                    progress_line = (f"\rVerify-LM: {progress_pct:.1f}% complete "
                                   f"({checked_words}/{total_words} words) | "
                                   f"Definitions: {checked_defs} checked, {failed_defs} failed ({fail_rate:.1f}% fail rate) | "
                                   f"Reprocess: {marked_reprocess} words{eta_str}")
                    sys.stderr.write(progress_line)
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
        # Final summary to stdout
        summary = {
            "total_words": total_words,
            "checked_words": checked_words,
            "marked_reprocess": marked_reprocess,
            "verified_defs": checked_defs,
            "failed_defs": failed_defs,
            "model": args.lmstudio_model,
        }

        # Calculate final statistics
        completion_pct = (checked_words / total_words * 100) if total_words > 0 else 0
        fail_rate = (failed_defs / checked_defs * 100) if checked_defs > 0 else 0
        reprocess_rate = (marked_reprocess / checked_words * 100) if checked_words > 0 else 0

        # Print human-readable summary
        try:
            sys.stderr.write("\n" + "="*60 + "\n")
            sys.stderr.write("VERIFICATION COMPLETE\n")
            sys.stderr.write("="*60 + "\n")
            sys.stderr.write(f"Progress: {checked_words}/{total_words} words ({completion_pct:.1f}% complete)\n")
            sys.stderr.write(f"Definitions: {checked_defs} checked, {failed_defs} failed ({fail_rate:.1f}% failure rate)\n")
            sys.stderr.write(f"Reprocessing: {marked_reprocess} words need reprocessing ({reprocess_rate:.1f}% of checked words)\n")
            sys.stderr.write(f"Model: {args.lmstudio_model}\n")
            sys.stderr.write("="*60 + "\n")
            sys.stderr.flush()
        except Exception:
            pass

        if args.output:
            ensure_parent_dir(args.output)
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(json.dumps(summary, ensure_ascii=False, indent=2))
        else:
            sys.stdout.write(json.dumps(summary, ensure_ascii=False) + "\n")
        return 0

    try:
        process_file(
            input_path=args.input or "",
            state_path=args.state,
            max_defs=args.max_defs,
            max_words_per_def=args.max_words,
            checkpoint_interval=args.checkpoint_interval,
            use_lmstudio=lm_cfg,
            mode=args.mode,
            only_reprocess=getattr(args, "only_reprocess", False),
            temperature=getattr(args, "temperature", None),
        )
    except KeyboardInterrupt:
        # Should be handled by our signal handler, but be safe.
        sys.stderr.write("Interrupted.\n")
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
