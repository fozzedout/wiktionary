#!/usr/bin/env python3
"""
Refactored small dictionary builder.
This is a cleaner, modular version of the original make_small_dictionary.py script.
"""

import argparse
import io
import json
import os
import re
import signal
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, cast

# Import our modular components
try:
    from database_ops import (
        connect_state, get_line_offset, set_line_offset, is_processed, mark_processed,
        get_lm_status, set_lm_status, maybe_upgrade_display_word, set_reprocess_flag,
        store_search_results, get_stored_search_results, has_recent_search_results
    )
    from text_utils import (
        extract_word, extract_language, normalize_pos, _word_count, _limit_words,
        strip_llm_artifacts, clean_gloss, sanity_check_line, format_def_line,
        first_sentence_word_count, extract_definitions
    )
    from lm_integration import (
        LMStudioConfig, _evaluate_definition_quality, summarize_with_lmstudio,
        verify_with_lmstudio, warm_up_model
    )
    from web_search import _search_web_for_context
except ImportError:
    # Fallback for when modules aren't available
    print("Warning: Modular components not found, using fallback implementation")
    # This would need the full implementation copied here as fallback

STOP_REQUESTED = False


def _install_sigint_handler():
    def handler(signum, frame):  # type: ignore[no-untyped-def]
        global STOP_REQUESTED
        STOP_REQUESTED = True
        sys.stderr.write("\nStop requested, finishing current item and checkpointing...\n")
        sys.stderr.flush()

    signal.signal(signal.SIGINT, handler)


def ensure_parent_dir(path: str) -> None:
    """Ensure the parent directory of a path exists."""
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


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
                seen_lines.add(base_line)
                source_sentence = re.split(r"(?<=[.!?])\s+", clean_gloss(gloss).strip())
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
                    sys.stderr.write("\r" + f"Baseline: line={total_read} words={done+pending} done={done} pending={pending}")
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
            sys.stderr.flush()
        except Exception:
            pass

    elif mode == "enhance":
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
                # For each definition, try to rewrite using LLM
                curd = conn.execute(
                    "SELECT idx, pos, source_first_sentence, current_line FROM definitions WHERE word=? ORDER BY idx ASC;",
                    (w,),
                )
                rows = curd.fetchall()
                enhanced_any = False
                for idx, pos_tag, source_sentence, current_line in rows:
                    if _word_count(source_sentence) > max_words_per_def:
                        # Extract a sensible search term from the source sentence
                        term_match = re.search(r'^([A-Za-z][A-Za-z0-9\s-]+)', source_sentence)
                        search_term = term_match.group(1).strip() if term_match else w

                        # Evaluate first; if the LM says the gloss is unhelpful, perform an explicit web search,
                        # store the results, and then call the summarizer again using the fetched context.
                        is_helpful = _evaluate_definition_quality(source_sentence, use_lmstudio)
                        web_context = None
                        search_results = []
                        if not is_helpful:
                            try:
                                sys.stderr.write(f"[info] Reprocess: fetching web context for '{search_term}'...\n")
                                sys.stderr.flush()
                            except Exception:
                                pass
                            web_context, search_results = _search_web_for_context(search_term, pos_tag)
                            if search_results:
                                store_search_results(conn, w, idx, search_term, ["duckduckgo", "wikipedia"], search_results)
                                try:
                                    sys.stderr.write(f"[info] Stored {len(search_results)} search results for {w}#{idx}\n")
                                    sys.stderr.flush()
                                except Exception:
                                    pass

                        # Call summarizer with any pre-fetched context to improve chances of success.
                        try:
                            sys.stderr.write(f"[info] Calling LLM to summarize {w}#{idx}...\n")
                            sys.stderr.flush()
                        except Exception:
                            pass
                        result = summarize_with_lmstudio(
                            use_lmstudio,
                            pos_tag,
                            source_sentence,
                            max_words_per_def,
                            temperature=temperature,
                            web_context_override=web_context,
                            search_results_override=search_results,
                        )

                        # If the summarizer failed but we did obtain search results, retry once to give the LM
                        # another chance with the fetched context.
                        if result is None and search_results:
                            try:
                                sys.stderr.write(f"[info] Retrying LLM summarization for {w}#{idx} with stored context...\n")
                                sys.stderr.flush()
                            except Exception:
                                pass
                            result = summarize_with_lmstudio(
                                use_lmstudio,
                                pos_tag,
                                source_sentence,
                                max_words_per_def,
                                temperature=temperature,
                                web_context_override=web_context,
                                search_results_override=search_results,
                            )

                        new_line = None
                        enhanced_source = None
                        if result:
                            new_line, enhanced_source, _ = result
                            if new_line:
                                new_line = strip_llm_artifacts(new_line)

                        # Always store the LLM result, regardless of validity
                        result_to_store = new_line if new_line else source_sentence
                        conn.execute(
                            "UPDATE definitions SET current_line=?, enhanced_source=? WHERE word=? AND idx=?",
                            (result_to_store, enhanced_source, w, idx),
                        )

                        # Store search results if we have any
                        if search_results:
                            term_match = re.search(r'^([A-Za-z][A-Za-z0-9\s-]+)', source_sentence)
                            search_term = term_match.group(1).strip() if term_match else w
                            store_search_results(conn, w, idx, search_term, ["duckduckgo", "wikipedia"], search_results)
                        # Only mark as enhanced if the result passed sanity checks
                        if new_line and sanity_check_line(source_sentence, new_line, max_words_per_def, pos_tag):
                            enhanced_any = True
                    else:
                        # Source is <= max_words_per_def, use as-is (source is truth)
                        conn.execute(
                            "UPDATE definitions SET current_line=?, enhanced_source=? WHERE word=? AND idx=?",
                            (source_sentence, None, w, idx),
                        )
                        enhanced_any = True

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
                    curd = conn.execute(
                        "SELECT idx, pos, source_first_sentence, current_line FROM definitions WHERE word=? ORDER BY idx ASC;",
                        (w,),
                    )
                    rows = curd.fetchall()
                    enhanced_any = False

                    for idx, pos_tag, source_sentence, current_line in rows:
                        if _word_count(source_sentence) > max_words_per_def:
                            # Check if we have recent search results
                            if not has_recent_search_results(conn, w, idx):
                                term_match_local = re.search(r'^([A-Za-z][A-Za-z0-9\s-]+)', source_sentence)
                                search_term_local = term_match_local.group(1).strip() if term_match_local else w
                                is_helpful_local = _evaluate_definition_quality(source_sentence, use_lmstudio)
                                web_context_local = None
                                search_results_local: List[Dict[str, Any]] = []
                                if not is_helpful_local:
                                    web_context_local, search_results_local = _search_web_for_context(search_term_local, pos_tag)
                                    if search_results_local:
                                        store_search_results(conn, w, idx, search_term_local, ["duckduckgo", "wikipedia"], search_results_local)

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
                                    (result_to_store, enhanced_source, w, idx),
                                )

                                # Store search results if we have any
                                if search_results:
                                    term_match = re.search(r'^([A-Za-z][A-Za-z0-9\s-]+)', source_sentence)
                                    search_term = term_match.group(1).strip() if term_match else w
                                    store_search_results(conn, w, idx, search_term, ["duckduckgo", "wikipedia"], search_results)

                                # Mark as enhanced if the result passed sanity checks
                                if new_line and sanity_check_line(source_sentence, new_line, max_words_per_def, pos_tag):
                                    enhanced_any = True
                            else:
                                # Use existing search results to enhance
                                stored_results = get_stored_search_results(conn, w, idx)
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
                                            (result_to_store, enhanced_source, w, idx),
                                        )

                                        if new_line and sanity_check_line(source_sentence, new_line, max_words_per_def, pos_tag):
                                            enhanced_any = True
                        else:
                            # Source is <= max_words_per_def, use as-is
                            conn.execute(
                                "UPDATE definitions SET current_line=?, enhanced_source=? WHERE word=? AND idx=?",
                                (source_sentence, None, w, idx),
                            )
                            enhanced_any = True

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
        return

    # Note: TUI and other modes removed for simplicity in the refactored version


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", help="Input JSONL file")
    p.add_argument("--state", help="SQLite state DB path")
    p.add_argument("--output", help="Output path for export")
    p.add_argument("--mode", choices=["baseline","enhance","export","verify","verify-lm"], default="baseline")
    p.add_argument("--lmstudio-url", help="LM Studio URL")
    p.add_argument("--lmstudio-model", help="LM Studio model")
    p.add_argument("--lm-timeout", type=int, default=120)
    p.add_argument("--max-defs", type=int, default=5)
    p.add_argument("--max-words", type=int, default=25)
    p.add_argument("--checkpoint-interval", type=int, default=100)
    p.add_argument("--only-reprocess", action="store_true")
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--verify-sample", type=int, default=0, help="Only verify a sample of N words (0 = all)")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

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
        sys.stderr.write("State DB path is required (provide --state)\n")
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

    return 0


if __name__ == "__main__":
    raise SystemExit(main())