#!/usr/bin/env python3
"""Stage-1 pipeline tests: ingest, plan-update, enhance (pocket defs), export-current."""

import json
import os
import sqlite3
import sys
import tempfile

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import scripts.make_small_dictionary as msd
from scripts.make_small_dictionary import (
    BUILDER_VERSION,
    ENHANCEMENT_VERSION,
    LMStudioConfig,
    connect_state,
    get_pipeline_status,
    hash_definition,
    hash_extract_file,
    hash_word,
    is_pure_alpha_word,
    mark_published,
    make_definition_key,
    normalize_for_hash,
    normalize_pos_for_hash,
    normalize_word_key,
    process_build_report,
    process_enhance_changed,
    process_export_current,
    process_ingest_extract,
    process_plan_update,
)

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def test_normalization():
    assert normalize_word_key("  Apple ") == "apple"
    assert normalize_word_key("RUN") == "run"
    assert normalize_pos_for_hash("adj") == "adjective"
    assert normalize_pos_for_hash("Noun") == "noun"
    assert normalize_pos_for_hash(None) == ""
    assert normalize_for_hash("  hello   world  ") == "hello world"
    assert normalize_for_hash("[[word]]") == "word"
    print("PASS: normalization")


def test_hash_stability():
    """Hashes must be deterministic and stable."""
    h1 = hash_definition("apple", "noun", "A fruit.")
    h2 = hash_definition("apple", "noun", "A fruit.")
    assert h1 == h2, "Hash not stable"

    h3 = hash_definition("apple", "noun", "A different fruit.")
    assert h1 != h3, "Different inputs should give different hashes"

    # Normalization should make these identical
    h4 = hash_definition("Apple", "Noun", "  A fruit. ")
    assert h1 == h4, "Normalized inputs should match"

    wh1 = hash_word("apple", "apple", [h1])
    wh2 = hash_word("apple", "apple", [h1])
    assert wh1 == wh2, "Word hash not stable"
    print("PASS: hash stability")


def test_pure_alpha():
    assert is_pure_alpha_word("apple") is True
    assert is_pure_alpha_word("ice-cream") is False
    assert is_pure_alpha_word("don't") is False
    assert is_pure_alpha_word("hello world") is False
    assert is_pure_alpha_word("ABC") is True
    print("PASS: pure alpha detection")


def test_enhancement_version():
    """Enhancement version should be pocket_v1."""
    assert ENHANCEMENT_VERSION == "pocket_v1"
    print("PASS: enhancement version")


def test_ingest_extract():
    """Ingest a tiny extract and verify tables are populated."""
    with tempfile.TemporaryDirectory() as tmpdir:
        state_path = os.path.join(tmpdir, "state.sqlite3")
        input_path = os.path.join(FIXTURES, "tiny_extract.jsonl")

        conn = connect_state(state_path)
        extract_id = process_ingest_extract(conn, input_path)

        # Verify extract_runs
        run = conn.execute("SELECT status, builder_version FROM extract_runs WHERE id=?", (extract_id,)).fetchone()
        assert run[0] == "complete", f"Expected complete, got {run[0]}"
        assert run[1] == BUILDER_VERSION

        # Verify source_words (should have apple, run, blue — not Haus which is German)
        words = conn.execute(
            "SELECT word_key FROM source_words WHERE extract_id=? ORDER BY word_key",
            (extract_id,),
        ).fetchall()
        word_keys = [r[0] for r in words]
        assert "apple" in word_keys
        assert "run" in word_keys
        assert "blue" in word_keys
        assert "haus" not in word_keys, "German word should be filtered"
        assert len(word_keys) == 3

        # Verify source_definitions
        apple_defs = conn.execute(
            "SELECT definition_key, pos, source_first_sentence FROM source_definitions WHERE extract_id=? AND word_key='apple'",
            (extract_id,),
        ).fetchall()
        assert len(apple_defs) == 1
        assert apple_defs[0][1] == "noun"

        # run should have 2 definitions (verb + noun from separate entries)
        run_defs = conn.execute(
            "SELECT definition_key, pos FROM source_definitions WHERE extract_id=? AND word_key='run'",
            (extract_id,),
        ).fetchall()
        assert len(run_defs) == 2

        # Verify is_pure_alpha
        apple_pure = conn.execute(
            "SELECT is_pure_alpha FROM source_words WHERE extract_id=? AND word_key='apple'",
            (extract_id,),
        ).fetchone()
        assert apple_pure[0] == 1

        # Re-ingest same file should skip
        extract_id2 = process_ingest_extract(conn, input_path)
        assert extract_id2 == extract_id, "Same file should return same extract_id"

        conn.close()
        print("PASS: ingest extract")


def test_plan_update_first_extract():
    """plan-update with only one extract (no prior)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        state_path = os.path.join(tmpdir, "state.sqlite3")
        input_path = os.path.join(FIXTURES, "tiny_extract.jsonl")
        report_path = os.path.join(tmpdir, "report.json")

        conn = connect_state(state_path)
        process_ingest_extract(conn, input_path)
        report = process_plan_update(conn, output_path=report_path)

        assert report["prior_extract_id"] is None
        assert report["new_words"] == 3  # apple, run, blue
        assert report["changed_words"] == 0
        assert report["deleted_words"] == 0
        assert report["unchanged_words"] == 0
        # All new words need pocket definitions
        assert report["words_to_regenerate"] == 3

        # Verify report file written
        assert os.path.exists(report_path)
        with open(report_path) as f:
            saved = json.load(f)
        assert saved["new_words"] == 3

        conn.close()
        print("PASS: plan update (first extract)")


def test_plan_update_diff():
    """plan-update comparing two extracts should detect changes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        state_path = os.path.join(tmpdir, "state.sqlite3")
        report_path = os.path.join(tmpdir, "report.json")

        conn = connect_state(state_path)

        # Ingest first extract
        id1 = process_ingest_extract(conn, os.path.join(FIXTURES, "tiny_extract.jsonl"))

        # Ingest second extract (changed)
        id2 = process_ingest_extract(conn, os.path.join(FIXTURES, "tiny_extract_changed.jsonl"))
        assert id2 != id1

        report = process_plan_update(conn, output_path=report_path)

        assert report["prior_extract_id"] == id1
        assert report["extract_id"] == id2
        # apple: unchanged, blue: unchanged, run: changed (verb definition changed)
        # green: new, Haus: was never included (German)
        assert report["unchanged_words"] == 2, f"Expected 2 unchanged, got {report['unchanged_words']}"
        assert report["changed_words"] == 1, f"Expected 1 changed, got {report['changed_words']}"
        assert report["new_words"] == 1, f"Expected 1 new (green), got {report['new_words']}"
        assert report["deleted_words"] == 0

        conn.close()
        print("PASS: plan update (diff)")


def test_export_current_single_definition():
    """export-current should produce one definition per word."""
    with tempfile.TemporaryDirectory() as tmpdir:
        state_path = os.path.join(tmpdir, "state.sqlite3")
        output_path = os.path.join(tmpdir, "dict.sqlite3")
        pure_path = os.path.join(tmpdir, "pure.sqlite3")
        words_path = os.path.join(tmpdir, "pure-words.txt")

        conn = connect_state(state_path)
        process_ingest_extract(conn, os.path.join(FIXTURES, "tiny_extract.jsonl"))

        process_export_current(conn, output_path, pure_db_path=pure_path, pure_words_path=words_path)

        # Verify full DB — one definition per word
        out = sqlite3.connect(output_path)
        word_count = out.execute("SELECT COUNT(*) FROM words").fetchone()[0]
        assert word_count == 3, f"Expected 3 words, got {word_count}"

        # Each word should have exactly one definition column (no separate definitions table)
        tables = [r[0] for r in out.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
        assert "definitions" not in tables, "Should not have a separate definitions table"

        # Verify each word has a definition
        for word in ["apple", "run", "blue"]:
            row = out.execute("SELECT definition FROM words WHERE word=?", (word,)).fetchone()
            assert row is not None, f"Missing word: {word}"
            assert row[0], f"Empty definition for: {word}"
        out.close()

        # Verify pure DB has same schema
        pure = sqlite3.connect(pure_path)
        pure_count = pure.execute("SELECT COUNT(*) FROM words").fetchone()[0]
        assert pure_count == 3  # all three are pure alpha
        pure.close()

        # Verify word list
        with open(words_path) as f:
            words = [l.strip() for l in f if l.strip()]
        assert "apple" in words
        assert "blue" in words
        assert "run" in words

        conn.close()
        print("PASS: export current (single definition per word)")


def test_enhance_pocket_definitions():
    """Enhancement should produce one pocket definition per word, reusing unchanged ones."""
    with tempfile.TemporaryDirectory() as tmpdir:
        state_path = os.path.join(tmpdir, "state.sqlite3")
        output_path = os.path.join(tmpdir, "dict.sqlite3")
        conn = connect_state(state_path)

        original_synthesizer = msd.synthesize_pocket_definition

        def fake_synthesizer(cfg, word, definitions, max_total_words, **kwargs):
            # Combine all definitions into a short pocket definition
            parts = []
            for idx, pos, text in definitions:
                parts.append(text.split()[0] if text.split() else text)
            return "; ".join(parts)

        try:
            msd.synthesize_pocket_definition = fake_synthesizer

            process_ingest_extract(conn, os.path.join(FIXTURES, "tiny_extract.jsonl"))
            process_plan_update(conn)

            # First enhance: all words need pocket definitions
            stats1 = process_enhance_changed(
                conn,
                LMStudioConfig(url="http://localhost:1234/v1/chat/completions", model="fake"),
                max_total_words=5,
            )
            # apple has 1 short def -> passthrough; run and blue have multiple -> llm
            pocket_count = conn.execute("SELECT COUNT(*) FROM enhanced_words").fetchone()[0]
            assert pocket_count == 3, f"Expected 3 pocket defs, got {pocket_count}"

            # Each word has exactly one pocket definition
            for word in ["apple", "run", "blue"]:
                rows = conn.execute(
                    "SELECT pocket_definition FROM enhanced_words WHERE word_key=?", (word,)
                ).fetchall()
                assert len(rows) == 1, f"Expected 1 pocket def for {word}, got {len(rows)}"
                assert rows[0][0], f"Empty pocket definition for {word}"

            # Second run on same extract: all should be reused (zero LLM work)
            stats2 = process_enhance_changed(
                conn,
                LMStudioConfig(url="http://localhost:1234/v1/chat/completions", model="fake"),
                max_total_words=5,
            )
            assert stats2["words_enhanced"] == 0, f"Expected 0 words enhanced on rerun, got {stats2['words_enhanced']}"
            assert stats2["words_reused"] == 3, f"Expected 3 reused, got {stats2['words_reused']}"

            # Now ingest changed extract and enhance again
            process_ingest_extract(conn, os.path.join(FIXTURES, "tiny_extract_changed.jsonl"))
            process_plan_update(conn)

            # Dry run should show words needing enhancement
            dry_stats = process_enhance_changed(
                conn,
                LMStudioConfig(url="http://localhost:1234/v1/chat/completions", model="fake"),
                max_total_words=5,
                dry_run=True,
            )
            # run: changed, green: new — apple, blue: unchanged (reused)
            assert dry_stats["words_to_enhance"] == 2, f"Expected 2 words to enhance, got {dry_stats}"
            assert dry_stats["words_reused"] == 2, f"Expected 2 reused, got {dry_stats}"

            before_count = conn.execute("SELECT COUNT(*) FROM enhanced_words").fetchone()[0]

            # Actual enhance
            stats3 = process_enhance_changed(
                conn,
                LMStudioConfig(url="http://localhost:1234/v1/chat/completions", model="fake"),
                max_total_words=5,
            )
            assert stats3["words_reused"] == 2

            # Export and verify single definition per word
            process_export_current(conn, output_path)
            out = sqlite3.connect(output_path)
            exported_words = out.execute("SELECT word, definition FROM words ORDER BY word").fetchall()
            out.close()

            word_dict = {w: d for w, d in exported_words}
            assert "apple" in word_dict
            assert "run" in word_dict
            assert "blue" in word_dict
            assert "green" in word_dict
            assert len(exported_words) == 4

            # Each word has exactly one definition
            for word, defn in exported_words:
                assert defn, f"Empty definition for {word}"

        finally:
            msd.synthesize_pocket_definition = original_synthesizer
            conn.close()

        print("PASS: enhance pocket definitions")


def test_build_report():
    with tempfile.TemporaryDirectory() as tmpdir:
        state_path = os.path.join(tmpdir, "state.sqlite3")
        report_path = os.path.join(tmpdir, "build-report.json")
        output_path = os.path.join(tmpdir, "dict.sqlite3")
        pure_path = os.path.join(tmpdir, "pure.sqlite3")

        conn = connect_state(state_path)
        process_ingest_extract(conn, os.path.join(FIXTURES, "tiny_extract.jsonl"))
        process_plan_update(conn)
        process_export_current(conn, output_path, pure_db_path=pure_path)

        report = process_build_report(
            conn,
            report_path,
            export_db_path=output_path,
            pure_db_path=pure_path,
        )
        assert os.path.exists(report_path)
        assert report["words"]["total"] == 3
        assert report["source_definitions"]["total"] == 5
        assert report["export"]["words"] == 3
        assert report["enhancement_version"] == "pocket_v1"
        conn.close()
        print("PASS: build report")


def test_pipeline_status_and_mark_published():
    with tempfile.TemporaryDirectory() as tmpdir:
        state_path = os.path.join(tmpdir, "stage1.sqlite3")
        conn = connect_state(state_path)
        process_ingest_extract(conn, os.path.join(FIXTURES, "tiny_extract.jsonl"))
        status = get_pipeline_status(conn)
        assert status["latest_extract"]["id"] == 1
        assert status["has_unpublished_extract"] is True

        published = mark_published(conn, "full", "extract-1")
        assert published["extract_id"] == 1

        status2 = get_pipeline_status(conn)
        assert status2["published"]["full"]["extract_id"] == 1
        assert status2["has_unpublished_extract"] is False
        conn.close()
        print("PASS: pipeline status and mark published")


if __name__ == "__main__":
    test_normalization()
    test_hash_stability()
    test_pure_alpha()
    test_enhancement_version()
    test_ingest_extract()
    test_plan_update_first_extract()
    test_plan_update_diff()
    test_enhance_pocket_definitions()
    test_export_current_single_definition()
    test_build_report()
    test_pipeline_status_and_mark_published()
    print("\nAll Stage-1 tests passed!")
