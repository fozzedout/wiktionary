#!/usr/bin/env python3
"""Dump pure_dictionary.sqlite3 to SQL for D1 import.

Two modes:
  Full:  exports all rows (first deploy or forced via --full)
  Delta: compares against a prior snapshot and emits only changes

Outputs:
  _words.sql       — word inserts/updates/deletes
  _definitions.sql — definition inserts/updates/deletes
  _deploy_meta.json — counts of operations
"""
import json
import sqlite3
import sys
import os


def quote(v):
    if v is None:
        return "NULL"
    s = str(v).replace("\r", " ").replace("\n", " ").replace("'", "''")
    return f"'{s}'"


def dump_full(db_path, out_dir, words_table, defs_table):
    """Full export — all rows as INSERTs."""
    conn = sqlite3.connect(db_path)

    words_file = os.path.join(out_dir, "_words.sql")
    with open(words_file, "w") as f:
        count = 0
        for row in conn.execute("SELECT id, word FROM words ORDER BY word"):
            f.write(f"INSERT OR REPLACE INTO {words_table} VALUES ({row[0]}, {quote(row[1])});\n")
            count += 1
    print(f"Full export: {count} words to {words_file}")

    defs_file = os.path.join(out_dir, "_definitions.sql")
    with open(defs_file, "w") as f:
        count_d = 0
        for row in conn.execute("SELECT id, definition FROM definitions ORDER BY id"):
            f.write(f"INSERT OR REPLACE INTO {defs_table} VALUES ({row[0]}, {quote(row[1])});\n")
            count_d += 1
    print(f"Full export: {count_d} definitions to {defs_file}")

    conn.close()
    return {"mode": "full", "words_inserted": count, "definitions_inserted": count_d}


def dump_delta(new_db_path, prior_db_path, out_dir, words_table, defs_table):
    """Delta export — only changes between prior and new."""
    new_conn = sqlite3.connect(new_db_path)
    new_conn.execute(f"ATTACH DATABASE ? AS prior", (prior_db_path,))

    words_file = os.path.join(out_dir, "_words.sql")
    defs_file = os.path.join(out_dir, "_definitions.sql")

    stats = {
        "mode": "delta",
        "words_inserted": 0,
        "words_updated": 0,
        "words_deleted": 0,
        "definitions_inserted": 0,
        "definitions_updated": 0,
        "definitions_deleted": 0,
    }

    with open(defs_file, "w") as f:
        # New definitions (id exists in new but not prior)
        for row in new_conn.execute(
            "SELECT n.id, n.definition FROM definitions n "
            "LEFT JOIN prior.definitions p ON p.id = n.id "
            "WHERE p.id IS NULL"
        ):
            f.write(f"INSERT INTO {defs_table} VALUES ({row[0]}, {quote(row[1])});\n")
            stats["definitions_inserted"] += 1

        # Changed definitions (id exists in both but definition differs)
        for row in new_conn.execute(
            "SELECT n.id, n.definition FROM definitions n "
            "JOIN prior.definitions p ON p.id = n.id "
            "WHERE n.definition != p.definition"
        ):
            f.write(f"UPDATE {defs_table} SET definition={quote(row[1])} WHERE id={row[0]};\n")
            stats["definitions_updated"] += 1

        # Deleted definitions (id in prior but not new)
        for row in new_conn.execute(
            "SELECT p.id FROM prior.definitions p "
            "LEFT JOIN definitions n ON n.id = p.id "
            "WHERE n.id IS NULL"
        ):
            f.write(f"DELETE FROM {defs_table} WHERE id={row[0]};\n")
            stats["definitions_deleted"] += 1

    with open(words_file, "w") as f:
        # New words (word in new but not prior)
        for row in new_conn.execute(
            "SELECT n.id, n.word FROM words n "
            "LEFT JOIN prior.words p ON p.word = n.word "
            "WHERE p.word IS NULL"
        ):
            f.write(f"INSERT INTO {words_table} VALUES ({row[0]}, {quote(row[1])});\n")
            stats["words_inserted"] += 1

        # Changed words (word exists but id changed — different base now)
        for row in new_conn.execute(
            "SELECT n.id, n.word FROM words n "
            "JOIN prior.words p ON p.word = n.word "
            "WHERE n.id != p.id"
        ):
            f.write(f"UPDATE {words_table} SET id={row[0]} WHERE word={quote(row[1])};\n")
            stats["words_updated"] += 1

        # Deleted words (word in prior but not new)
        for row in new_conn.execute(
            "SELECT p.word FROM prior.words p "
            "LEFT JOIN words n ON n.word = p.word "
            "WHERE n.word IS NULL"
        ):
            f.write(f"DELETE FROM {words_table} WHERE word={quote(row[0])};\n")
            stats["words_deleted"] += 1

    new_conn.close()

    total_ops = sum(stats[k] for k in stats if k != "mode")
    print(f"Delta export: {total_ops} operations")
    for k, v in stats.items():
        if k != "mode" and v > 0:
            print(f"  {k}: {v}")

    return stats


def main():
    db_path = sys.argv[1] if len(sys.argv) > 1 else "out/pure_dictionary.sqlite3"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "out"
    words_table = sys.argv[3] if len(sys.argv) > 3 else "words"
    defs_table = sys.argv[4] if len(sys.argv) > 4 else "definitions"
    prior_db = sys.argv[5] if len(sys.argv) > 5 else None

    if prior_db and os.path.exists(prior_db):
        stats = dump_delta(db_path, prior_db, out_dir, words_table, defs_table)
    else:
        if prior_db:
            print(f"Prior DB not found at {prior_db} — doing full export")
        stats = dump_full(db_path, out_dir, words_table, defs_table)

    meta_file = os.path.join(out_dir, "_deploy_meta.json")
    with open(meta_file, "w") as f:
        json.dump(stats, f, indent=2)


if __name__ == "__main__":
    main()
