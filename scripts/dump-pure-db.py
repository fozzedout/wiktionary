#!/usr/bin/env python3
"""Dump pure_dictionary.sqlite3 to SQL INSERT statements for D1 import."""
import sqlite3
import sys
import os

def quote(v):
    if v is None:
        return "NULL"
    s = str(v).replace("'", "''")
    return f"'{s}'"

def main():
    db_path = sys.argv[1] if len(sys.argv) > 1 else "out/pure_dictionary.sqlite3"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "out"

    conn = sqlite3.connect(db_path)

    # Dump words
    words_file = os.path.join(out_dir, "_words.sql")
    with open(words_file, "w") as f:
        cur = conn.execute("SELECT word, raw FROM words ORDER BY word")
        count = 0
        for row in cur:
            f.write(f"INSERT OR IGNORE INTO words VALUES ({quote(row[0])}, {quote(row[1])});\n")
            count += 1
    print(f"Exported {count} words to {words_file}")

    # Dump definitions
    defs_file = os.path.join(out_dir, "_definitions.sql")
    with open(defs_file, "w") as f:
        cur = conn.execute("SELECT word, idx, pos, definition FROM definitions ORDER BY word, idx")
        count = 0
        for row in cur:
            f.write(f"INSERT OR IGNORE INTO definitions VALUES ({quote(row[0])}, {row[1]}, {quote(row[2])}, {quote(row[3])});\n")
            count += 1
    print(f"Exported {count} definitions to {defs_file}")

    conn.close()

if __name__ == "__main__":
    main()
