#!/usr/bin/env python3
"""Build a fast SQLite FTS5 full-text search index from data/processed/articles.jsonl."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sqlite3
import time
from pathlib import Path


def open_jsonl(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def build_db(input_path: Path, output_db: Path, batch_size: int = 10000) -> None:
    t0 = time.perf_counter()
    output_db.parent.mkdir(parents=True, exist_ok=True)
    final_db = output_db
    # Build into a temp file and atomically rename, so a live server never sees a half-built DB.
    output_db = final_db.with_name(final_db.name + ".tmp")
    if output_db.exists():
        output_db.unlink()

    conn = sqlite3.connect(str(output_db))
    conn.execute("PRAGMA journal_mode = OFF;")
    conn.execute("PRAGMA synchronous = OFF;")
    conn.execute("PRAGMA temp_store = MEMORY;")
    conn.execute("PRAGMA cache_size = -200000;")  # ~200MB cache

    conn.executescript(
        """
        CREATE TABLE articles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            article_id TEXT NOT NULL,
            king_code TEXT NOT NULL,
            king_name_ko TEXT NOT NULL,
            king_name_hanja TEXT NOT NULL,
            date_iso TEXT NOT NULL,
            date_ganji TEXT NOT NULL,
            date_reign TEXT NOT NULL,
            subjects TEXT NOT NULL,
            title_ko TEXT NOT NULL,
            body_hanja TEXT NOT NULL,
            postscripts TEXT NOT NULL,
            has_saron INTEGER NOT NULL DEFAULT 0
        );

        CREATE VIRTUAL TABLE articles_fts USING fts5(
            king_name_ko,
            date_reign,
            subjects,
            title_ko,
            body_hanja,
            postscripts,
            content='articles',
            content_rowid='id',
            tokenize='unicode61'
        );
        """
    )

    insert_sql = """
        INSERT INTO articles (
            article_id, king_code, king_name_ko, king_name_hanja,
            date_iso, date_ganji, date_reign, subjects,
            title_ko, body_hanja, postscripts, has_saron
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    batch = []
    total = 0
    saron_count = 0

    with open_jsonl(input_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            subjects_list = obj.get("subjects") or []
            postscripts_list = obj.get("postscripts") or []
            has_saron = 1 if len(postscripts_list) > 0 else 0
            if has_saron:
                saron_count += 1

            batch.append(
                (
                    obj.get("article_id", ""),
                    obj.get("king_code", ""),
                    obj.get("king_name_ko", ""),
                    obj.get("king_name_hanja", ""),
                    obj.get("date_iso", ""),
                    obj.get("date_ganji", ""),
                    obj.get("date_reign", ""),
                    " ".join(subjects_list),
                    obj.get("title_ko", ""),
                    obj.get("body_hanja", ""),
                    "\n".join(postscripts_list),
                    has_saron,
                )
            )
            total += 1
            if len(batch) >= batch_size:
                conn.executemany(insert_sql, batch)
                conn.commit()
                batch.clear()
                if total % 50000 == 0:
                    print(f"[index] Inserted {total:,} articles ({time.perf_counter() - t0:.1f}s)", flush=True)

    if batch:
        conn.executemany(insert_sql, batch)
        conn.commit()
        batch.clear()

    print(f"[index] Total {total:,} articles ({saron_count:,} with Saron). Building FTS5 & B-tree indexes...", flush=True)
    conn.execute("CREATE INDEX idx_articles_king_code ON articles(king_code);")
    conn.execute("CREATE INDEX idx_articles_king_name ON articles(king_name_ko);")
    conn.execute("CREATE INDEX idx_articles_has_saron ON articles(has_saron);")
    conn.execute("INSERT INTO articles_fts(articles_fts) VALUES('rebuild');")
    conn.commit()
    conn.execute("INSERT INTO articles_fts(articles_fts) VALUES('optimize');")
    conn.commit()
    conn.close()
    os.replace(output_db, final_db)

    elapsed = time.perf_counter() - t0
    size_mb = final_db.stat().st_size / (1024 * 1024)
    print(f"[index] Done! Built {final_db} ({size_mb:.1f} MB) in {elapsed:.1f}s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SQLite FTS5 search index for Sillok.")
    parser.add_argument("--input", type=Path, default=Path("data/processed/articles.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/sillok_search.db"))
    args = parser.parse_args()
    build_db(args.input, args.output)


if __name__ == "__main__":
    main()
