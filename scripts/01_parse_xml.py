#!/usr/bin/env python3
"""Parse raw NHIC Sillok XML files in `data/` into structured JSONL articles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sillok.data.xml_parser import parse_sillok_corpus


def _make_progress_printer() -> callable:
    try:
        from tqdm import tqdm  # type: ignore

        bar: list[tqdm] = []

        def _cb(done: int, total: int, fname: str) -> None:
            if not bar:
                bar.append(tqdm(total=total, desc="Parsing XML", unit="file"))
            bar[0].set_postfix_str(fname, refresh=False)
            bar[0].update(1)
            if done >= total:
                bar[0].close()

        return _cb
    except ImportError:

        def _cb(done: int, total: int, fname: str) -> None:
            if done == 1 or done % 50 == 0 or done == total:
                print(f"[{done}/{total}] parsed {fname}")

        return _cb


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse Sillok XML files into JSONL.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Directory containing 2nd_*.xml files")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"), help="Output directory")
    parser.add_argument("--pattern", type=str, default="2nd_*.xml", help="Glob pattern for XML files")
    parser.add_argument("--limit-files", type=int, default=None, help="Limit number of XML files for testing")
    parser.add_argument("--workers", type=int, default=None, help="Number of worker processes")
    parser.add_argument("--no-annotations", action="store_true", help="Exclude inline annotations (원주 등)")
    parser.add_argument("--include-collation", action="store_true", help="Include 교감주 notes")
    parser.add_argument("--extract-entities", action="store_true", help="Store <index> named entities in JSONL")
    args = parser.parse_args()

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl = out_dir / "articles.jsonl"

    summary = parse_sillok_corpus(
        args.data_dir,
        output_jsonl,
        pattern=args.pattern,
        limit_files=args.limit_files,
        workers=args.workers,
        include_annotations=not args.no_annotations,
        include_collation_notes=args.include_collation,
        extract_entities=args.extract_entities,
        progress_callback=_make_progress_printer(),
    )

    summary_path = out_dir / "parse_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"Done: {summary['files_processed']} files -> "
        f"{summary['articles_extracted']:,} articles saved to {output_jsonl}"
    )


if __name__ == "__main__":
    main()
