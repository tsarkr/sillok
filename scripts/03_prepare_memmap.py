#!/usr/bin/env python3
"""Stream-encode parsed Sillok JSONL into packed uint16/uint32 train.bin and val.bin files.

Uses deterministic hash-based article splitting and Rust `encode_batch` so peak RAM stays
under ~50 MB even on 3.7 GiB hosts when processing Sillok articles.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from sillok.data.formatting import format_article_for_lm
from sillok.data.xml_parser import SillokArticle


def is_validation_article(article_id: str, val_ratio: float, seed: int = 42) -> bool:
    """Deterministic, zero-memory split assignment keyed by `article_id` and `seed`."""
    if val_ratio <= 0.0:
        return False
    if val_ratio >= 1.0:
        return True
    digest = hashlib.blake2b(f"{seed}:{article_id}".encode("utf-8"), digest_size=8).digest()
    bucket = int.from_bytes(digest, "big") % 100_000
    return bucket < int(round(val_ratio * 100_000))


def main() -> None:
    import numpy as np
    from sillok.tokenizer.trainer import load_sillok_tokenizer

    parser = argparse.ArgumentParser(description="Prepare packed train.bin / val.bin memmap files.")
    parser.add_argument("--jsonl", type=Path, default=Path("data/processed/articles.jsonl"))
    parser.add_argument("--tokenizer", type=Path, default=Path("artifacts/tokenizer/tokenizer.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--val-ratio", type=float, default=0.02, help="Fraction of articles held out for validation")
    parser.add_argument("--reverse-title-prob", type=float, default=0.2)
    parser.add_argument("--chunk-tokens", type=int, default=1_000_000)
    parser.add_argument("--max-articles", type=int, default=None, help="Optional cap (strided across all kings)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    tokenizer = load_sillok_tokenizer(args.tokenizer)
    vocab_size = tokenizer.get_vocab_size()
    dtype_name = "uint16" if vocab_size <= 65535 else "uint32"
    np_dtype = np.dtype(dtype_name)

    stride = 1
    if args.max_articles is not None and args.max_articles > 0:
        with args.jsonl.open("rb") as fbin:
            total_lines = sum(1 for _ in fbin)
        stride = max(1, total_lines // args.max_articles)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_bin = args.output_dir / "train.bin"
    val_bin = args.output_dir / "val.bin"

    rng = random.Random(args.seed)
    train_buf: list[int] = []
    val_buf: list[int] = []
    train_articles = 0
    val_articles = 0
    train_tokens = 0
    val_tokens = 0

    def _flush(buf: list[int], fout) -> int:
        if not buf:
            return 0
        np.asarray(buf, dtype=np_dtype).tofile(fout)
        count = len(buf)
        buf.clear()
        return count

    batch_texts: list[str] = []
    batch_is_val: list[bool] = []

    def _process_batch(f_train, f_val) -> None:
        nonlocal train_articles, val_articles, train_tokens, val_tokens
        if not batch_texts:
            return
        encodings = tokenizer.encode_batch(batch_texts)
        for enc, is_val in zip(encodings, batch_is_val):
            if is_val:
                val_articles += 1
                val_buf.extend(enc.ids)
                if len(val_buf) >= args.chunk_tokens:
                    val_tokens += _flush(val_buf, f_val)
            else:
                train_articles += 1
                train_buf.extend(enc.ids)
                if len(train_buf) >= args.chunk_tokens:
                    train_tokens += _flush(train_buf, f_train)
        batch_texts.clear()
        batch_is_val.clear()

    processed = 0
    with (
        args.jsonl.open("r", encoding="utf-8") as fin,
        train_bin.open("wb") as f_train,
        val_bin.open("wb") as f_val,
    ):
        for idx, line in enumerate(fin):
            if idx % stride != 0:
                continue
            if args.max_articles is not None and processed >= args.max_articles:
                break
            line = line.strip()
            if not line:
                continue
            processed += 1
            art = SillokArticle.from_dict(json.loads(line))
            is_val = is_validation_article(art.article_id or str(idx), args.val_ratio, args.seed)
            text = format_article_for_lm(
                art,
                reverse_title_prob=0.0 if is_val else args.reverse_title_prob,
                rng=rng,
            )
            batch_texts.append(text)
            batch_is_val.append(is_val)
            if len(batch_texts) >= 512:
                _process_batch(f_train, f_val)

        _process_batch(f_train, f_val)
        train_tokens += _flush(train_buf, f_train)
        val_tokens += _flush(val_buf, f_val)

    meta = {
        "vocab_size": vocab_size,
        "dtype": dtype_name,
        "total_articles": train_articles + val_articles,
        "train_articles": train_articles,
        "val_articles": val_articles,
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "split_method": "blake2b_streaming",
        "stride": stride,
        "seed": args.seed,
    }
    (args.output_dir / "dataset_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"Packed {train_articles:,} train articles ({train_tokens:,} tokens) -> {train_bin}\n"
        f"Packed {val_articles:,} val articles ({val_tokens:,} tokens) -> {val_bin}"
    )


if __name__ == "__main__":
    main()
