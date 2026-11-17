#!/usr/bin/env python3
"""Train a custom Byte-Level BPE tokenizer on parsed Sillok JSONL."""

from __future__ import annotations

import argparse
from pathlib import Path

from sillok.tokenizer.trainer import load_sillok_tokenizer, train_sillok_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Sillok-LM BPE tokenizer.")
    parser.add_argument("--jsonl", type=Path, default=Path("data/processed/articles.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/tokenizer"))
    parser.add_argument("--vocab-size", type=int, default=32768)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--reverse-title-prob", type=float, default=0.2)
    parser.add_argument("--max-articles", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    tok_path = train_sillok_tokenizer(
        args.jsonl,
        args.output_dir,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        reverse_title_prob=args.reverse_title_prob,
        max_articles=args.max_articles,
        seed=args.seed,
    )
    tok = load_sillok_tokenizer(tok_path)

    samples = [
        "<|bos|><|king|>태조실록(太祖實錄)<|title|>태조가 수창궁에서 왕위에 오르다<|hanja|>太祖卽位于壽昌宮。<|eos|>",
        "上曰: 卿等宜各盡心力, 以輔予不逮。",
    ]
    print(f"Tokenizer saved to {tok_path} (vocab_size={tok.get_vocab_size()})")
    for sample in samples:
        enc = tok.encode(sample)
        chars = max(1, len(sample))
        print(f"Sample len={chars} chars -> {len(enc.ids)} tokens (ratio={chars/len(enc.ids):.2f} chars/tok)")


if __name__ == "__main__":
    main()
