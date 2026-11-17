#!/usr/bin/env python3
"""Generate Classical Hanja articles, Korean titles, or Saron commentary from Sillok-LM."""

from __future__ import annotations

import argparse
from pathlib import Path

from sillok.training.generate import generate_from_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Conditional generation with Sillok-LM.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/sillok_100m_t4/infer.pt"),
        help="Path to checkpoint (.pt)",
    )
    parser.add_argument("--tokenizer", type=Path, default=Path("artifacts/tokenizer/tokenizer.json"))
    parser.add_argument("--king", type=str, default="태조실록(太祖實錄)")
    parser.add_argument("--date", type=str, default="태조 1년 7월 17일 | 1392-07-17L0 | 丙申")
    parser.add_argument("--subject", type=str, default="왕실-국왕/임면")
    parser.add_argument("--title-ko", type=str, default="")
    parser.add_argument("--body-hanja", type=str, default="")
    parser.add_argument("--mode", choices=["hanja", "title", "saron", "ko"], default="hanja")
    parser.add_argument("--max-new-tokens", type=int, default=150)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.92)
    parser.add_argument("--repetition-penalty", type=float, default=1.15)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    result = generate_from_checkpoint(
        args.checkpoint,
        args.tokenizer,
        king=args.king,
        date=args.date,
        subject=args.subject,
        title_ko=args.title_ko,
        body_hanja=args.body_hanja,
        target_mode=args.mode,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        device=args.device,
    )
    print("=== Prompt ===")
    print(result["prompt"])
    print("\n=== Generated ===")
    print(result["generated"])
    print(
        f"\n[telemetry] step={result['global_step']} | device={result['device']} | "
        f"tokens={result['generated_tokens']} | time={result['elapsed_sec']}s ({result['tokens_per_sec']} tok/s)"
    )


if __name__ == "__main__":
    main()
