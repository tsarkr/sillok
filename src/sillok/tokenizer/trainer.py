"""Train and inspect a Byte-Level BPE tokenizer tailored for Classical Hanja + Korean."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterator

from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, processors, trainers

from sillok.constants import SPECIAL_TOKENS
from sillok.data.formatting import format_article_for_lm
from sillok.data.xml_parser import SillokArticle


def _iter_formatted_texts(
    jsonl_path: str | Path,
    *,
    reverse_title_prob: float = 0.2,
    max_articles: int | None = None,
    seed: int = 42,
) -> Iterator[str]:
    path = Path(jsonl_path)
    stride = 1
    if max_articles is not None and max_articles > 0:
        with path.open("rb") as fbin:
            total_lines = sum(1 for _ in fbin)
        stride = max(1, total_lines // max_articles)

    rng = random.Random(seed)
    yielded = 0
    with path.open("r", encoding="utf-8") as fin:
        for idx, line in enumerate(fin):
            if idx % stride != 0:
                continue
            if max_articles is not None and yielded >= max_articles:
                break
            line = line.strip()
            if not line:
                continue
            article = SillokArticle.from_dict(json.loads(line))
            yielded += 1
            yield format_article_for_lm(
                article,
                reverse_title_prob=reverse_title_prob,
                rng=rng,
            )


def train_sillok_tokenizer(
    jsonl_path: str | Path,
    output_dir: str | Path,
    *,
    vocab_size: int = 32768,
    min_frequency: int = 2,
    reverse_title_prob: float = 0.2,
    max_articles: int | None = None,
    seed: int = 42,
) -> Path:
    """Train a Byte-Level BPE tokenizer on parsed Sillok JSONL records."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))
    tokenizer.normalizer = normalizers.NFC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )

    iterator = _iter_formatted_texts(
        jsonl_path,
        reverse_title_prob=reverse_title_prob,
        max_articles=max_articles,
        seed=seed,
    )
    tokenizer.train_from_iterator(iterator, trainer=trainer)

    tokenizer_file = out_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_file))

    meta: dict[str, Any] = {
        "vocab_size": tokenizer.get_vocab_size(),
        "special_tokens": {tok: tokenizer.token_to_id(tok) for tok in SPECIAL_TOKENS},
        "min_frequency": min_frequency,
    }
    (out_dir / "tokenizer_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return tokenizer_file


def load_sillok_tokenizer(tokenizer_path: str | Path) -> Tokenizer:
    """Load a trained HuggingFace `tokenizers.Tokenizer` from file or directory."""
    path = Path(tokenizer_path)
    if path.is_dir():
        path = path / "tokenizer.json"
    if not path.exists():
        raise FileNotFoundError(f"Tokenizer file not found: {path}")
    return Tokenizer.from_file(str(path))
