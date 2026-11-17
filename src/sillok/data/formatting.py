"""Sequence formatting helpers for pretraining and conditional generation."""

from __future__ import annotations

import random

from sillok.constants import DEFAULT_SPECIAL_TOKENS
from sillok.data.xml_parser import SillokArticle


def _build_metadata_prefix(article: SillokArticle, tokens: dict[str, str]) -> list[str]:
    parts: list[str] = [tokens["bos"]]
    king_label = article.king_name_ko or article.king_code
    if article.king_name_hanja and article.king_name_hanja != king_label:
        king_label = f"{king_label}({article.king_name_hanja})"
    parts.append(f"{tokens['king']}{king_label}")

    date_bits = [bit for bit in (article.date_reign, article.date_iso, article.date_ganji) if bit]
    if date_bits:
        parts.append(f"{tokens['date']}{' | '.join(date_bits)}")

    if article.subjects:
        parts.append(f"{tokens['subject']}{', '.join(article.subjects)}")

    return parts


def format_article_for_lm(
    article: SillokArticle,
    *,
    reverse_title_prob: float = 0.2,
    rng: random.Random | None = None,
    include_postscript: bool = True,
    tokens: dict[str, str] = DEFAULT_SPECIAL_TOKENS,
) -> str:
    """Format a `SillokArticle` into a unified causal-LM training sequence.

    Forward format (default):
        <|bos|><|king|>...<|date|>...<|subject|>...<|title|>한글제목<|hanja|>한문원문[<|postscript|>사론]<|eos|>
    Reverse format (when sampled with `reverse_title_prob` and both title & body exist):
        <|bos|><|king|>...<|date|>...<|subject|>...<|hanja|>한문원문[<|postscript|>사론]<|title|>한글제목<|eos|>
    """
    parts = _build_metadata_prefix(article, tokens)
    postscript_block = ""
    if include_postscript and article.postscripts:
        postscript_block = f"{tokens['postscript']}" + "\n".join(article.postscripts)

    has_title = bool(article.title_ko)
    has_body = bool(article.body_hanja)

    use_reverse = False
    if has_title and has_body and reverse_title_prob > 0.0:
        rand_val = rng.random() if rng is not None else random.random()
        use_reverse = rand_val < reverse_title_prob

    if use_reverse:
        parts.append(f"{tokens['hanja']}{article.body_hanja}")
        if postscript_block:
            parts.append(postscript_block)
        parts.append(f"{tokens['title']}{article.title_ko}")
    else:
        if has_title:
            parts.append(f"{tokens['title']}{article.title_ko}")
        if has_body:
            parts.append(f"{tokens['hanja']}{article.body_hanja}")
        if postscript_block:
            parts.append(postscript_block)

    parts.append(tokens["eos"])
    return "".join(parts)


def format_generation_prompt(
    *,
    king: str = "",
    date: str = "",
    subject: str = "",
    title_ko: str = "",
    body_hanja: str = "",
    target_mode: str = "hanja",
    tokens: dict[str, str] = DEFAULT_SPECIAL_TOKENS,
) -> str:
    """Build a conditional prompt ending with the target section start token."""
    parts: list[str] = [tokens["bos"]]
    if king:
        parts.append(f"{tokens['king']}{king}")
    if date:
        parts.append(f"{tokens['date']}{date}")
    if subject:
        parts.append(f"{tokens['subject']}{subject}")

    if target_mode == "title":
        if body_hanja:
            parts.append(f"{tokens['hanja']}{body_hanja}")
        parts.append(tokens["title"])
    elif target_mode == "ko":
        if body_hanja:
            parts.append(f"{tokens['hanja']}{body_hanja}")
        parts.append(tokens["ko"])
    elif target_mode == "saron":
        if title_ko:
            parts.append(f"{tokens['title']}{title_ko}")
        if body_hanja:
            parts.append(f"{tokens['hanja']}{body_hanja}")
        parts.append(tokens["postscript"])
    else:
        if title_ko:
            parts.append(f"{tokens['title']}{title_ko}")
        parts.append(tokens["hanja"])

    return "".join(parts)
