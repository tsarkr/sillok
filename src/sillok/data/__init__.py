"""Data parsing, formatting, and memory-mapped dataset utilities for Sillok-LM."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from sillok.data.formatting import format_article_for_lm, format_generation_prompt
from sillok.data.xml_parser import (
    SillokArticle,
    iter_xml_files,
    parse_sillok_corpus,
    parse_sillok_xml,
    sanitize_xml_surrogates,
)

__all__ = [
    "PackedMemmapDataset",
    "SillokArticle",
    "format_article_for_lm",
    "format_generation_prompt",
    "iter_xml_files",
    "parse_sillok_corpus",
    "parse_sillok_xml",
    "sanitize_xml_surrogates",
]


def __getattr__(name: str) -> Any:
    if name == "PackedMemmapDataset":
        module = import_module("sillok.data.dataset")
        return getattr(module, name)
    raise AttributeError(f"module 'sillok.data' has no attribute {name!r}")
