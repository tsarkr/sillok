"""Custom BPE tokenizer training and loading utilities for Sillok-LM."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["load_sillok_tokenizer", "train_sillok_tokenizer"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        module = import_module("sillok.tokenizer.trainer")
        return getattr(module, name)
    raise AttributeError(f"module 'sillok.tokenizer' has no attribute {name!r}")
