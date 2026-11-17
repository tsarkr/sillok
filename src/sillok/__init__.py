"""Sillok-LM: From-scratch language modeling toolkit for the Annals of the Joseon Dynasty."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from sillok.constants import (
    DATASET_INFO,
    DEFAULT_SPECIAL_TOKENS,
    KING_CODE_MAP,
    SPECIAL_TOKEN_IDS,
    SPECIAL_TOKENS,
)

__version__ = "0.1.0"

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "SillokLMConfig": ("sillok.model.config", "SillokLMConfig"),
    "SillokForCausalLM": ("sillok.model.transformer", "SillokForCausalLM"),
    "SillokArticle": ("sillok.data.xml_parser", "SillokArticle"),
}

__all__ = [
    "__version__",
    "DATASET_INFO",
    "DEFAULT_SPECIAL_TOKENS",
    "KING_CODE_MAP",
    "SPECIAL_TOKEN_IDS",
    "SPECIAL_TOKENS",
    "SillokArticle",
    "SillokLMConfig",
    "SillokForCausalLM",
]


def __getattr__(name: str) -> Any:
    if name in _LAZY_EXPORTS:
        module_name, attr_name = _LAZY_EXPORTS[name]
        module = import_module(module_name)
        return getattr(module, attr_name)
    raise AttributeError(f"module 'sillok' has no attribute {name!r}")
