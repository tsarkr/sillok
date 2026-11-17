"""Model configuration and Llama-style transformer architecture for Sillok-LM."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from sillok.model.config import SillokLMConfig

__all__ = ["SillokLMConfig", "SillokForCausalLM"]


def __getattr__(name: str) -> Any:
    if name == "SillokForCausalLM":
        module = import_module("sillok.model.transformer")
        return getattr(module, name)
    raise AttributeError(f"module 'sillok.model' has no attribute {name!r}")
