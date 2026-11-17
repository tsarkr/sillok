"""Training and inference helpers for Sillok-LM."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["SillokPretrainer", "TrainingConfig", "generate_from_checkpoint"]


def __getattr__(name: str) -> Any:
    if name in ("SillokPretrainer", "TrainingConfig"):
        module = import_module("sillok.training.trainer")
        return getattr(module, name)
    if name == "generate_from_checkpoint":
        module = import_module("sillok.training.generate")
        return getattr(module, name)
    raise AttributeError(f"module 'sillok.training' has no attribute {name!r}")
