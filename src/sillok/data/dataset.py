"""Zero-copy memory-mapped token dataset for high-throughput pretraining."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class PackedMemmapDataset(Dataset):
    """Slices a flat uint16/uint32 token binary stream into fixed `max_seq_len` chunks."""

    def __init__(
        self,
        bin_path: str | Path,
        max_seq_len: int,
        *,
        dtype: str = "uint16",
        stride: int | None = None,
    ) -> None:
        self.bin_path = Path(bin_path)
        if not self.bin_path.exists():
            raise FileNotFoundError(f"Token binary file not found: {self.bin_path}")
        if max_seq_len < 2:
            raise ValueError("max_seq_len must be >= 2")

        self.max_seq_len = max_seq_len
        self.np_dtype = np.dtype(dtype)
        self.stride = stride if stride is not None else max_seq_len

        self.tokens = np.memmap(self.bin_path, dtype=self.np_dtype, mode="r")
        self.total_tokens = int(len(self.tokens))
        # Each sample consumes `max_seq_len + 1` tokens so input=[:-1] and label=[1:]
        # or `max_seq_len` tokens with internal CausalLM shift. Using `max_seq_len`
        # directly allows exact block alignment and standard CausalLM shift.
        if self.total_tokens < self.max_seq_len:
            raise ValueError(
                f"Binary file {self.bin_path} has {self.total_tokens} tokens, "
                f"fewer than max_seq_len={self.max_seq_len}."
            )
        self.num_samples = (self.total_tokens - self.max_seq_len) // self.stride + 1

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.stride
        end = start + self.max_seq_len
        chunk = np.asarray(self.tokens[start:end], dtype=np.int64)
        tensor = torch.from_numpy(chunk)
        return tensor, tensor.clone()
