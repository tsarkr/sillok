"""Model configuration and parameter count estimator for Sillok-LM."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class SillokLMConfig:
    """Configuration for Llama-style Decoder-only Transformer."""

    vocab_size: int = 32768
    hidden_size: int = 1024
    intermediate_size: int = 3072
    num_hidden_layers: int = 26
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    max_seq_len: int = 2048
    rms_norm_eps: float = 1e-5
    rope_theta: float = 500000.0
    tie_word_embeddings: bool = True
    dropout: float = 0.0
    use_gradient_checkpointing: bool = False
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_attention_heads ({self.num_attention_heads})."
            )
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) must be divisible by "
                f"num_key_value_heads ({self.num_key_value_heads})."
            )
        if self.head_dim % 2 != 0:
            raise ValueError(f"head_dim ({self.head_dim}) must be even for RoPE.")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def kv_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    def estimate_num_parameters(self) -> int:
        """Return exact analytical parameter count of `SillokForCausalLM`."""
        embed_params = self.vocab_size * self.hidden_size
        lm_head_params = 0 if self.tie_word_embeddings else embed_params

        kv_dim = self.num_key_value_heads * self.head_dim
        attn_params = (
            self.hidden_size * self.hidden_size  # q_proj
            + self.hidden_size * kv_dim  # k_proj
            + self.hidden_size * kv_dim  # v_proj
            + self.hidden_size * self.hidden_size  # o_proj
        )
        mlp_params = 3 * self.hidden_size * self.intermediate_size  # gate, up, down
        norm_params_per_layer = 2 * self.hidden_size
        layer_params = attn_params + mlp_params + norm_params_per_layer

        final_norm_params = self.hidden_size
        return (
            embed_params
            + lm_head_params
            + self.num_hidden_layers * layer_params
            + final_norm_params
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SillokLMConfig":
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "SillokLMConfig":
        payload = yaml.safe_load(Path(yaml_path).read_text(encoding="utf-8"))
        model_dict = payload.get("model", payload)
        return cls.from_dict(model_dict)
