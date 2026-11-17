"""From-scratch Llama-style Decoder-only Transformer for Sillok-LM."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from sillok.model.config import SillokLMConfig

# Disable oneDNN/MKL-DNN AVX2 dispatch on non-AVX2 CPUs (e.g., Intel Celeron J3160)
try:
    if Path("/proc/cpuinfo").exists():
        _cpu_flags = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="ignore")
        if "avx2" not in _cpu_flags and hasattr(torch.backends, "mkldnn"):
            torch.backends.mkldnn.enabled = False
except Exception:
    pass


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization computed in float32 for stability."""

    def __init__(self, hidden_size: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_fp32 = x.float()
        rms = torch.rsqrt(x_fp32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        out = (x_fp32 * rms).to(orig_dtype)
        return out * self.weight


class RotaryEmbedding(nn.Module):
    """Precomputed Rotary Position Embedding (RoPE) cache."""

    def __init__(self, head_dim: int, max_seq_len: int, theta: float = 500000.0) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.theta = theta
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len_cached = 0
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
        self.max_seq_len_cached = seq_len

    def forward(self, seq_len: int, dtype: torch.dtype, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.max_seq_len_cached or self.cos_cached.device != device:
            if self.inv_freq.device != device:
                self.inv_freq = self.inv_freq.to(device)
            self._build_cache(max(seq_len, self.max_seq_len_cached))
        return (
            self.cos_cached[:seq_len].to(dtype=dtype),
            self.sin_cached[:seq_len].to(dtype=dtype),
        )


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos_b = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim)
    sin_b = sin.unsqueeze(0).unsqueeze(0)
    q_rot = (q * cos_b) + (_rotate_half(q) * sin_b)
    k_rot = (k * cos_b) + (_rotate_half(k) * sin_b)
    return q_rot, k_rot


def _repeat_kv(hidden_states: torch.Tensor, num_repeats: int) -> torch.Tensor:
    if num_repeats == 1:
        return hidden_states
    bsz, n_kv_heads, seq_len, head_dim = hidden_states.shape
    expanded = hidden_states[:, :, None, :, :].expand(
        bsz, n_kv_heads, num_repeats, seq_len, head_dim
    )
    return expanded.reshape(bsz, n_kv_heads * num_repeats, seq_len, head_dim)


class SillokAttention(nn.Module):
    """Grouped Query Attention using PyTorch SDPA (FlashAttention on A100)."""

    def __init__(self, config: SillokLMConfig) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.kv_groups = config.kv_groups
        self.dropout = config.dropout

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        k = _repeat_kv(k, self.kv_groups)
        v = _repeat_kv(v, self.kv_groups)

        attn_dropout = self.dropout if self.training else 0.0
        if q.device.type == "cpu" and hasattr(torch.nn, "attention"):
            with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
                attn_out = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=None,
                    dropout_p=attn_dropout,
                    is_causal=True,
                )
        else:
            attn_out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=attn_dropout,
                is_causal=True,
            )
        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, seq_len, self.num_heads * self.head_dim)
        return self.o_proj(attn_out)


class SillokMLP(nn.Module):
    """SwiGLU Feed-Forward Network."""

    def __init__(self, config: SillokLMConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class SillokDecoderLayer(nn.Module):
    """Pre-norm Transformer Decoder block."""

    def __init__(self, config: SillokLMConfig) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = SillokAttention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = SillokMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.self_attn(
            self.input_layernorm(hidden_states), cos, sin
        )
        hidden_states = hidden_states + self.mlp(
            self.post_attention_layernorm(hidden_states)
        )
        return hidden_states


class SillokForCausalLM(nn.Module):
    """From-scratch Causal Language Model for the Annals of the Joseon Dynasty."""

    def __init__(self, config: SillokLMConfig, *, init_weights: bool = True) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.rotary_emb = RotaryEmbedding(
            head_dim=config.head_dim,
            max_seq_len=config.max_seq_len,
            theta=config.rope_theta,
        )
        self.layers = nn.ModuleList(
            [SillokDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        if init_weights:
            self.apply(self._init_weights)
            self._rescale_residual_projections()

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _rescale_residual_projections(self) -> None:
        residual_std = 0.02 / math.sqrt(2 * self.config.num_hidden_layers)
        for layer in self.layers:
            nn.init.normal_(layer.self_attn.o_proj.weight, mean=0.0, std=residual_std)
            nn.init.normal_(layer.mlp.down_proj.weight, mean=0.0, std=residual_std)

    def num_parameters(self, trainable_only: bool = False) -> int:
        seen_ptrs: set[int] = set()
        total = 0
        for param in self.parameters():
            if trainable_only and not param.requires_grad:
                continue
            ptr = param.data_ptr()
            if ptr in seen_ptrs:
                continue
            seen_ptrs.add(ptr)
            total += param.numel()
        return total

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        _, seq_len = input_ids.shape
        hidden_states = self.embed_tokens(input_ids)
        cos, sin = self.rotary_emb(seq_len, hidden_states.dtype, hidden_states.device)

        use_ckpt = self.config.use_gradient_checkpointing and self.training
        for layer in self.layers:
            if use_ckpt:
                hidden_states = checkpoint(
                    layer,
                    hidden_states,
                    cos,
                    sin,
                    use_reentrant=False,
                )
            else:
                hidden_states = layer(hidden_states, cos, sin)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        loss: Optional[torch.Tensor] = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return logits, loss

    def _forward_with_cache(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None,
        start_pos: int = 0,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        bsz, seq_len = input_ids.shape
        hidden_states = self.embed_tokens(input_ids)
        total_len = start_pos + seq_len
        cos_all, sin_all = self.rotary_emb(total_len, hidden_states.dtype, hidden_states.device)
        cos = cos_all[start_pos:total_len]
        sin = sin_all[start_pos:total_len]

        new_past: list[tuple[torch.Tensor, torch.Tensor]] = []
        for idx, layer in enumerate(self.layers):
            normed = layer.input_layernorm(hidden_states)
            attn = layer.self_attn
            q = attn.q_proj(normed).view(bsz, seq_len, attn.num_heads, attn.head_dim).transpose(1, 2)
            k = attn.k_proj(normed).view(bsz, seq_len, attn.num_kv_heads, attn.head_dim).transpose(1, 2)
            v = attn.v_proj(normed).view(bsz, seq_len, attn.num_kv_heads, attn.head_dim).transpose(1, 2)

            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            if past_key_values is not None:
                pk, pv = past_key_values[idx]
                k = torch.cat((pk, k), dim=2)
                v = torch.cat((pv, v), dim=2)
            new_past.append((k, v))

            k_rep = _repeat_kv(k, attn.kv_groups)
            v_rep = _repeat_kv(v, attn.kv_groups)
            is_causal = seq_len > 1

            if q.device.type == "cpu" and hasattr(torch.nn, "attention"):
                with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
                    attn_out = F.scaled_dot_product_attention(
                        q, k_rep, v_rep, attn_mask=None, dropout_p=0.0, is_causal=is_causal
                    )
            else:
                attn_out = F.scaled_dot_product_attention(
                    q, k_rep, v_rep, attn_mask=None, dropout_p=0.0, is_causal=is_causal
                )
            attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, seq_len, attn.num_heads * attn.head_dim)
            hidden_states = hidden_states + attn.o_proj(attn_out)
            hidden_states = hidden_states + layer.mlp(layer.post_attention_layernorm(hidden_states))

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states[:, -1:, :])
        return logits, new_past

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        repetition_penalty: float = 1.0,
        eos_token_id: Optional[int] = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """Autoregressive sampling with KV-cache and optional repetition penalty."""
        was_training = self.training
        self.eval()
        eos_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id

        generated = input_ids[:, -self.config.max_seq_len :]
        past_kv: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None
        start_pos = 0

        for step_idx in range(max_new_tokens):
            if use_cache and (start_pos + (1 if past_kv is not None else generated.size(1))) <= self.config.max_seq_len:
                if past_kv is None:
                    cur_in = generated
                    start_pos = 0
                else:
                    cur_in = generated[:, -1:]
                logits, past_kv = self._forward_with_cache(cur_in, past_key_values=past_kv, start_pos=start_pos)
                start_pos += cur_in.size(1)
                next_logits = logits[:, -1, :].float()
            else:
                context = generated[:, -self.config.max_seq_len :]
                logits, _ = self.forward(context)
                next_logits = logits[:, -1, :].float()

            if repetition_penalty != 1.0 and repetition_penalty > 0.0:
                for b_idx in range(generated.size(0)):
                    seen_tokens = torch.unique(generated[b_idx])
                    score = next_logits[b_idx, seen_tokens]
                    next_logits[b_idx, seen_tokens] = torch.where(
                        score < 0, score * repetition_penalty, score / repetition_penalty
                    )

            if temperature <= 0.0:
                next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
            else:
                next_logits = next_logits / temperature
                if top_k > 0:
                    k = min(top_k, next_logits.size(-1))
                    kth_vals = torch.topk(next_logits, k, dim=-1).values[:, -1:]
                    next_logits = next_logits.masked_fill(next_logits < kth_vals, float("-inf"))

                if 0.0 < top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_logits, descending=True, dim=-1)
                    probs = F.softmax(sorted_logits, dim=-1)
                    cum_probs = torch.cumsum(probs, dim=-1)
                    remove_mask = cum_probs > top_p
                    remove_mask[:, 1:] = remove_mask[:, :-1].clone()
                    remove_mask[:, 0] = False
                    sorted_logits = sorted_logits.masked_fill(remove_mask, float("-inf"))
                    next_logits = torch.scatter(
                        next_logits, dim=-1, index=sorted_indices, src=sorted_logits
                    )

                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat((generated, next_token), dim=1)
            if eos_id is not None and torch.all(next_token == eos_id):
                break

        if was_training:
            self.train()
        return generated
