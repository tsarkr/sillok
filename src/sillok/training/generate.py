"""Conditional text generation and persistent inference engine for Sillok-LM."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch

from sillok.data.formatting import format_generation_prompt
from sillok.model.config import SillokLMConfig
from sillok.model.transformer import SillokForCausalLM
from sillok.tokenizer.trainer import load_sillok_tokenizer


def _clean_state_dict(
    state_dict: dict[str, torch.Tensor],
    target_dtype: torch.dtype | None = None,
) -> dict[str, torch.Tensor]:
    cleaned: dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        clean_k = k[10:] if k.startswith("_orig_mod.") else k
        if target_dtype is not None and v.is_floating_point():
            cleaned[clean_k] = v.to(dtype=target_dtype)
        else:
            cleaned[clean_k] = v
    return cleaned


class SillokInferenceEngine:
    """Persistent in-memory inference engine for Sillok-LM checkpoints (`infer.pt` / `best.pt` / `latest.pt`)."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        tokenizer_path: str | Path = "artifacts/tokenizer/tokenizer.json",
        *,
        device: str | None = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.tokenizer_path = Path(tokenizer_path)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        t0 = time.time()
        payload = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        self.global_step = int(payload.get("global_step", -1))
        self.best_val_loss = float(payload.get("best_val_loss", float("nan")))

        self.model_cfg = SillokLMConfig.from_dict(payload["model_config"])
        self.model = SillokForCausalLM(self.model_cfg, init_weights=False)

        # On CPU, float32 is faster and universally supported; on CUDA, keep float16 if stored in fp16
        target_dtype = torch.float32 if self.device.type == "cpu" else None
        cleaned_sd = _clean_state_dict(payload["model_state_dict"], target_dtype=target_dtype)
        self.model.load_state_dict(cleaned_sd)
        self.model.to(self.device)
        self.model.eval()
        del payload

        self.tokenizer = load_sillok_tokenizer(self.tokenizer_path)
        self.eos_id = self.tokenizer.token_to_id("<|eos|>")
        self.load_time_sec = time.time() - t0

    def status_dict(self) -> dict[str, Any]:
        return {
            "checkpoint": str(self.checkpoint_path),
            "device": str(self.device),
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
            "parameters": self.model.num_parameters(),
            "vocab_size": self.model_cfg.vocab_size,
            "max_seq_len": self.model_cfg.max_seq_len,
            "load_time_sec": round(self.load_time_sec, 3),
        }

    @torch.no_grad()
    def generate(
        self,
        *,
        king: str = "",
        date: str = "",
        subject: str = "",
        title_ko: str = "",
        body_hanja: str = "",
        raw_prompt: str = "",
        target_mode: str = "hanja",
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.92,
        repetition_penalty: float = 1.15,
    ) -> dict[str, Any]:
        if raw_prompt.strip():
            prompt = raw_prompt.strip()
        else:
            prompt = format_generation_prompt(
                king=king,
                date=date,
                subject=subject,
                title_ko=title_ko,
                body_hanja=body_hanja,
                target_mode=target_mode,
            )

        encoding = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([encoding.ids], dtype=torch.long, device=self.device)

        t0 = time.time()
        use_amp = self.device.type == "cuda"
        with torch.amp.autocast(self.device.type, dtype=torch.float16, enabled=use_amp):
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                eos_token_id=self.eos_id,
            )
        elapsed = max(1e-4, time.time() - t0)

        gen_ids = output_ids[0, len(encoding.ids) :].tolist()
        generated_raw = self.tokenizer.decode(gen_ids, skip_special_tokens=False)
        generated_clean = generated_raw.replace("<|eos|>", "").strip()
        full_text = self.tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=False)

        return {
            "prompt": prompt,
            "generated": generated_clean,
            "generated_raw": generated_raw,
            "full_text": full_text,
            "prompt_tokens": len(encoding.ids),
            "generated_tokens": len(gen_ids),
            "elapsed_sec": round(elapsed, 3),
            "tokens_per_sec": round(len(gen_ids) / elapsed, 2),
            "global_step": self.global_step,
            "device": str(self.device),
        }


def generate_from_checkpoint(
    checkpoint_path: str | Path,
    tokenizer_path: str | Path,
    *,
    king: str = "",
    date: str = "",
    subject: str = "",
    title_ko: str = "",
    body_hanja: str = "",
    target_mode: str = "hanja",
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.95,
    repetition_penalty: float = 1.15,
    device: str | None = None,
) -> dict[str, Any]:
    """Load a checkpoint and generate Classical Hanja, Korean titles, or Saron commentary."""
    engine = SillokInferenceEngine(checkpoint_path, tokenizer_path, device=device)
    return engine.generate(
        king=king,
        date=date,
        subject=subject,
        title_ko=title_ko,
        body_hanja=body_hanja,
        target_mode=target_mode,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )
