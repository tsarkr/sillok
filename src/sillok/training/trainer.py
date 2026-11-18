"""A100-optimized from-scratch pretraining loop with checkpoint resume support."""

from __future__ import annotations

import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from sillok.data.dataset import PackedMemmapDataset
from sillok.model.config import SillokLMConfig
from sillok.model.transformer import SillokForCausalLM


@dataclass
class TrainingConfig:
    """Hyperparameters for Sillok-LM causal pretraining."""

    train_bin: str = "data/processed/train.bin"
    val_bin: str = "data/processed/val.bin"
    output_dir: str = "checkpoints/sillok"
    batch_size: int = 8
    gradient_accumulation_steps: int = 8
    max_steps: int = 20000
    warmup_steps: int = 1000
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    eval_interval: int = 500
    eval_steps: int = 50
    save_interval: int = 1000
    max_keep_checkpoints: int = 3
    log_interval: int = 20
    num_workers: int = 2
    dtype: str = "bfloat16"
    token_dtype: str = "uint16"
    compile_model: bool = True
    resume_from: str | None = "auto"
    seed: int = 42
    wandb_project: str | None = None

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "TrainingConfig":
        payload = yaml.safe_load(Path(yaml_path).read_text(encoding="utf-8"))
        train_dict = payload.get("training", payload)
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in train_dict.items() if k in valid_fields}
        return cls(**filtered)


def _infinite_loader(loader: DataLoader) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    while True:
        for batch in loader:
            yield batch


def _unwrap_model(model: torch.nn.Module) -> SillokForCausalLM:
    return getattr(model, "_orig_mod", model)


class SillokPretrainer:
    """Manages from-scratch pretraining, evaluation, and checkpointing."""

    def __init__(
        self,
        model_config: SillokLMConfig,
        train_config: TrainingConfig,
    ) -> None:
        self.model_config = model_config
        self.train_config = train_config
        self.output_dir = Path(train_config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._seed_everything(train_config.seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.amp_dtype = self._resolve_amp_dtype(train_config.dtype)
        self.use_amp = self.device.type == "cuda" and self.amp_dtype != torch.float32
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=(self.device.type == "cuda" and self.amp_dtype == torch.float16)
        )

        self.raw_model = SillokForCausalLM(model_config).to(self.device)
        self.model: torch.nn.Module = self.raw_model
        if train_config.compile_model and self.device.type == "cuda" and hasattr(torch, "compile"):
            self.model = torch.compile(self.raw_model)

        self.optimizer = self._build_optimizer()
        self.global_step = 0
        self.best_val_loss = float("inf")

        self.train_loader, self.val_loader = self._build_dataloaders()
        self._maybe_resume()

    @staticmethod
    def _seed_everything(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _resolve_amp_dtype(self, dtype_name: str) -> torch.dtype:
        mapping = {
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        dtype = mapping.get(dtype_name.lower(), torch.float32)
        if dtype == torch.bfloat16 and self.device.type == "cuda":
            if not torch.cuda.is_bf16_supported():
                return torch.float16
        if self.device.type == "cpu" and dtype == torch.float16:
            return torch.float32
        return dtype

    def _build_optimizer(self) -> torch.optim.AdamW:
        decay_params: list[torch.nn.Parameter] = []
        no_decay_params: list[torch.nn.Parameter] = []
        seen: set[int] = set()
        for param in self.raw_model.parameters():
            if not param.requires_grad:
                continue
            ptr = param.data_ptr()
            if ptr in seen:
                continue
            seen.add(ptr)
            if param.dim() >= 2:
                decay_params.append(param)
            else:
                no_decay_params.append(param)

        param_groups = [
            {"params": decay_params, "weight_decay": self.train_config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        use_fused = self.device.type == "cuda"
        try:
            return torch.optim.AdamW(
                param_groups,
                lr=self.train_config.learning_rate,
                betas=(self.train_config.beta1, self.train_config.beta2),
                fused=use_fused,
            )
        except TypeError:
            return torch.optim.AdamW(
                param_groups,
                lr=self.train_config.learning_rate,
                betas=(self.train_config.beta1, self.train_config.beta2),
            )

    def _build_dataloaders(self) -> tuple[DataLoader, DataLoader | None]:
        train_ds = PackedMemmapDataset(
            self.train_config.train_bin,
            self.model_config.max_seq_len,
            dtype=self.train_config.token_dtype,
        )
        pin = self.device.type == "cuda"
        train_loader = DataLoader(
            train_ds,
            batch_size=self.train_config.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.train_config.num_workers,
            pin_memory=pin,
        )

        val_loader: DataLoader | None = None
        val_path = Path(self.train_config.val_bin)
        if val_path.exists():
            val_ds = PackedMemmapDataset(
                val_path,
                self.model_config.max_seq_len,
                dtype=self.train_config.token_dtype,
            )
            val_loader = DataLoader(
                val_ds,
                batch_size=self.train_config.batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=self.train_config.num_workers,
                pin_memory=pin,
            )
        return train_loader, val_loader

    def _get_lr(self, step: int) -> float:
        cfg = self.train_config
        if step < cfg.warmup_steps:
            return cfg.learning_rate * float(step + 1) / float(max(1, cfg.warmup_steps))
        if step >= cfg.max_steps:
            return cfg.min_learning_rate
        progress = float(step - cfg.warmup_steps) / float(max(1, cfg.max_steps - cfg.warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return cfg.min_learning_rate + cosine * (cfg.learning_rate - cfg.min_learning_rate)

    def _maybe_resume(self) -> None:
        resume_cfg = self.train_config.resume_from
        if not resume_cfg:
            return
        ckpt_path = self.output_dir / "latest.pt" if resume_cfg == "auto" else Path(resume_cfg)
        if not ckpt_path.exists():
            return
        payload = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        _unwrap_model(self.model).load_state_dict(payload["model_state_dict"])
        self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        self.global_step = int(payload.get("global_step", 0))
        self.best_val_loss = float(payload.get("best_val_loss", float("inf")))
        if "scaler_state_dict" in payload and self.scaler.is_enabled():
            self.scaler.load_state_dict(payload["scaler_state_dict"])
        print(f"[resume] Loaded checkpoint {ckpt_path} at step {self.global_step}")

    def save_checkpoint(self, filename: str) -> Path:
        ckpt_path = self.output_dir / filename
        tmp_path = self.output_dir / f".{filename}.tmp"
        payload: dict[str, Any] = {
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
            "model_config": self.model_config.to_dict(),
            "train_config": asdict(self.train_config),
            "model_state_dict": _unwrap_model(self.model).state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        if self.scaler.is_enabled():
            payload["scaler_state_dict"] = self.scaler.state_dict()
        torch.save(payload, tmp_path)
        tmp_path.replace(ckpt_path)

        if filename.startswith("step_") and self.train_config.max_keep_checkpoints > 0:
            step_files = sorted(self.output_dir.glob("step_*.pt"))
            for old_ckpt in step_files[: -self.train_config.max_keep_checkpoints]:
                try:
                    old_ckpt.unlink()
                except OSError:
                    pass
        return ckpt_path

    @torch.no_grad()
    def evaluate(self) -> float:
        if self.val_loader is None:
            return float("nan")
        self.model.eval()
        losses: list[float] = []
        for idx, (input_ids, labels) in enumerate(self.val_loader):
            if idx >= self.train_config.eval_steps:
                break
            input_ids = input_ids.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            with torch.amp.autocast(self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
                _, loss = self.model(input_ids, labels=labels)
            if loss is not None:
                losses.append(float(loss.item()))
        self.model.train()
        return sum(losses) / len(losses) if losses else float("nan")

    def train(self) -> dict[str, float]:
        """Run the full pretraining loop up to `train_config.max_steps`."""
        cfg = self.train_config
        n_params = self.raw_model.num_parameters()
        tokens_per_step = cfg.batch_size * cfg.gradient_accumulation_steps * self.model_config.max_seq_len
        print(
            f"[train] device={self.device} dtype={self.amp_dtype} "
            f"params={n_params:,} tokens/step={tokens_per_step:,} max_steps={cfg.max_steps}"
        )

        wandb_run = None
        if cfg.wandb_project:
            try:
                import wandb  # type: ignore

                wandb_run = wandb.init(
                    project=cfg.wandb_project,
                    config={"model": self.model_config.to_dict(), "training": asdict(cfg)},
                )
            except ImportError:
                print("[warn] wandb not installed; skipping remote W&B logging.")

        self.model.train()
        data_iter = _infinite_loader(self.train_loader)
        last_loss = float("nan")
        t0 = time.perf_counter()

        while self.global_step < cfg.max_steps:
            lr = self._get_lr(self.global_step)
            for group in self.optimizer.param_groups:
                group["lr"] = lr

            self.optimizer.zero_grad(set_to_none=True)
            accum_loss = 0.0

            for _ in range(cfg.gradient_accumulation_steps):
                input_ids, labels = next(data_iter)
                input_ids = input_ids.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                with torch.amp.autocast(self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
                    _, loss = self.model(input_ids, labels=labels)
                    assert loss is not None
                    scaled_loss = loss / cfg.gradient_accumulation_steps

                if self.scaler.is_enabled():
                    self.scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
                accum_loss += float(loss.item()) / cfg.gradient_accumulation_steps

            if self.scaler.is_enabled():
                self.scaler.unscale_(self.optimizer)
            if cfg.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(self.raw_model.parameters(), cfg.grad_clip)

            if self.scaler.is_enabled():
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            self.global_step += 1
            last_loss = accum_loss

            if self.global_step % cfg.log_interval == 0 or self.global_step == 1:
                elapsed = max(1e-6, time.perf_counter() - t0)
                tok_per_sec = (tokens_per_step * cfg.log_interval) / elapsed if self.global_step > 1 else tokens_per_step / elapsed
                t0 = time.perf_counter()
                print(
                    f"step={self.global_step:06d}/{cfg.max_steps:06d} "
                    f"loss={last_loss:.4f} lr={lr:.3e} tok/s={tok_per_sec:,.0f}",
                    flush=True,
                )
                if wandb_run is not None:
                    wandb_run.log({"train/loss": last_loss, "train/lr": lr, "perf/tok_s": tok_per_sec}, step=self.global_step)

            if cfg.eval_interval > 0 and self.global_step % cfg.eval_interval == 0:
                val_loss = self.evaluate()
                if not math.isnan(val_loss):
                    ppl = math.exp(min(20.0, val_loss))
                    print(f"[eval] step={self.global_step:06d} val_loss={val_loss:.4f} ppl={ppl:.2f}")
                    if wandb_run is not None:
                        wandb_run.log({"val/loss": val_loss, "val/ppl": ppl}, step=self.global_step)
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.save_checkpoint("best.pt")

            if cfg.save_interval > 0 and self.global_step % cfg.save_interval == 0:
                self.save_checkpoint(f"step_{self.global_step:07d}.pt")
                self.save_checkpoint("latest.pt")

        self.save_checkpoint("latest.pt")
        if wandb_run is not None:
            wandb_run.finish()
        return {"final_step": float(self.global_step), "last_train_loss": last_loss, "best_val_loss": self.best_val_loss}
