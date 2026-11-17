#!/usr/bin/env python3
"""Launch from-scratch Sillok-LM pretraining from a YAML config."""

from __future__ import annotations

import argparse
from pathlib import Path

from sillok.model.config import SillokLMConfig
from sillok.remote.colab_env import detect_runtime, format_runtime_report, setup_drive_checkpoint_symlink
from sillok.training.trainer import SillokPretrainer, TrainingConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain Sillok-LM from scratch.")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config file")
    parser.add_argument("--output-dir", type=Path, default=None, help="Override training.output_dir")
    parser.add_argument("--drive-ckpt-dir", type=Path, default=None, help="Colab Drive dir to symlink checkpoints")
    parser.add_argument("--max-steps", type=int, default=None, help="Override training.max_steps")
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile")
    args = parser.parse_args()

    runtime = detect_runtime()
    print(format_runtime_report(runtime))

    model_cfg = SillokLMConfig.from_yaml(args.config)
    train_cfg = TrainingConfig.from_yaml(args.config)

    if args.output_dir is not None:
        train_cfg.output_dir = str(args.output_dir)
    if args.max_steps is not None:
        train_cfg.max_steps = args.max_steps
    if args.no_compile:
        train_cfg.compile_model = False

    if args.drive_ckpt_dir is not None:
        setup_drive_checkpoint_symlink(train_cfg.output_dir, args.drive_ckpt_dir)
        print(f"[remote] Linked {train_cfg.output_dir} -> {args.drive_ckpt_dir}")

    print(f"[model] Analytical parameters: {model_cfg.estimate_num_parameters():,}")
    trainer = SillokPretrainer(model_cfg, train_cfg)
    metrics = trainer.train()
    print(f"Training completed: {metrics}")


if __name__ == "__main__":
    main()
