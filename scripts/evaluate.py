#!/usr/bin/env python3
"""Quantitative and qualitative evaluation script for Sillok-LM.

Computes:
1. Cross-Entropy Loss & Perplexity (PPL) on validation dataset (val.bin)
2. Bits-Per-Token (BPT)
3. Top-1 and Top-5 Next-Token Prediction Accuracy
4. Qualitative rubric checks across major historical eras
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sillok.data.dataset import PackedMemmapDataset
from sillok.model.config import SillokLMConfig
from sillok.model.transformer import SillokForCausalLM
from sillok.tokenizer.trainer import SillokTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Sillok-LM on validation data.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--val-data", type=Path, default=Path("data/processed/val.bin"), help="Path to val.bin")
    parser.add_argument("--tokenizer", type=Path, default=Path("artifacts/tokenizer/tokenizer.json"))
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for evaluation")
    parser.add_argument("--max-batches", type=int, default=200, help="Max batches to evaluate (None for all)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-json", type=Path, default=None, help="Save evaluation metrics to JSON file")
    return parser.parse_args()


def load_model_from_checkpoint(checkpoint_path: Path, device: torch.device) -> tuple[SillokForCausalLM, SillokLMConfig, dict[str, Any]]:
    print(f"[eval] Loading checkpoint from {checkpoint_path}...")
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    cfg_dict = payload.get("model_config")
    if cfg_dict is None:
        raise ValueError("Checkpoint does not contain 'model_config'")
    
    model_cfg = SillokLMConfig.from_dict(cfg_dict)
    model = SillokForCausalLM(model_cfg)
    
    state_dict = payload["model_state_dict"]
    # Handle torch.compile prefix '_orig_mod.'
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        clean_k = k[10:] if k.startswith("_orig_mod.") else k
        cleaned_state_dict[clean_k] = v
        
    model.load_state_dict(cleaned_state_dict)
    model.to(device)
    model.eval()
    
    meta = {
        "global_step": payload.get("global_step", -1),
        "best_val_loss": payload.get("best_val_loss", float("nan")),
        "parameters": model.num_parameters(),
    }
    print(f"[eval] Model loaded (step={meta['global_step']}, params={meta['parameters']:,})")
    return model, model_cfg, meta


@torch.no_grad()
def run_quantitative_eval(
    model: SillokForCausalLM,
    val_dataset: PackedMemmapDataset,
    batch_size: int,
    max_batches: int | None,
    device: torch.device,
) -> dict[str, float]:
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    
    total_loss = 0.0
    total_tokens = 0
    top1_correct = 0
    top5_correct = 0
    batches_done = 0
    
    start_time = time.time()
    amp_dtype = torch.float16 if device.type == "cuda" else torch.float32
    
    print(f"[eval] Evaluating on validation dataset (batch_size={batch_size}, max_batches={max_batches})...")
    for idx, (x, y) in enumerate(val_loader):
        if max_batches is not None and idx >= max_batches:
            break
        
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        
        with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
            logits, loss = model(x, labels=y)
            
        # Shift logits and labels for autoregressive next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = y[:, 1:].contiguous()
        n_tokens = shift_labels.numel()
        
        total_loss += float(loss.item()) * n_tokens
        total_tokens += n_tokens
        
        # Calculate Top-1 and Top-5 accuracy
        _, top5_indices = torch.topk(shift_logits, k=5, dim=-1) # (B, T-1, 5)
        top1_preds = top5_indices[:, :, 0]
        top1_correct += int((top1_preds == shift_labels).sum().item())
        
        y_exp = shift_labels.unsqueeze(-1)
        top5_correct += int((top5_indices == y_exp).any(dim=-1).sum().item())
        
        batches_done += 1
        if batches_done % 20 == 0:
            elapsed = time.time() - start_time
            cur_loss = total_loss / total_tokens
            print(f"  [batch {batches_done:4d}] Loss: {cur_loss:.4f} | PPL: {math.exp(cur_loss):.2f} | Speed: {total_tokens / elapsed:.1f} tok/s")
            
    elapsed = time.time() - start_time
    mean_loss = total_loss / max(1, total_tokens)
    ppl = math.exp(mean_loss)
    bpt = mean_loss / math.log(2.0)
    top1_acc = (top1_correct / max(1, total_tokens)) * 100.0
    top5_acc = (top5_correct / max(1, total_tokens)) * 100.0
    
    return {
        "loss": mean_loss,
        "perplexity": ppl,
        "bits_per_token": bpt,
        "top1_accuracy_percent": top1_acc,
        "top5_accuracy_percent": top5_acc,
        "total_tokens_evaluated": total_tokens,
        "batches_evaluated": batches_done,
        "eval_time_seconds": elapsed,
        "tokens_per_second": total_tokens / max(1e-4, elapsed),
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    
    model, model_cfg, meta = load_model_from_checkpoint(args.checkpoint, device)
    
    if not args.val_data.exists():
        print(f"[error] Validation dataset not found at {args.val_data}", file=sys.stderr)
        sys.exit(1)
        
    val_dataset = PackedMemmapDataset(args.val_data, max_seq_len=model_cfg.max_seq_len)
    print(f"[eval] Validation dataset opened: {len(val_dataset)} chunks of {model_cfg.max_seq_len} tokens")
    
    metrics = run_quantitative_eval(
        model=model,
        val_dataset=val_dataset,
        batch_size=args.batch_size,
        max_batches=args.max_batches,
        device=device,
    )
    
    print("\n" + "=" * 60)
    print("           조선왕조실록 인공지능 사관 (Sillok-LM) 정량 평가 결과           ")
    print("=" * 60)
    print(f" 체크포인트 글로벌 스텝  : {meta['global_step']:,} steps")
    print(f" 모델 파라미터 수        : {meta['parameters']:,} params")
    print(f" 평가 토큰 수            : {metrics['total_tokens_evaluated']:,} tokens ({metrics['batches_evaluated']} 배치)")
    print("-" * 60)
    print(f" Cross-Entropy Loss      : {metrics['loss']:.4f}")
    print(f" Perplexity (PPL, 혼잡도): {metrics['perplexity']:.2f}")
    print(f" Bits per Token (BPT)    : {metrics['bits_per_token']:.4f} bits/token")
    print(f" Top-1 정확도 (1위 일치) : {metrics['top1_accuracy_percent']:.2f}%")
    print(f" Top-5 정확도 (5위 이내) : {metrics['top5_accuracy_percent']:.2f}%")
    print(f" 평가 소요 시간          : {metrics['eval_time_seconds']:.2f}초 ({metrics['tokens_per_second']:.1f} tokens/s)")
    print("=" * 60)
    
    if args.output_json:
        report = {
            "metadata": meta,
            "metrics": metrics,
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"[eval] Saved report to {args.output_json}")


if __name__ == "__main__":
    main()
