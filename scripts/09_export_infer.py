#!/usr/bin/env python3
"""Export a lightweight FP16 inference checkpoint (`infer.pt`) from a training checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def export_inference_checkpoint(src_path: Path, dst_path: Path, dtype: str = "float16") -> dict[str, object]:
    payload = torch.load(src_path, map_location="cpu", weights_only=False)
    target_dtype = torch.float16 if dtype == "float16" else torch.float32

    clean_sd = {}
    for k, v in payload["model_state_dict"].items():
        clean_k = k[10:] if k.startswith("_orig_mod.") else k
        clean_sd[clean_k] = v.to(dtype=target_dtype) if v.is_floating_point() else v

    infer_payload = {
        "global_step": int(payload.get("global_step", 0)),
        "best_val_loss": float(payload.get("best_val_loss", float("nan"))),
        "model_config": payload["model_config"],
        "model_state_dict": clean_sd,
        "dtype": dtype,
    }
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(infer_payload, dst_path)

    src_mb = src_path.stat().st_size / (1024 * 1024)
    dst_mb = dst_path.stat().st_size / (1024 * 1024)
    return {
        "src_path": str(src_path),
        "dst_path": str(dst_path),
        "global_step": infer_payload["global_step"],
        "best_val_loss": infer_payload["best_val_loss"],
        "src_size_mb": round(src_mb, 2),
        "dst_size_mb": round(dst_mb, 2),
        "reduction_percent": round((1.0 - dst_mb / max(1e-6, src_mb)) * 100.0, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export lightweight inference checkpoint.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("checkpoints/sillok_100m_t4/latest.pt"),
        help="Source training checkpoint (.pt)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("checkpoints/sillok_100m_t4/infer.pt"),
        help="Destination inference checkpoint (.pt)",
    )
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    args = parser.parse_args()

    info = export_inference_checkpoint(args.input, args.output, dtype=args.dtype)
    print(
        f"[export] {info['src_path']} ({info['src_size_mb']} MiB) -> "
        f"{info['dst_path']} ({info['dst_size_mb']} MiB, -{info['reduction_percent']}%) "
        f"at step {info['global_step']}"
    )


if __name__ == "__main__":
    main()
