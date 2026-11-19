#!/usr/bin/env python3
"""Ephemeral-runtime-safe pretraining launcher.

Colab runtimes are recycled without warning and `/content` is wiped. This launcher:
  1. Restores `latest.pt` from a private Hugging Face repo (if present) so training resumes.
  2. Runs `04_pretrain.py` as a subprocess (resume_from: auto picks up latest.pt).
  3. Watches for new immutable `step_*.pt` checkpoints and uploads each as `latest.pt`
     (+ a weights-only `best_model.pt` when `best.pt` changes), squashing repo history
     so overwritten multi-GB files don't accumulate storage.
  4. Writes a `DONE` marker to the repo on successful completion; re-runs exit immediately.

Requires HF_TOKEN in the environment. Safe to launch repeatedly (e.g. from an auto-healer).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
import yaml
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError


def log(msg: str) -> None:
    print(f"[hf-sync {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def remote_has(api: HfApi, repo_id: str, filename: str) -> bool:
    try:
        return filename in api.list_repo_files(repo_id)
    except Exception:
        return False


def upload(api: HfApi, repo_id: str, local: Path, remote_name: str, message: str) -> None:
    t0 = time.time()
    api.upload_file(path_or_fileobj=str(local), path_in_repo=remote_name, repo_id=repo_id, commit_message=message)
    try:
        api.super_squash_history(repo_id=repo_id)
    except Exception as exc:  # squashing is best-effort storage hygiene
        log(f"history squash skipped: {exc}")
    log(f"uploaded {local.name} -> {remote_name} ({local.stat().st_size / 1e9:.2f} GB, {time.time() - t0:.0f}s)")


def export_best_weights(best_pt: Path, out: Path) -> None:
    payload = torch.load(best_pt, map_location="cpu", weights_only=False)
    torch.save(
        {
            "global_step": payload.get("global_step"),
            "best_val_loss": payload.get("best_val_loss"),
            "model_config": payload.get("model_config"),
            "model_state_dict": payload["model_state_dict"],
        },
        out,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo-id", default="darkstar-11e/sillok-350m-ckpt")
    parser.add_argument("--poll-sec", type=int, default=60)
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("HF_TOKEN is required")
    api = HfApi(token=token)
    api.create_repo(args.repo_id, private=True, exist_ok=True)

    if remote_has(api, args.repo_id, "DONE"):
        log("DONE marker present in repo; training already completed. Exiting.")
        return

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    out_dir = Path(cfg["training"]["output_dir"])
    if not out_dir.is_absolute():
        out_dir = Path.cwd() / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    latest = out_dir / "latest.pt"
    if not latest.exists() and remote_has(api, args.repo_id, "latest.pt"):
        log("restoring latest.pt from Hugging Face...")
        hf_hub_download(args.repo_id, "latest.pt", local_dir=str(out_dir), token=token)
        step = torch.load(latest, map_location="cpu", weights_only=False).get("global_step")
        log(f"restored checkpoint at step {step}")
    elif latest.exists():
        log("local latest.pt present; resuming from it")
    else:
        log("no checkpoint found; starting from scratch")

    cmd = [sys.executable, "-u", "scripts/04_pretrain.py", "--config", str(args.config)]
    if args.no_compile:
        cmd.append("--no-compile")
    proc = subprocess.Popen(cmd, env=dict(os.environ, PYTHONUNBUFFERED="1"))
    log(f"pretraining started (pid {proc.pid})")

    uploaded_step: str | None = None
    best_mtime = (out_dir / "best.pt").stat().st_mtime if (out_dir / "best.pt").exists() else 0.0

    def sync() -> None:
        nonlocal uploaded_step, best_mtime
        steps = sorted(out_dir.glob("step_*.pt"))
        if steps and steps[-1].name != uploaded_step:
            newest = steps[-1]
            try:
                upload(api, args.repo_id, newest, "latest.pt", f"checkpoint {newest.stem}")
                uploaded_step = newest.name
            except Exception as exc:
                log(f"upload failed (will retry): {exc}")
        best = out_dir / "best.pt"
        if best.exists() and best.stat().st_mtime > best_mtime:
            try:
                tmp = out_dir / "best_model.export.pt"
                export_best_weights(best, tmp)
                upload(api, args.repo_id, tmp, "best_model.pt", "best validation weights")
                tmp.unlink(missing_ok=True)
                best_mtime = best.stat().st_mtime
            except Exception as exc:
                log(f"best upload failed (will retry): {exc}")

    while proc.poll() is None:
        time.sleep(args.poll_sec)
        sync()

    rc = proc.returncode
    log(f"pretraining exited with code {rc}")
    if rc == 0:
        # Final latest.pt is written at loop end and may be newer than the last step file.
        try:
            upload(api, args.repo_id, latest, "latest.pt", "final checkpoint")
        except Exception as exc:
            log(f"final upload failed: {exc}")
        sync()
        marker = out_dir / "DONE"
        marker.write_text(time.strftime("%Y-%m-%dT%H:%M:%S"))
        upload(api, args.repo_id, marker, "DONE", "training complete")
    sys.exit(rc)


if __name__ == "__main__":
    main()
