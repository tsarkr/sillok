#!/usr/bin/env python3
"""Read-only HTTP artifact & status server on 127.0.0.1:8000 for `cloudflared tunnel --url http://localhost:8000`.

Serves ONLY `data/processed/` (`/processed/`), `artifacts/` (`/artifacts/`),
`checkpoints/` (`/checkpoints/`), `docs/` (`/docs/`), and `logs/` (`/logs/`),
plus `/manifest.json` and `/status.json` live training telemetry.
"""

from __future__ import annotations

import argparse
import http.server
import io
import json
import os
import secrets
import tarfile
import urllib.parse
import urllib.request
from pathlib import Path

_MIME_BY_SUFFIX: dict[str, str] = {
    ".bin": "application/octet-stream",
    ".pt": "application/octet-stream",
    ".json": "application/json; charset=utf-8",
    ".jsonl": "application/x-ndjan; charset=utf-8",
    ".md": "text/markdown; charset=utf-8",
    ".log": "text/plain; charset=utf-8",
    ".txt": "text/plain; charset=utf-8",
    ".yaml": "text/yaml; charset=utf-8",
}


class _ArtifactHandler(http.server.SimpleHTTPRequestHandler):
    """Read-only HTTP handler restricted to configured mount prefixes."""

    mounts: dict[str, Path] = {}
    project_root: Path = Path(".")
    upload_token: str = ""

    def do_POST(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        req_path = urllib.parse.unquote(parsed.path)

        if req_path == "/api/generate":
            length = int(self.headers.get("Content-Length", "0") or 0)
            data = self.rfile.read(length) if length > 0 else b"{}"
            try:
                req = urllib.request.Request(
                    "http://127.0.0.1:8020/api/generate",
                    data=data,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=120) as r:
                    body = r.read()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                    self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
                    self.end_headers()
                    self.wfile.write(body)
                    return
            except Exception as exc:
                self.send_error(502, f"Studio backend error: {exc}")
                return

        if not req_path.startswith("/upload_checkpoint/"):
            self.send_error(404, "Not found")
            return

        auth = self.headers.get("Authorization", "")
        if not self.upload_token or not auth.startswith("Bearer ") or not secrets.compare_digest(auth[7:].strip(), self.upload_token):
            self.send_error(401, "Unauthorized")
            return

        rel = req_path.removeprefix("/upload_checkpoint/").lstrip("/")
        ckpt_root = self.mounts.get("checkpoints")
        if not rel or ckpt_root is None:
            self.send_error(400, "Invalid checkpoint path")
            return
        root = ckpt_root.resolve()
        dest = (root / rel).resolve()
        if dest == root or root not in dest.parents:
            self.send_error(403, "Forbidden path")
            return

        length = int(self.headers.get("Content-Length", "0") or 0)
        if length <= 0:
            self.send_error(400, "Missing Content-Length")
            return

        qs = urllib.parse.parse_qs(parsed.query)
        chunk_idx = int(qs.get("chunk_index", ["0"])[0])
        total_chunks = int(qs.get("total_chunks", ["1"])[0])

        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(f".{dest.name}.part")
        mode = "wb" if chunk_idx == 0 else "ab"
        remaining = length
        with tmp.open(mode) as fout:
            while remaining > 0:
                chunk = self.rfile.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                fout.write(chunk)
                remaining -= len(chunk)
        if remaining != 0:
            self.send_error(400, "Incomplete upload")
            return
        if chunk_idx + 1 >= total_chunks:
            tmp.replace(dest)
        self._send_json_bytes({"ok": True, "saved": str(dest.relative_to(root)), "chunk": chunk_idx, "total_chunks": total_chunks})

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()

    def do_HEAD(self) -> None:  # noqa: N802
        self.do_GET()

    def do_GET(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        req_path = urllib.parse.unquote(parsed.path)

        if req_path in ("/studio", "/studio/"):
            try:
                with urllib.request.urlopen("http://127.0.0.1:8020/", timeout=5) as r:
                    body = r.read()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
            except Exception as exc:
                self.send_error(502, f"Studio backend unavailable: {exc}")
                return

        if req_path == "/api/status":
            try:
                with urllib.request.urlopen("http://127.0.0.1:8020/api/status", timeout=5) as r:
                    body = r.read()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                    self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
                    self.end_headers()
                    self.wfile.write(body)
                    return
            except Exception as exc:
                self.send_error(502, f"Studio backend unavailable: {exc}")
                return

        if req_path in ("/", "/manifest.json"):
            self._send_manifest()
            return
        if req_path == "/status.json":
            self._send_status()
            return
        if req_path == "/job.json":
            self._send_job()
            return
        if req_path == "/code.tar.gz":
            self._send_live_code_bundle()
            return

        target = self._resolve_mount_path(req_path)
        if target is None or not target.exists():
            self.send_error(404, "Artifact not found")
            return
        if target.is_dir():
            self._send_dir_json(req_path, target)
            return

        file_size = target.stat().st_size
        ctype = _MIME_BY_SUFFIX.get(target.suffix.lower(), "application/octet-stream")
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(file_size))
        self.end_headers()
        with target.open("rb") as fin:
            while True:
                chunk = fin.read(1024 * 1024)
                if not chunk:
                    break
                self.wfile.write(chunk)

    def _resolve_mount_path(self, req_path: str) -> Path | None:
        clean = req_path.lstrip("/")
        if not clean:
            return None
        parts = clean.split("/", 1)
        prefix = parts[0]
        if prefix not in self.mounts:
            return None
        root = self.mounts[prefix].resolve()
        rel = parts[1] if len(parts) > 1 else ""
        candidate = (root / rel).resolve()
        if candidate != root and root not in candidate.parents:
            return None
        return candidate

    def _send_json_bytes(self, obj: object) -> None:
        payload = json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_dir_json(self, req_path: str, directory: Path) -> None:
        items = [
            {
                "name": p.name,
                "is_dir": p.is_dir(),
                "size_bytes": p.stat().st_size if p.is_file() else 0,
            }
            for p in sorted(directory.iterdir())
        ]
        self._send_json_bytes({"path": req_path, "entries": items})

    def _send_manifest(self) -> None:
        files: list[dict[str, object]] = []
        for prefix, root in sorted(self.mounts.items()):
            if not root.exists():
                continue
            for dirpath, _, filenames in os.walk(root):
                for fname in sorted(filenames):
                    fpath = Path(dirpath) / fname
                    rel = fpath.relative_to(root).as_posix()
                    files.append(
                        {
                            "url_path": f"/{prefix}/{rel}",
                            "size_bytes": fpath.stat().st_size,
                        }
                    )
        self._send_json_bytes(
            {
                "service": "Sillok-LM Artifact & Telemetry Server",
                "endpoints": ["/manifest.json", "/status.json", "/processed/", "/artifacts/", "/checkpoints/", "/docs/", "/logs/"],
                "mounts": {k: str(v) for k, v in self.mounts.items()},
                "files": files,
            }
        )

    def _send_status(self) -> None:
        status: dict[str, object] = {}
        proc_root = self.mounts.get("processed")
        if proc_root and (proc_root / "dataset_meta.json").exists():
            try:
                status["dataset_meta"] = json.loads((proc_root / "dataset_meta.json").read_text(encoding="utf-8"))
            except Exception:
                pass
        art_root = self.mounts.get("artifacts")
        if art_root and (art_root / "tokenizer" / "tokenizer_meta.json").exists():
            try:
                status["tokenizer_meta"] = json.loads((art_root / "tokenizer" / "tokenizer_meta.json").read_text(encoding="utf-8"))
            except Exception:
                pass
        log_root = self.mounts.get("logs")
        if log_root and (log_root / "pretrain_daemon.log").exists():
            lines = (log_root / "pretrain_daemon.log").read_text(encoding="utf-8", errors="ignore").splitlines()
            status["recent_training_log"] = lines[-25:]
        self._send_json_bytes(status)

    def _send_job(self) -> None:
        job_path = self.project_root / "configs" / "colab_job.json"
        if not job_path.exists():
            self.send_error(404, "colab_job.json not found")
            return
        try:
            job = json.loads(job_path.read_text(encoding="utf-8"))
        except Exception as exc:
            self.send_error(500, f"Invalid colab_job.json: {exc}")
            return
        self._send_json_bytes(job)

    def _send_live_code_bundle(self) -> None:
        buf = io.BytesIO()
        include_items = ("pyproject.toml", "requirements.txt", "src", "configs", "scripts", "tests")

        def _filter(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
            if "__pycache__" in info.name or info.name.endswith((".pyc", ".pyo")):
                return None
            return info

        with tarfile.open(fileobj=buf, mode="w:gz") as tf:
            for item in include_items:
                target = self.project_root / item
                if target.exists():
                    tf.add(target, arcname=item, filter=_filter)

        payload = buf.getvalue()
        self.send_response(200)
        self.send_header("Content-Type", "application/gzip")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve processed Sillok artifacts for Cloudflare Tunnel.")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--checkpoints-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    parser.add_argument("--logs-dir", type=Path, default=Path("logs"))
    args = parser.parse_args()

    mounts = {
        "processed": args.processed_dir.resolve(),
        "artifacts": args.artifacts_dir.resolve(),
        "checkpoints": args.checkpoints_dir.resolve(),
        "docs": args.docs_dir.resolve(),
        "logs": args.logs_dir.resolve(),
    }
    token_file = Path.home() / ".colab-gpu" / "token"
    upload_token = token_file.read_text(encoding="utf-8").strip() if token_file.exists() else ""

    handler_cls = type(
        "BoundArtifactHandler",
        (_ArtifactHandler,),
        {"mounts": mounts, "project_root": Path.cwd().resolve(), "upload_token": upload_token},
    )
    server = http.server.ThreadingHTTPServer((args.host, args.port), handler_cls)
    print(f"Serving read-only Sillok artifacts on http://{args.host}:{args.port}", flush=True)
    for prefix, path in mounts.items():
        print(f"  /{prefix}/ -> {path}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down artifact server.", flush=True)
        server.server_close()


if __name__ == "__main__":
    main()
