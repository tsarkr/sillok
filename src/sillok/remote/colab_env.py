"""Detect local Edge host vs. Google Colab Pro runtimes, tunnel links, and Drive persistence."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_TRYCLOUDFLARE_URL_RE = re.compile(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")


@dataclass
class HostProcessInfo:
    """Active remote-access and tunnel services observed on the Linux host."""

    has_sshd: bool
    has_vscode_server: bool
    has_cloudflared: bool
    cloudflared_target_url: str
    cloudflared_public_url: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RuntimeInfo:
    """Snapshot of the active local Edge host or Colab remote runtime."""

    is_colab: bool
    has_google_drive: bool
    drive_mount_root: str
    python_version: str
    cpu_model: str
    cpu_count: int
    mem_total_gb: float
    mem_available_gb: float
    torch_installed: bool
    cuda_available: bool
    gpu_name: str
    gpu_vram_gb: float
    bf16_supported: bool
    host_processes: HostProcessInfo

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _read_proc_meminfo() -> tuple[float, float]:
    meminfo = Path("/proc/meminfo")
    if not meminfo.exists():
        return 0.0, 0.0
    total_kb = 0
    avail_kb = 0
    for line in meminfo.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("MemTotal:"):
            total_kb = int(line.split()[1])
        elif line.startswith("MemAvailable:"):
            avail_kb = int(line.split()[1])
    return round(total_kb / (1024**2), 2), round(avail_kb / (1024**2), 2)


def _read_cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if not cpuinfo.exists():
        return "unknown"
    for line in cpuinfo.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.lower().startswith("model name"):
            parts = line.split(":", 1)
            if len(parts) == 2:
                return parts[1].strip()
    return "unknown"


def detect_host_processes() -> HostProcessInfo:
    """Detect active sshd, VS Code Remote-SSH server, and cloudflared tunnel via /proc or ps."""
    has_sshd = False
    has_vscode = False
    has_cloudflared = False
    tunnel_url = ""

    cmdlines: list[str] = []
    proc_dir = Path("/proc")
    if proc_dir.exists():
        for entry in proc_dir.iterdir():
            if not entry.name.isdigit():
                continue
            cmdline_file = entry / "cmdline"
            try:
                raw = cmdline_file.read_bytes()
            except OSError:
                continue
            if raw:
                cmdlines.append(raw.replace(b"\x00", b" ").decode("utf-8", errors="ignore").strip())
    else:
        try:
            out = subprocess.check_output(["ps", "-eo", "args"], text=True, stderr=subprocess.DEVNULL)
            cmdlines.extend(out.splitlines())
        except Exception:
            pass

    for cmd in cmdlines:
        if "sshd:" in cmd or "/usr/sbin/sshd" in cmd:
            has_sshd = True
        if ".vscode-server" in cmd or "code-server" in cmd:
            has_vscode = True
        if "cloudflared" in cmd and "tunnel" in cmd:
            has_cloudflared = True
            tokens = cmd.split()
            for idx, tok in enumerate(tokens):
                if tok == "--url" and idx + 1 < len(tokens):
                    tunnel_url = tokens[idx + 1]
                    break
                if tok.startswith("--url="):
                    tunnel_url = tok.split("=", 1)[1]
                    break

    public_url = ""
    for candidate_log in (Path("/home/nsua/tunnel.log"), Path.home() / "tunnel.log"):
        if candidate_log.exists():
            try:
                content = candidate_log.read_text(encoding="utf-8", errors="ignore")
                matches = _TRYCLOUDFLARE_URL_RE.findall(content)
                if matches:
                    public_url = matches[-1]
                    break
            except OSError:
                pass

    return HostProcessInfo(
        has_sshd=has_sshd,
        has_vscode_server=has_vscode,
        has_cloudflared=has_cloudflared,
        cloudflared_target_url=tunnel_url,
        cloudflared_public_url=public_url,
    )


def detect_runtime() -> RuntimeInfo:
    """Inspect runtime environment without requiring torch to be installed."""
    is_colab = (
        "COLAB_RELEASE_TAG" in os.environ
        or "COLAB_GPU" in os.environ
        or (Path("/content").exists() and Path("/usr/local/lib").exists() and "google.colab" in sys.modules)
    )
    if not is_colab:
        try:
            import google.colab  # type: ignore  # noqa: F401

            is_colab = True
        except ImportError:
            is_colab = False

    drive_root = "/content/drive/MyDrive"
    has_drive = Path(drive_root).exists()
    mem_total_gb, mem_avail_gb = _read_proc_meminfo()
    cpu_model = _read_cpu_model()
    host_procs = detect_host_processes()

    torch_installed = False
    cuda_available = False
    gpu_name = "none"
    gpu_vram_gb = 0.0
    bf16_supported = False

    try:
        import torch

        torch_installed = True
        cuda_available = bool(torch.cuda.is_available())
        if cuda_available:
            props = torch.cuda.get_device_properties(0)
            gpu_name = props.name
            gpu_vram_gb = round(props.total_memory / (1024**3), 2)
            bf16_supported = bool(torch.cuda.is_bf16_supported())
    except ImportError:
        pass

    return RuntimeInfo(
        is_colab=is_colab,
        has_google_drive=has_drive,
        drive_mount_root=drive_root if has_drive else "",
        python_version=sys.version.split()[0],
        cpu_model=cpu_model,
        cpu_count=os.cpu_count() or 1,
        mem_total_gb=mem_total_gb,
        mem_available_gb=mem_avail_gb,
        torch_installed=torch_installed,
        cuda_available=cuda_available,
        gpu_name=gpu_name,
        gpu_vram_gb=gpu_vram_gb,
        bf16_supported=bf16_supported,
        host_processes=host_procs,
    )


def format_runtime_report(info: RuntimeInfo | None = None) -> str:
    """Return a readable summary string for diagnostics and notebooks."""
    snap = info or detect_runtime()
    mode = "Google Colab Remote" if snap.is_colab else "Local / Edge SSH Host"
    procs = snap.host_processes
    if procs.has_cloudflared:
        pub = f", public={procs.cloudflared_public_url}" if procs.cloudflared_public_url else ""
        tunnel_desc = f"active (-> {procs.cloudflared_target_url or 'unknown'}{pub})"
    else:
        tunnel_desc = "inactive"
    lines = [
        f"Runtime Mode    : {mode}",
        f"Python          : {snap.python_version}",
        f"CPU             : {snap.cpu_model} ({snap.cpu_count} cores)",
        f"System Memory   : {snap.mem_available_gb:.2f} / {snap.mem_total_gb:.2f} GiB available",
        f"PyTorch         : {'installed' if snap.torch_installed else 'not installed'}",
        f"CUDA / GPU      : {snap.gpu_name} ({snap.gpu_vram_gb:.2f} GB, bf16={snap.bf16_supported})"
        if snap.cuda_available
        else "CUDA / GPU      : unavailable",
        f"Host Services   : sshd={procs.has_sshd}, vscode-server={procs.has_vscode_server}, cloudflared={tunnel_desc}",
        f"Google Drive    : {snap.drive_mount_root}"
        if snap.has_google_drive
        else "Google Drive    : not mounted",
    ]
    return "\n".join(lines)


def download_from_tunnel(
    base_url: str,
    remote_rel_path: str,
    local_dest_path: str | Path,
    *,
    chunk_size: int = 1024 * 1024,
    timeout: float = 120.0,
) -> Path:
    """Stream-download a preprocessed artifact from the edge host's Cloudflare tunnel URL."""
    clean_base = base_url.rstrip("/")
    clean_rel = remote_rel_path.lstrip("/")
    url = urllib.parse.urljoin(f"{clean_base}/", clean_rel)

    dest = Path(local_dest_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_dest = dest.with_suffix(dest.suffix + ".part")

    req = urllib.request.Request(url, headers={"User-Agent": "Sillok-LM-Client/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as resp, tmp_dest.open("wb") as fout:
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            fout.write(chunk)

    tmp_dest.replace(dest)
    return dest


def mount_google_drive(mount_point: str = "/content/drive") -> Path:
    """Mount Google Drive inside a Colab runtime."""
    try:
        from google.colab import drive  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "mount_google_drive() can only run inside a Google Colab runtime."
        ) from exc

    drive.mount(mount_point, force_remount=False)
    return Path(mount_point) / "MyDrive"


def setup_drive_checkpoint_symlink(
    local_ckpt_dir: str | Path,
    drive_ckpt_dir: str | Path,
) -> Path:
    """Point `local_ckpt_dir` to `drive_ckpt_dir` via symlink for Colab durability."""
    local_path = Path(local_ckpt_dir)
    drive_path = Path(drive_ckpt_dir)
    drive_path.mkdir(parents=True, exist_ok=True)
    local_path.parent.mkdir(parents=True, exist_ok=True)

    if local_path.is_symlink():
        if local_path.resolve() == drive_path.resolve():
            return local_path
        local_path.unlink()
    elif local_path.exists():
        for item in local_path.iterdir():
            dest = drive_path / item.name
            if not dest.exists():
                shutil.move(str(item), str(dest))
        shutil.rmtree(local_path)

    os.symlink(drive_path, local_path)
    return local_path
