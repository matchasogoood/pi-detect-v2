"""One place that decides which compute device the project uses.

It looks at the operating system first and then only asks for the GPU type that system can have:
  Windows / Linux : NVIDIA GPU (CUDA)
  macOS           : Apple-silicon GPU (MPS); Intel Macs have none usable here
Everything else, or a GPU that PyTorch cannot use, falls back to the CPU.  PI_DEVICE (cuda / mps / cpu) overrides it.
"""
from __future__ import annotations
import os
import platform
import shutil
import subprocess


def os_name() -> str:
    """'windows', 'macos', 'linux' or 'other'."""
    return {"Windows": "windows", "Darwin": "macos", "Linux": "linux"}.get(platform.system(), "other")


def hardware_gpu() -> str | None:
    """Name of the GPU the machine physically has (found without PyTorch), or None.
    Lets the checker tell 'no GPU' apart from 'GPU present but PyTorch is a CPU-only build'."""
    o = os_name()
    try:
        if o in ("windows", "linux"):
            exe = shutil.which("nvidia-smi")
            if exe:
                out = subprocess.run([exe, "--query-gpu=name", "--format=csv,noheader"], capture_output=True,
                                     text=True, timeout=10).stdout.strip().splitlines()
                if out and out[0].strip():
                    return out[0].strip()
        elif o == "macos":
            if platform.machine() == "arm64":          # Apple silicon: the GPU is part of the chip
                chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                      text=True, timeout=10).stdout.strip()
                return chip or "Apple silicon GPU"
    except Exception:  # noqa: BLE001
        pass
    return None


def available_devices() -> dict:
    """What PyTorch can use here: {'os', 'cuda': name|None, 'mps': bool, 'hardware': name|None, 'cpu': True}."""
    o = os_name()
    info = {"os": o, "cuda": None, "mps": False, "hardware": hardware_gpu(), "cpu": True}
    try:
        import torch
        if o in ("windows", "linux", "other") and torch.cuda.is_available():
            info["cuda"] = torch.cuda.get_device_name(0)
        if o == "macos":
            mps = getattr(getattr(torch, "backends", None), "mps", None)
            info["mps"] = bool(mps and mps.is_available())
    except Exception:  # noqa: BLE001 - torch missing or broken: CPU only
        pass
    return info


def best_device() -> str:
    forced = os.environ.get("PI_DEVICE", "").strip().lower()
    if forced in ("cuda", "mps", "cpu"):
        return forced
    d = available_devices()
    if d["cuda"]:
        return "cuda"
    if d["mps"]:
        return "mps"
    return "cpu"
