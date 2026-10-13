"""Detects whether this machine can really run CUDA code.

CUDA is reported as available only when all of these hold:
  1. nvidia-smi is on PATH and reports at least one NVIDIA GPU, and
  2. nvcc (the CUDA Toolkit compiler) is on PATH.
Nothing is assumed from the operating system alone.
"""

from __future__ import annotations

import platform
import re
import shutil
import subprocess
from typing import Optional


def _run(cmd: list[str], timeout: float = 15.0) -> Optional[str]:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout


def _first_line(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return None


def detect_cuda() -> dict:
    status: dict = {
        "available": False,
        "reason": "",
        "gpu_name": None,
        "driver_version": None,
        "compute_capability": None,
        "cuda_version": None,
    }

    smi = shutil.which("nvidia-smi")
    if not smi:
        status["reason"] = (
            "No NVIDIA driver found (nvidia-smi is not on PATH). "
            "Local CUDA cannot run on this machine; use the Colab notebook instead."
        )
        return status

    gpu_line = _first_line(_run([smi, "--query-gpu=name,driver_version", "--format=csv,noheader,nounits"]))
    if not gpu_line or "," not in gpu_line:
        status["reason"] = "nvidia-smi is present but did not report an NVIDIA GPU."
        return status

    name, driver = [part.strip() for part in gpu_line.split(",", 1)]
    status["gpu_name"] = name
    status["driver_version"] = driver

    cc = _first_line(_run([smi, "--query-gpu=compute_cap", "--format=csv,noheader,nounits"]))
    if cc and re.fullmatch(r"\d+\.\d+", cc):
        status["compute_capability"] = cc

    nvcc = shutil.which("nvcc")
    if not nvcc:
        status["reason"] = (
            f"NVIDIA GPU '{name}' was found, but the CUDA Toolkit compiler (nvcc) is not on PATH. "
            "Install the CUDA Toolkit or use the Colab notebook."
        )
        return status

    nvcc_out = _run([nvcc, "--version"]) or ""
    match = re.search(r"release\s+(\d+\.\d+)", nvcc_out)
    status["cuda_version"] = match.group(1) if match else None
    status["available"] = True
    return status


def detect_compiler() -> dict:
    gpp = shutil.which("g++")
    if not gpp:
        return {
            "available": False,
            "name": None,
            "version": None,
            "reason": "g++ was not found on PATH. Install MinGW-w64 (g++) to run the CPU baseline locally.",
        }
    version = _first_line(_run([gpp, "--version"]))
    return {"available": True, "name": "g++", "version": version, "reason": ""}


def environment_info() -> dict:
    return {
        "os": platform.platform(),
        "python": platform.python_version(),
        "cpu_model": platform.processor() or platform.machine(),
        "local_cuda": detect_cuda(),
        "local_compiler": detect_compiler(),
    }
