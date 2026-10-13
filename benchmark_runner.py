"""Compiles and runs the real benchmark programs in ../cuda on this machine.

CUDA kernels are refused (HTTP 409) unless local CUDA is detected. The CPU
baseline needs g++. Each program prints one JSON line, which is parsed and
validated against the shared results schema.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import HTTPException

from .config import BUILD_DIR, CUDA_DIR, RESULTS_SCHEMA
from .cuda_detect import environment_info
from .schemas import ResultsDocument

MAX_REPS = 20

PROGRAMS: dict[str, dict] = {
    "cpu_matmul": {
        "source": "matrix_cpu.cpp",
        "toolchain": "cpp",
        "sizes": [128, 256, 512, 1024, 2048],
        "default_reps": 3,
    },
    "cuda_naive_matmul": {
        "source": "matrix_cuda_naive.cu",
        "toolchain": "cuda",
        "sizes": [128, 256, 512, 1024, 2048, 4096],
        "default_reps": 5,
    },
    "cuda_shared_matmul": {
        "source": "matrix_cuda_shared.cu",
        "toolchain": "cuda",
        "sizes": [128, 256, 512, 1024, 2048, 4096],
        "default_reps": 5,
    },
    "cuda_vector_add": {
        "source": "vector_add.cu",
        "toolchain": "cuda",
        "sizes": [2**20, 2**22, 2**24, 2**26],
        "default_reps": 10,
    },
}


def _exe_path(source: str) -> Path:
    stem = Path(source).stem
    name = stem + (".exe" if os.name == "nt" else "")
    return BUILD_DIR / name


def _compile(kernel: str, compute_capability: Optional[str]) -> Path:
    spec = PROGRAMS[kernel]
    src = CUDA_DIR / spec["source"]
    if not src.is_file():
        raise HTTPException(500, detail=f"Missing source file: {src}")

    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    exe = _exe_path(spec["source"])

    if spec["toolchain"] == "cuda":
        compiler = shutil.which("nvcc")
        if compute_capability:
            arch = f"-arch=sm_{compute_capability.replace('.', '')}"
        else:
            arch = "-arch=native"
        cmd = [compiler, "-O3", "-std=c++17", arch, "-I", str(CUDA_DIR), str(src), "-o", str(exe)]
    else:
        compiler = shutil.which("g++")
        cmd = [compiler, "-O3", "-std=c++17", "-I", str(CUDA_DIR), str(src), "-o", str(exe)]

    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        raise HTTPException(500, detail=f"Compilation failed for {spec['source']}:\n{proc.stderr[-4000:]}")
    return exe


def _run_program(exe: Path, n: int, reps: int) -> dict:
    try:
        proc = subprocess.run(
            [str(exe), str(n), str(reps)],
            capture_output=True,
            text=True,
            timeout=1800,
        )
    except subprocess.TimeoutExpired:
        raise HTTPException(504, detail=f"{exe.name} timed out at n={n}")
    if proc.returncode != 0:
        raise HTTPException(500, detail=f"{exe.name} failed at n={n}:\n{proc.stderr[-2000:]}")
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    if not lines:
        raise HTTPException(500, detail=f"{exe.name} produced no output at n={n}")
    try:
        return json.loads(lines[-1])
    except json.JSONDecodeError:
        raise HTTPException(500, detail=f"{exe.name} printed invalid JSON at n={n}")


def run_local_benchmark(kernel: str, sizes: Optional[list[int]], reps: Optional[int]) -> ResultsDocument:
    if kernel not in PROGRAMS:
        raise HTTPException(400, detail=f"Unknown kernel '{kernel}'")

    spec = PROGRAMS[kernel]
    allowed = set(spec["sizes"])
    chosen = sizes if sizes else list(spec["sizes"])
    if not chosen or any(s not in allowed for s in chosen):
        raise HTTPException(400, detail=f"sizes must be a non-empty subset of {spec['sizes']}")
    reps_value = reps if reps else spec["default_reps"]
    if not 1 <= reps_value <= MAX_REPS:
        raise HTTPException(400, detail=f"reps must be between 1 and {MAX_REPS}")

    env = environment_info()
    cuda = env["local_cuda"]
    compiler = env["local_compiler"]

    if spec["toolchain"] == "cuda":
        if not cuda["available"]:
            raise HTTPException(409, detail=f"Local CUDA unavailable: {cuda['reason']}")
        source_label = "local-cuda"
    else:
        if not compiler["available"]:
            raise HTTPException(409, detail=compiler["reason"])
        source_label = "local-cpu"

    exe = _compile(kernel, cuda["compute_capability"])
    runs = [_run_program(exe, n, reps_value) for n in chosen]

    document = {
        "schema": RESULTS_SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": source_label,
        "environment": {
            "os": env["os"],
            "python": env["python"],
            "cpu_model": env["cpu_model"],
            "gpu_name": cuda["gpu_name"],
            "driver_version": cuda["driver_version"],
            "compute_capability": cuda["compute_capability"],
            "cuda_version": cuda["cuda_version"],
            "compiler": compiler["version"],
        },
        "runs": runs,
    }
    try:
        return ResultsDocument.model_validate(document)
    except Exception as exc:  # pragma: no cover - indicates a program/schema mismatch
        raise HTTPException(500, detail=f"Program output did not match the results schema: {exc}")
