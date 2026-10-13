"""Paths and constants shared by the backend."""

import os
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = BACKEND_DIR.parent
CUDA_DIR = PROJECT_DIR / "cuda"

# Both can be overridden (the tests use temporary directories).
BUILD_DIR = Path(os.environ.get("CUDA_MATRIXLAB_BUILD_DIR", BACKEND_DIR / ".build"))
DATA_DIR = Path(os.environ.get("CUDA_MATRIXLAB_DATA_DIR", BACKEND_DIR / "data"))
RESULTS_DIR = DATA_DIR / "results"

MAX_UPLOAD_BYTES = 5 * 1024 * 1024
RESULTS_SCHEMA = "cuda-matrixlab-results/v1"
