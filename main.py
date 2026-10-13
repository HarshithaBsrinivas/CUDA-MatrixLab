"""CUDA MatrixLab API.

Run from the backend folder:
    uvicorn app.main:app --reload --port 8000
"""

from __future__ import annotations

import json
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ValidationError

from .benchmark_runner import MAX_REPS, PROGRAMS, run_local_benchmark
from .config import MAX_UPLOAD_BYTES
from .cuda_detect import environment_info
from .results_store import list_documents, load_document, save_document
from .schemas import ResultsDocument

app = FastAPI(title="CUDA MatrixLab API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class LocalBenchmarkRequest(BaseModel):
    kernel: str
    sizes: Optional[list[int]] = None
    reps: Optional[int] = Field(default=None, ge=1, le=MAX_REPS)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/environment")
def environment() -> dict:
    """Reports whether local CUDA and the g++ compiler are usable on this machine."""
    return environment_info()


@app.get("/api/kernels")
def kernels() -> dict:
    return {
        name: {
            "toolchain": spec["toolchain"],
            "sizes": spec["sizes"],
            "default_reps": spec["default_reps"],
        }
        for name, spec in PROGRAMS.items()
    }


@app.post("/api/benchmarks/local")
def local_benchmark(req: LocalBenchmarkRequest) -> dict:
    """Compile and run a benchmark on this machine. CUDA kernels need local CUDA."""
    if req.kernel not in PROGRAMS:
        raise HTTPException(400, detail=f"Unknown kernel '{req.kernel}'")
    doc = run_local_benchmark(req.kernel, req.sizes, req.reps)
    result_id = save_document(doc)
    return {"id": result_id, "document": doc.to_json_dict()}


@app.post("/api/results/import")
async def import_results(file: UploadFile = File(...)) -> dict:
    """Validate and store a results JSON file exported from the Colab notebook."""
    raw = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, detail="File is larger than 5 MB")
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise HTTPException(400, detail="File is not valid UTF-8 JSON")
    try:
        doc = ResultsDocument.model_validate(data)
    except ValidationError as exc:
        raise HTTPException(422, detail=json.loads(exc.json()))
    result_id = save_document(doc)
    return {"id": result_id, "document": doc.to_json_dict()}


@app.get("/api/results")
def results_list() -> list[dict]:
    return list_documents()


@app.get("/api/results/{result_id}")
def results_get(result_id: str) -> dict:
    try:
        doc = load_document(result_id)
    except ValueError:
        raise HTTPException(400, detail="Invalid result id")
    if doc is None:
        raise HTTPException(404, detail="Result not found")
    return doc
