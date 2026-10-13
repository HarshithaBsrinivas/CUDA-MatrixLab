"""Backend tests. Run from the backend folder:  python -m pytest -q"""

import copy
import json
import os
import tempfile

# Keep test output out of the real data/ and .build/ folders.
os.environ["CUDA_MATRIXLAB_DATA_DIR"] = tempfile.mkdtemp(prefix="cmlab-data-")
os.environ["CUDA_MATRIXLAB_BUILD_DIR"] = tempfile.mkdtemp(prefix="cmlab-build-")

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402
from app.cuda_detect import detect_compiler, detect_cuda  # noqa: E402

client = TestClient(app)

# Structurally valid schema input. Numbers are placeholders for testing the
# validator only; they are NOT benchmark results.
SCHEMA_TEST_DOC = {
    "schema": "cuda-matrixlab-results/v1",
    "generated_at": "2026-01-01T00:00:00+00:00",
    "source": "schema-test-fixture",
    "environment": {"gpu_name": "SCHEMA-TEST"},
    "runs": [
        {
            "kernel": "cuda_shared_matmul",
            "n": 256,
            "reps": 2,
            "device": "SCHEMA-TEST",
            "compute_capability": "7.5",
            "time_scope": "kernel_only_cuda_events",
            "times_ms": [1.0, 1.2],
            "median_ms": 1.1,
            "min_ms": 1.0,
            "metric": {"type": "gflops", "value": 30.0},
            "verified": True,
            "max_abs_error": 0.0,
        }
    ],
}


def _upload(payload: bytes):
    return client.post(
        "/api/results/import",
        files={"file": ("results.json", payload, "application/json")},
    )


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_environment_reports_cuda_status_honestly():
    body = client.get("/api/environment").json()
    cuda = body["local_cuda"]
    assert isinstance(cuda["available"], bool)
    assert isinstance(cuda["reason"], str)
    if not cuda["available"]:
        assert cuda["reason"], "an unavailable CUDA status must explain why"


def test_kernels_list_contains_all_programs():
    body = client.get("/api/kernels").json()
    assert set(body) == {
        "cpu_matmul",
        "cuda_naive_matmul",
        "cuda_shared_matmul",
        "cuda_vector_add",
    }


def test_import_valid_document_round_trips():
    r = _upload(json.dumps(SCHEMA_TEST_DOC).encode())
    assert r.status_code == 200, r.text
    result_id = r.json()["id"]
    got = client.get(f"/api/results/{result_id}")
    assert got.status_code == 200
    assert got.json()["schema"] == "cuda-matrixlab-results/v1"
    assert got.json()["runs"][0]["kernel"] == "cuda_shared_matmul"


def test_import_rejects_wrong_schema_version():
    doc = copy.deepcopy(SCHEMA_TEST_DOC)
    doc["schema"] = "something-else/v9"
    assert _upload(json.dumps(doc).encode()).status_code == 422


def test_import_rejects_unknown_kernel():
    doc = copy.deepcopy(SCHEMA_TEST_DOC)
    doc["runs"][0]["kernel"] = "not_a_kernel"
    assert _upload(json.dumps(doc).encode()).status_code == 422


def test_import_rejects_non_json():
    assert _upload(b"this is not json").status_code == 400


def test_results_list_includes_imported_document():
    _upload(json.dumps(SCHEMA_TEST_DOC).encode())
    listing = client.get("/api/results").json()
    assert any(item["source"] == "schema-test-fixture" for item in listing)


def test_invalid_result_id_is_rejected():
    assert client.get("/api/results/..%2Fsecrets").status_code in (400, 404)
    assert client.get("/api/results/not-a-valid-id").status_code == 400


def test_unknown_kernel_is_rejected_for_local_run():
    r = client.post("/api/benchmarks/local", json={"kernel": "nope"})
    assert r.status_code == 400


def test_cuda_kernel_is_refused_without_local_cuda():
    if detect_cuda()["available"]:
        return  # this machine can run CUDA; the refusal path does not apply
    r = client.post("/api/benchmarks/local", json={"kernel": "cuda_vector_add", "sizes": [1048576]})
    assert r.status_code == 409
    assert "Local CUDA unavailable" in r.json()["detail"]


def test_local_cpu_benchmark_end_to_end():
    if not detect_compiler()["available"]:
        return  # g++ not installed; the refusal path is covered below
    r = client.post(
        "/api/benchmarks/local",
        json={"kernel": "cpu_matmul", "sizes": [128], "reps": 1},
    )
    assert r.status_code == 200, r.text
    doc = r.json()["document"]
    assert doc["source"] == "local-cpu"
    run = doc["runs"][0]
    assert run["kernel"] == "cpu_matmul"
    assert run["n"] == 128
    assert run["verified"] is True
    assert run["median_ms"] > 0


def test_cpu_benchmark_refused_without_compiler():
    if detect_compiler()["available"]:
        return
    r = client.post("/api/benchmarks/local", json={"kernel": "cpu_matmul", "sizes": [128], "reps": 1})
    assert r.status_code == 409
