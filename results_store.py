"""Stores result documents as JSON files under backend/data/results/."""

from __future__ import annotations

import json
import re
import secrets
from datetime import datetime, timezone
from typing import Optional

from .config import RESULTS_DIR
from .schemas import ResultsDocument

ID_PATTERN = re.compile(r"^\d{8}T\d{6}Z-[a-f0-9]{8}$")


def new_result_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{secrets.token_hex(4)}"


def save_document(doc: ResultsDocument) -> str:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result_id = new_result_id()
    path = RESULTS_DIR / f"{result_id}.json"
    path.write_text(json.dumps(doc.to_json_dict(), indent=2), encoding="utf-8")
    return result_id


def load_document(result_id: str) -> Optional[dict]:
    if not ID_PATTERN.fullmatch(result_id):
        raise ValueError("invalid result id")
    path = RESULTS_DIR / f"{result_id}.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def list_documents() -> list[dict]:
    if not RESULTS_DIR.is_dir():
        return []
    summaries: list[dict] = []
    for path in sorted(RESULTS_DIR.glob("*.json"), reverse=True):
        if not ID_PATTERN.fullmatch(path.stem):
            continue
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
            runs = doc.get("runs", [])
            env = doc.get("environment", {}) or {}
            device = env.get("gpu_name") or (runs[0].get("device") if runs else None) or "unknown"
            summaries.append(
                {
                    "id": path.stem,
                    "source": doc.get("source", "unknown"),
                    "generated_at": doc.get("generated_at", ""),
                    "device": device,
                    "kernels": sorted({r.get("kernel") for r in runs}),
                    "run_count": len(runs),
                }
            )
        except (OSError, json.JSONDecodeError, AttributeError):
            continue
    return summaries
