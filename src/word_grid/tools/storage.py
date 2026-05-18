"""Persist tool outputs under ``results/`` for cross-run reference."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = _PROJECT_ROOT / "results"
INDEX_FILE = RESULTS_DIR / "index.json"


def _ensure_dir() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _load_index() -> list[dict[str, Any]]:
    _ensure_dir()
    if not INDEX_FILE.exists():
        return []
    return json.loads(INDEX_FILE.read_text(encoding="utf-8"))


def _save_index(entries: list[dict[str, Any]]) -> None:
    _ensure_dir()
    INDEX_FILE.write_text(json.dumps(entries, indent=2), encoding="utf-8")


def save_result(
    tool: str,
    payload: dict[str, Any],
    *,
    label: str | None = None,
    refs: list[str] | None = None,
) -> str:
    """Write payload to ``results/<id>.json`` and update the index."""
    _ensure_dir()
    result_id = uuid.uuid4().hex[:12]
    created_at = datetime.now(timezone.utc).isoformat()
    record = {
        "id": result_id,
        "tool": tool,
        "label": label,
        "created_at": created_at,
        "refs": refs or [],
        "payload": payload,
    }
    path = RESULTS_DIR / f"{result_id}.json"
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")

    entries = _load_index()
    entries.append(
        {
            "id": result_id,
            "tool": tool,
            "label": label,
            "created_at": created_at,
            "refs": refs or [],
        }
    )
    _save_index(entries)
    return result_id


def load_result(result_id: str) -> dict[str, Any]:
    path = RESULTS_DIR / f"{result_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Result not found: {result_id}")
    return json.loads(path.read_text(encoding="utf-8"))


def list_results(tool: str | None = None) -> list[dict[str, Any]]:
    entries = _load_index()
    if tool:
        entries = [e for e in entries if e["tool"] == tool]
    return sorted(entries, key=lambda e: e["created_at"], reverse=True)


def save_artifact(result_id: str, filename: str, content: str) -> Path:
    """Save a sidecar file (e.g. dictionary.txt) next to the result."""
    _ensure_dir()
    art_dir = RESULTS_DIR / result_id
    art_dir.mkdir(parents=True, exist_ok=True)
    path = art_dir / filename
    path.write_text(content, encoding="utf-8")
    return path


def artifact_path(result_id: str, filename: str) -> Path:
    return RESULTS_DIR / result_id / filename
