"""Gallery of filled word grids."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from word_grid.tools import storage

GALLERY_INDEX = storage.RESULTS_DIR / "gallery_index.json"


def _load_gallery() -> list[dict[str, Any]]:
    storage._ensure_dir()
    if not GALLERY_INDEX.exists():
        return []
    return json.loads(GALLERY_INDEX.read_text(encoding="utf-8"))


def _save_gallery(entries: list[dict[str, Any]]) -> None:
    storage._ensure_dir()
    GALLERY_INDEX.write_text(json.dumps(entries, indent=2), encoding="utf-8")


def save_grid(
    cells: list[list[str]],
    *,
    metadata: dict[str, Any] | None = None,
    steps: list[dict] | None = None,
    benchmark_score: float | None = None,
    special: bool = False,
) -> str:
    meta = dict(metadata or {})
    meta.setdefault("created_at", datetime.now(timezone.utc).isoformat())
    if benchmark_score is not None:
        meta["benchmark_score"] = benchmark_score
    if special:
        meta["special"] = True
    payload = {
        "cells": cells,
        "n": len(cells),
        "metadata": meta,
        "steps": steps,
    }
    return storage.save_result("gallery", payload, label=f"{len(cells)}x{len(cells)} grid")


def list_grids() -> list[dict[str, Any]]:
    return storage.list_results("gallery")


def get_grid(gallery_id: str) -> dict[str, Any]:
    return storage.load_result(gallery_id)


def update_grid(gallery_id: str, **updates: Any) -> dict[str, Any]:
    record = storage.load_result(gallery_id)
    payload = record["payload"]
    if "benchmark_score" in updates:
        payload["metadata"]["benchmark_score"] = updates["benchmark_score"]
    if "special" in updates:
        payload["metadata"]["special"] = bool(updates["special"])
    if "tags" in updates:
        payload["metadata"]["tags"] = updates["tags"]
    path = storage.RESULTS_DIR / f"{gallery_id}.json"
    record["payload"] = payload
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record
