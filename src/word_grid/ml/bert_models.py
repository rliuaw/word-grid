"""BERT / RoBERTa / DistilBERT fill-mask pipelines (Mac-friendly)."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Callable

import torch
from transformers import pipeline

AVAILABLE_MODELS: list[dict[str, str]] = [
    {
        "id": "bert-base-uncased",
        "label": "BERT Base",
        "mask_token": "[MASK]",
    },
    {
        "id": "roberta-base",
        "label": "RoBERTa Base",
        "mask_token": "<mask>",
    },
    {
        "id": "distilbert-base-uncased",
        "label": "DistilBERT Base",
        "mask_token": "[MASK]",
    },
]

_MODEL_IDS = {m["id"] for m in AVAILABLE_MODELS}
ReportFn = Callable[[float, str, str], None]


def get_device() -> int | str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return 0
    return -1


@lru_cache(maxsize=3)
def _cached_fill_mask(model_id: str) -> Any:
    device = get_device()
    return pipeline(
        "fill-mask",
        model=model_id,
        device=device,
        top_k=50,
    )


def get_fill_mask(model_id: str, report: ReportFn | None = None) -> Any:
    if model_id not in _MODEL_IDS:
        raise ValueError(f"Unknown model: {model_id}")
    if report:
        report(15, f"Loading {model_id} (download if needed)…", "model")
        report(55, "Building fill-mask pipeline…", "pipeline")
    pipe = _cached_fill_mask(model_id)
    if report:
        report(75, "Model ready", "ready")
    return pipe


def mask_token_for(model_id: str) -> str:
    for m in AVAILABLE_MODELS:
        if m["id"] == model_id:
            return m["mask_token"]
    return "[MASK]"
