"""BERT unmasker — score top-K tokens for a masked sentence."""

from __future__ import annotations

from word_grid.ml.bert_models import get_fill_mask, mask_token_for
from word_grid.tools import storage


def _normalize_prediction(pred: dict) -> dict:
    return {
        "token": pred["token_str"],
        "word": pred["token_str"].strip(),
        "score": float(pred["score"]),
        "sequence": pred["sequence"],
    }


def _parse_fill_mask_output(raw: list) -> list[dict]:
    """Normalize pipeline output for one or more mask positions.

    A single ``[MASK]`` yields ``[{token_str, score, ...}, ...]``.
    Multiple masks yield ``[[...], [...]]`` — one ranked list per mask.
    """
    if not raw:
        return [{"index": 0, "results": []}]

    if isinstance(raw[0], list):
        return [
            {"index": idx, "results": [_normalize_prediction(r) for r in mask_preds]}
            for idx, mask_preds in enumerate(raw)
        ]

    return [{"index": 0, "results": [_normalize_prediction(r) for r in raw]}]


def unmask(
    sentence: str,
    model_id: str = "bert-base-uncased",
    top_k: int = 10,
    *,
    persist: bool = True,
    report=None,
) -> dict:
    mask = mask_token_for(model_id)
    if mask not in sentence:
        raise ValueError(f"Sentence must contain mask token {mask!r}")
    pipe = get_fill_mask(model_id, report=report)
    if report:
        report(85, "Scoring masked tokens…", "inference")
    raw = pipe(sentence, top_k=top_k)
    masks = _parse_fill_mask_output(raw)
    mask_count = len(masks)

    payload = {
        "sentence": sentence,
        "model_id": model_id,
        "top_k": top_k,
        "mask_count": mask_count,
        "masks": masks,
        # Single-mask sentences keep flat ``results`` for older callers.
        "results": masks[0]["results"] if mask_count == 1 else None,
    }
    if persist:
        payload["result_id"] = storage.save_result("unmasker", payload, label=sentence[:40])
    return payload
