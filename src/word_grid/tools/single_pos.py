"""Single POS generator — random BERT-vocab word for a POS constraint."""

from __future__ import annotations

from word_grid.ml.pos_vocab import random_word_for_pos, warmup
from word_grid.tools import storage


def generate(
    pos: str,
    model_id: str = "bert-base-uncased",
    *,
    persist: bool = True,
    report=None,
) -> dict:
    if report:
        from word_grid.ml.progress import warmup_with_progress

        warmup_with_progress(model_id, report)
    else:
        warmup(model_id)
    word = random_word_for_pos(pos, model_id)
    if word is None:
        raise ValueError(f"No vocabulary token found for POS tag: {pos}")
    payload = {"pos": pos.upper(), "word": word, "model_id": model_id}
    if persist:
        payload["result_id"] = storage.save_result("single_pos", payload, label=pos.upper())
    return payload
