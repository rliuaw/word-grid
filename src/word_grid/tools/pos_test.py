"""Single POS combination testing — random sentence for a POS sequence."""

from __future__ import annotations

from word_grid.ml.pos_vocab import random_word_for_pos, warmup
from word_grid.pos.tags import parse_pos_sequence
from word_grid.tools import storage


def generate(
    pos_sequence: str,
    model_id: str = "bert-base-uncased",
    *,
    persist: bool = True,
    report=None,
) -> dict:
    tags = parse_pos_sequence(pos_sequence)
    if not tags:
        raise ValueError("Empty POS sequence")
    if report:
        from word_grid.ml.progress import warmup_with_progress

        warmup_with_progress(model_id, report)
    else:
        warmup(model_id)
    words: list[str] = []
    for tag in tags:
        w = random_word_for_pos(tag, model_id)
        if w is None:
            w = f"<{tag}>"
        words.append(w)
    sentence = " ".join(words)
    payload = {
        "pos_sequence": list(tags),
        "sentence": sentence,
        "words": words,
        "model_id": model_id,
    }
    if persist:
        payload["result_id"] = storage.save_result(
            "pos_test",
            payload,
            label=" ".join(tags),
        )
    return payload
