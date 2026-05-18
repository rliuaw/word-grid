"""Pre-tagged BERT vocabulary for the single-POS generator."""

from __future__ import annotations

import random
import re
from functools import lru_cache
from typing import Iterator

import nltk
from transformers import AutoTokenizer

from word_grid.ml.bert_models import AVAILABLE_MODELS

_WORD_RE = re.compile(r"^[a-zA-Z][a-zA-Z'-]*$")

# Penn Treebank tags treated as punctuation for unmask filtering
PENN_PUNCT_TAGS = frozenset({
    ".",
    ",",
    ":",
    "''",
    "``",
    "#",
    "$",
    "-LRB-",
    "-RRB-",
    "HYPH",
    "NFP",
    "PUNCT",
    "LS",
    "SYM",
})


def _ensure_nltk() -> None:
    for pkg in ("punkt", "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng"):
        try:
            nltk.data.find(f"taggers/{pkg}" if "tagger" in pkg else f"tokenizers/{pkg}")
        except LookupError:
            nltk.download(pkg, quiet=True)


@lru_cache(maxsize=1)
def _load_vocab_tags(model_id: str = "bert-base-uncased") -> dict[str, set[str]]:
    """Tag every suitable tokenizer token with Penn Treebank tags."""
    _ensure_nltk()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    vocab: dict[str, set[str]] = {}
    for token in tokenizer.vocab:
        if not token or token.startswith("##"):
            continue
        word = token.replace("##", "")
        if not _WORD_RE.match(word):
            continue
        tags = {tag for _, tag in nltk.pos_tag([word.lower()])}
        vocab.setdefault(word.lower(), set()).update(tags)
    return vocab


def _tags_for_word(word: str, model_id: str) -> set[str]:
    w = word.strip().lower()
    if not w or not _WORD_RE.match(w):
        return set()
    vocab = _load_vocab_tags(model_id)
    if w in vocab:
        return vocab[w]
    _ensure_nltk()
    return {tag for _, tag in nltk.pos_tag([w])}


def is_punctuation_token(word: str, model_id: str = "bert-base-uncased") -> bool:
    """True if *word* is punctuation by form or Penn tag."""
    w = word.strip().lower()
    if not w:
        return True
    if not _WORD_RE.match(w):
        return True
    tags = _tags_for_word(w, model_id)
    return bool(tags) and tags <= PENN_PUNCT_TAGS


def word_matches_pos(word: str, pos: str, model_id: str = "bert-base-uncased") -> bool:
    """True if *word* is a non-punct token whose Penn tag(s) include *pos*."""
    w = word.strip().lower()
    if not w or is_punctuation_token(w, model_id):
        return False
    pos = pos.upper()
    return pos in _tags_for_word(w, model_id)


def iter_words_for_pos(pos: str, model_id: str = "bert-base-uncased") -> Iterator[str]:
    pos = pos.upper()
    vocab = _load_vocab_tags(model_id)
    for word, tags in vocab.items():
        if pos in tags:
            yield word


def random_word_for_pos(pos: str, model_id: str = "bert-base-uncased") -> str | None:
    words = list(iter_words_for_pos(pos, model_id))
    if not words:
        return None
    return random.choice(words)


def warmup(model_id: str = "bert-base-uncased") -> int:
    """Pre-initialize vocab tags; returns vocabulary size."""
    return len(_load_vocab_tags(model_id))
