"""POS combination generators (algorithms #1–#3) → dictionary.txt."""

from __future__ import annotations

import random
from collections import Counter
from typing import Callable

from word_grid.pos.tags import NONTERMINAL_TO_POS, RULE_GRAMMAR, sequence_to_dict_line
from word_grid.tools import storage

ReportFn = Callable[[float, str, str], None]
ComboWithCount = tuple[tuple[str, ...], int]
AlgorithmFn = Callable[..., list[ComboWithCount]]


def _wikitext_sentences(max_sentences: int, report: ReportFn | None = None) -> list[str]:
    from datasets import load_dataset

    if report:
        report(5, "Loading Wikitext dataset…", "dataset")
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    texts: list[str] = []
    for row in ds:
        line = (row.get("text") or "").strip()
        if len(line) > 20 and not line.startswith("="):
            texts.append(line)
        if len(texts) >= max_sentences:
            break
    if report:
        report(15, f"Loaded {len(texts)} sentences from Wikitext", "dataset")
    return texts


def _get_nlp(report: ReportFn | None = None):
    import spacy

    if report:
        report(18, "Loading spaCy model…", "spacy")
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        from spacy.cli import download

        if report:
            report(20, "Downloading en_core_web_sm…", "spacy")
        download("en_core_web_sm")
        return spacy.load("en_core_web_sm")


def _spacy_to_penn(tag: str) -> str:
    mapping = {
        "NOUN": "NN",
        "PROPN": "NNP",
        "VERB": "VB",
        "ADJ": "JJ",
        "ADV": "RB",
        "ADP": "IN",
        "DET": "DT",
        "PRON": "PRP",
        "CCONJ": "CC",
        "SCONJ": "IN",
        "PART": "RP",
        "NUM": "CD",
        "INTJ": "UH",
        "AUX": "VBZ",
    }
    return mapping.get(tag, tag[:2].upper() if tag else "XX")


def algorithm_tagger(
    n: int,
    k: int,
    *,
    max_sentences: int = 2000,
    report: ReportFn | None = None,
) -> list[ComboWithCount]:
    """Algorithm #1 — most common POS n-grams from Wikitext via spaCy."""
    texts = _wikitext_sentences(max_sentences, report)
    nlp = _get_nlp(report)
    counter: Counter[tuple[str, ...]] = Counter()
    total = len(texts)
    for idx, text in enumerate(texts):
        doc = nlp(text)
        tags = [_spacy_to_penn(t.pos_) for t in doc if not t.is_space and not t.is_punct]
        for i in range(len(tags) - n + 1):
            counter[tuple(tags[i : i + n])] += 1
        if report and total and idx % max(1, total // 50) == 0:
            pct = 20 + 65 * (idx + 1) / total
            report(pct, f"Tagging sentences ({idx + 1}/{total})…", "tagging")
    if report:
        report(88, f"Found {len(counter):,} unique combinations", "ranking")
    return counter.most_common(k)


HOMOGRAPHS: dict[str, list[str]] = {
    "read": ["VB", "VBD", "NN"],
    "lead": ["VB", "NN"],
    "close": ["VB", "JJ", "NN"],
    "record": ["NN", "VB"],
    "present": ["JJ", "NN", "VB"],
    "object": ["NN", "VB"],
    "content": ["NN", "JJ"],
    "refuse": ["VB", "NN"],
    "produce": ["VB", "NN"],
    "wind": ["NN", "VB"],
    "tear": ["NN", "VB"],
    "bow": ["NN", "VB"],
    "row": ["NN", "VB"],
}


def algorithm_tagger_homographs(
    n: int,
    k: int,
    *,
    max_sentences: int = 2000,
    report: ReportFn | None = None,
) -> list[ComboWithCount]:
    """Algorithm #2 — tagger with homograph-aware tag expansion."""
    texts = _wikitext_sentences(max_sentences, report)
    nlp = _get_nlp(report)
    counter: Counter[tuple[str, ...]] = Counter()

    def tag_variants(token) -> list[str]:
        lower = token.text.lower()
        if lower in HOMOGRAPHS:
            return HOMOGRAPHS[lower]
        return [_spacy_to_penn(token.pos_)]

    def expand(pos_lists: list[list[str]]) -> list[list[str]]:
        if not pos_lists:
            return [[]]
        first, *rest = pos_lists
        tail = expand(rest)
        out: list[list[str]] = []
        for t in first:
            for r in tail:
                out.append([t, *r])
        return out

    total = len(texts)
    for idx, text in enumerate(texts):
        doc = nlp(text)
        tokens = [t for t in doc if not t.is_space and not t.is_punct]
        if len(tokens) < n:
            continue
        variant_lists = [tag_variants(t) for t in tokens]
        all_tag_seqs = expand(variant_lists)
        for tags in all_tag_seqs:
            if len(tags) < n:
                continue
            for i in range(len(tags) - n + 1):
                counter[tuple(tags[i : i + n])] += 1
        if report and total and idx % max(1, total // 50) == 0:
            pct = 20 + 65 * (idx + 1) / total
            report(pct, f"Tagging with homographs ({idx + 1}/{total})…", "tagging")
    if report:
        report(88, f"Found {len(counter):,} unique combinations", "ranking")
    return counter.most_common(k)


def _expand_nonterminal(name: str) -> list[list[str]]:
    if name in NONTERMINAL_TO_POS:
        return [[t] for t in NONTERMINAL_TO_POS[name]]
    productions = RULE_GRAMMAR.get(name, [[name]])
    sequences: list[list[str]] = []
    for prod in productions:
        symbol_options: list[list[list[str]]] = [
            _expand_nonterminal(sym) for sym in prod
        ]
        acc: list[list[str]] = [[]]
        for options in symbol_options:
            nxt: list[list[str]] = []
            for prefix in acc:
                for option in options:
                    nxt.append(prefix + option)
            acc = nxt
        sequences.extend(acc)
    return sequences


def algorithm_rule_based(
    n: int, k: int, *, report: ReportFn | None = None
) -> list[ComboWithCount]:
    if report:
        report(30, "Expanding grammar rules…", "rules")
    candidates: list[tuple[str, ...]] = []
    for start in ("Root", "Sentence"):
        for seq in _expand_nonterminal(start):
            if len(seq) == n:
                candidates.append(tuple(seq))
    seen: set[tuple[str, ...]] = set()
    unique: list[tuple[str, ...]] = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    random.shuffle(unique)
    if report:
        report(80, f"Generated {len(unique[:k])} rule-based combinations", "rules")
    return [(c, 0) for c in unique[:k]]


ALGORITHMS: dict[str, tuple[str, AlgorithmFn]] = {
    "tagger": ("Tagger (Wikitext + spaCy)", algorithm_tagger),
    "tagger_homographs": ("Tagger + homographs", algorithm_tagger_homographs),
    "rule_based": ("Rule-based CFG", algorithm_rule_based),
}

TAGGER_ALGORITHMS = frozenset({"tagger", "tagger_homographs"})


def combinations_to_dictionary(combos: list[tuple[str, ...]]) -> str:
    return "\n".join(sequence_to_dict_line(c) for c in combos) + "\n"


def _combo_entries(
    ranked: list[ComboWithCount], *, include_occurrences: bool
) -> list[dict]:
    entries: list[dict] = []
    for combo, count in ranked:
        entry: dict = {
            "sequence": list(combo),
            "display": " ".join(combo),
        }
        if include_occurrences:
            entry["occurrences"] = count
        entries.append(entry)
    return entries


def generate(
    n: int,
    k: int,
    algorithm: str,
    *,
    persist: bool = True,
    max_sentences: int = 2000,
    report: ReportFn | None = None,
) -> dict:
    if algorithm not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    label, fn = ALGORITHMS[algorithm]
    include_occurrences = algorithm in TAGGER_ALGORITHMS

    if report:
        report(2, f"Starting {label}…", "init")

    if algorithm == "rule_based":
        ranked = fn(n, k, report=report)
    else:
        ranked = fn(n, k, max_sentences=max_sentences, report=report)

    combos = [c for c, _ in ranked]
    entries = _combo_entries(ranked, include_occurrences=include_occurrences)
    dictionary = combinations_to_dictionary(combos)

    if report:
        report(95, "Saving results…", "save")

    payload = {
        "n": n,
        "k": k,
        "algorithm": algorithm,
        "algorithm_label": label,
        "count": len(entries),
        "has_occurrences": include_occurrences,
        "combinations": entries,
        "dictionary_preview": dictionary[:8000],
    }
    if persist:
        result_id = storage.save_result(
            "pos_combinations",
            {
                **payload,
                "combinations": entries,
                "combo_tuples": [list(c) for c in combos],
            },
            label=f"{algorithm} N={n} K={len(entries)}",
        )
        storage.save_artifact(result_id, "dictionary.txt", dictionary)
        payload["result_id"] = result_id
        payload["dictionary_path"] = str(
            storage.artifact_path(result_id, "dictionary.txt")
        )
    return payload
