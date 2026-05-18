"""Penn Treebank POS helpers and rule-grammar definitions."""

from __future__ import annotations

DICT_SEP = "|"


def sequence_to_dict_line(tags: list[str]) -> str:
    """Encode a POS sequence as one dictionary.txt line."""
    return DICT_SEP.join(tags)


def parse_pos_sequence(line: str) -> tuple[str, ...]:
    """Decode a dictionary line or space-separated constraint."""
    line = line.strip()
    if DICT_SEP in line:
        return tuple(t for t in line.split(DICT_SEP) if t)
    return tuple(t for t in line.split() if t)


# Unmasker fill order: prepositions > verbs > nouns > rest
FILL_PRIORITY: dict[str, int] = {
    "IN": 0,
    "TO": 0,
    "RP": 0,
    "VB": 1,
    "VBD": 1,
    "VBG": 1,
    "VBN": 1,
    "VBP": 1,
    "VBZ": 1,
    "NN": 2,
    "NNS": 2,
    "NNP": 2,
    "NNPS": 2,
    "PRP": 2,
    "PRP$": 2,
}


def fill_priority(tag: str) -> int:
    return FILL_PRIORITY.get(tag, 10)


# Rule-based algorithm (#3) — expandable nonterminals → Penn tags
NONTERMINAL_TO_POS: dict[str, list[str]] = {
    "Noun": ["NN", "NNS", "NNP"],
    "Verb": ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"],
    "Adjective": ["JJ", "JJR", "JJS"],
    "Adverb": ["RB", "RBR", "RBS"],
    "Determiner": ["DT"],
    "Preposition": ["IN"],
    "Pronoun": ["PRP", "PRP$"],
    "Conjunction": ["CC"],
    "Modal": ["MD"],
}

RULE_GRAMMAR: dict[str, list[list[str]]] = {
    "Root": [["Noun Phrase", "Verb Phrase"], ["Pronoun", "Verb Phrase"]],
    "Noun Phrase": [
        ["Determiner", "Adjective", "Noun"],
        ["Adjective", "Noun"],
        ["Determiner", "Noun"],
        ["Noun"],
    ],
    "Verb Phrase": [
        ["Modal", "Verb"],
        ["Adverb", "Verb"],
        ["Verb", "Preposition"],
        ["Verb"],
    ],
    "Sentence": [["Root"]],
}
