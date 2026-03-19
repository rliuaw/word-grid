"""Grid data structure, parsing, and sentence extraction for NxN word grids."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def strip_thinking_tokens(text: str) -> str:
    """Remove ``[THINK]…[/THINK]`` reasoning blocks produced by Magistral."""
    return re.sub(r"\[THINK\].*?\[/THINK\]", "", text, flags=re.DOTALL).strip()


def _extract_fenced_block(text: str) -> str | None:
    """Return the content of the first fenced code block, or ``None``.

    Matches both bare ````` ``` ````` and language-tagged ````` ```txt ````` blocks.
    """
    m = re.search(r"```\w*\s*\n(.*?)```", text, flags=re.DOTALL)
    return m.group(1).strip() if m else None


# ------------------------------------------------------------------
# WordGrid
# ------------------------------------------------------------------

@dataclass
class WordGrid:
    """An NxN grid of word tokens.

    Each cell is a single token that may include trailing punctuation
    (e.g. ``"Lost?"``).  Sentences are formed by joining cells with
    spaces.
    """

    cells: list[list[str]]
    n: int = field(init=False)

    def __post_init__(self) -> None:
        self.n = len(self.cells)
        for idx, row in enumerate(self.cells):
            if len(row) != self.n:
                raise ValueError(
                    f"Row {idx} has {len(row)} words; expected {self.n}"
                )

    # ---- sentences ------------------------------------------------

    def row_sentence(self, i: int) -> str:
        """Return the sentence formed by the *i*-th row (0-indexed)."""
        return " ".join(self.cells[i])

    def col_sentence(self, j: int) -> str:
        """Return the sentence formed by the *j*-th column (0-indexed)."""
        return " ".join(self.cells[i][j] for i in range(self.n))

    def all_sentences(self) -> list[tuple[str, str]]:
        """Return ``[(label, sentence), ...]`` for every row and column.

        Labels follow the notation from the problem statement:
        ``SR_1 … SR_N`` for rows, ``SC_1 … SC_N`` for columns.
        """
        sentences: list[tuple[str, str]] = []
        for i in range(self.n):
            sentences.append((f"SR_{i + 1}", self.row_sentence(i)))
        for j in range(self.n):
            sentences.append((f"SC_{j + 1}", self.col_sentence(j)))
        return sentences

    # ---- display --------------------------------------------------

    def format_grid(self) -> str:
        """Pretty-print the grid with aligned columns."""
        col_widths = [
            max(len(self.cells[i][j]) for i in range(self.n))
            for j in range(self.n)
        ]
        lines: list[str] = []
        for i in range(self.n):
            row_str = "  ".join(
                self.cells[i][j].ljust(col_widths[j])
                for j in range(self.n)
            )
            lines.append(row_str)
        return "\n".join(lines)

    def __repr__(self) -> str:  # noqa: D105
        return f"WordGrid(n={self.n})\n{self.format_grid()}"


# ------------------------------------------------------------------
# Parsing
# ------------------------------------------------------------------

def _clean_line(line: str) -> list[str]:
    """Strip common formatting prefixes and split into words."""
    s = line.strip()
    # Remove leading numbering  ("1. ", "1) ", "#1 ")
    s = re.sub(r"^[#]?\d+[.\)]\s*", "", s)
    # Remove bullet points
    s = re.sub(r"^[-*•]\s*", "", s)
    # Remove surrounding table pipes
    s = re.sub(r"^\||\|$", "", s).strip()
    # Remove markdown bold/italic
    s = re.sub(r"[*_]{1,2}", "", s)
    return s.split()


def _parse_lines(lines: list[str], n: int) -> list[list[str]]:
    """Scan *lines* for *n* rows that each contain exactly *n* tokens."""
    grid_rows: list[list[str]] = []
    for line in lines:
        words = _clean_line(line)
        if len(words) == n:
            grid_rows.append(words)
        if len(grid_rows) == n:
            break
    return grid_rows


def parse_grid_from_text(text: str, n: int = 5) -> Optional[WordGrid]:
    """Best-effort parse of an NxN word grid from free-form model output.

    The parser first looks for a fenced code block (delimited by triple
    backticks) and extracts the grid from it.  If no fenced block is
    found, it falls back to scanning every line of the output for *n*
    lines that each contain exactly *n* whitespace-separated tokens.

    Both paths tolerate numbering, bullets, markdown pipes, and
    ``[THINK]`` blocks.

    Parameters
    ----------
    text:
        Raw model output (may include reasoning / preamble).
    n:
        Expected grid dimension.

    Returns
    -------
    WordGrid | None
        A validated ``WordGrid`` or ``None`` if parsing fails.
    """
    text = strip_thinking_tokens(text)

    # --- Strategy 1: extract from fenced code block ---
    fenced = _extract_fenced_block(text)
    if fenced is not None:
        grid_rows = _parse_lines(fenced.splitlines(), n)
        if len(grid_rows) == n:
            try:
                return WordGrid(cells=grid_rows)
            except (ValueError, IndexError):
                pass  # fall through to full-text scan

    # No fenced block found (or it didn't contain a valid grid).
    return None


# ------------------------------------------------------------------
# Convenience
# ------------------------------------------------------------------

def score_report(
    grid: WordGrid,
    scores: dict[str, float],
) -> str:
    """Return a human-readable scoring report.

    Parameters
    ----------
    grid:
        The word grid.
    scores:
        Mapping ``{label: syntax_score}`` for each sentence label.
    """
    lines: list[str] = [grid.format_grid(), ""]
    total = 0.0
    count = 0
    for label, sentence in grid.all_sentences():
        s = scores.get(label, 0.0)
        total += s
        count += 1
        lines.append(f"  {label}: {sentence!r:50s}  Syntax = {s:.1f}")
    avg = total / count if count else 0.0
    lines.append(f"\n  Grid score (avg): {avg:.2f}")
    return "\n".join(lines)
