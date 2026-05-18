"""Benchmarking helpers and Gemini scoring prompt."""

from __future__ import annotations

from word_grid.grid import WordGrid
from word_grid.tools import gallery


def grid_to_poems(cells: list[list[str]]) -> dict[str, str]:
    grid = WordGrid(cells=cells)
    horizontal = "\n".join(grid.row_sentence(i) for i in range(grid.n))
    vertical = "\n".join(grid.col_sentence(j) for j in range(grid.n))
    return {"horizontal_poem": horizontal, "vertical_poem": vertical}


GEMINI_PROMPT_TEMPLATE = """You are evaluating an N×N word grid puzzle. Each row reads left-to-right as a sentence; each column reads top-to-bottom as a sentence.

Score the grid on three factors from 0 to 10:
1. **Syntax** — grammatical correctness of all sentences
2. **Legibility** — clarity and readability
3. **Artistic value** — creativity, cohesion, and poetic quality

## Grid ({n}×{n})

```
{grid_text}
```

## Horizontal poem (rows)

{horizontal_poem}

## Vertical poem (columns)

{vertical_poem}

Respond in JSON:
{{"syntax": <0-10>, "legibility": <0-10>, "artistic_value": <0-10>, "notes": "<brief explanation>"}}
"""


def gemini_prompt(gallery_id: str) -> dict:
    record = gallery.get_grid(gallery_id)
    cells = record["payload"]["cells"]
    grid = WordGrid(cells=cells)
    poems = grid_to_poems(cells)
    prompt = GEMINI_PROMPT_TEMPLATE.format(
        n=grid.n,
        grid_text=grid.format_grid(),
        horizontal_poem=poems["horizontal_poem"],
        vertical_poem=poems["vertical_poem"],
    )
    return {
        "gallery_id": gallery_id,
        "prompt": prompt,
        "poems": poems,
        "metadata": record["payload"].get("metadata", {}),
    }


def set_benchmark_score(gallery_id: str, score: float, *, special: bool = False) -> dict:
    return gallery.update_grid(gallery_id, benchmark_score=score, special=special)
