"""POS grid generator — NxN POS grids from dictionary.txt via constraint solver."""

from __future__ import annotations

import random
from pathlib import Path

from word_grid.pos.tags import parse_pos_sequence
from word_grid.tools import storage


def load_dictionary(path: str | Path) -> set[tuple[str, ...]]:
    combos: set[tuple[str, ...]] = set()
    text = Path(path).read_text(encoding="utf-8")
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        combos.add(parse_pos_sequence(line))
    return combos


def _combo_to_tuple(item) -> tuple[str, ...]:
    if isinstance(item, dict):
        return tuple(item["sequence"])
    if isinstance(item, list):
        return tuple(item)
    return parse_pos_sequence(str(item))


def load_dictionary_from_result(result_id: str) -> set[tuple[str, ...]]:
    path = storage.artifact_path(result_id, "dictionary.txt")
    if path.exists():
        return load_dictionary(path)
    record = storage.load_result(result_id)
    payload = record["payload"]
    if payload.get("combo_tuples"):
        return {tuple(t) for t in payload["combo_tuples"]}
    combos = payload.get("combinations", [])
    return {_combo_to_tuple(c) for c in combos}


def solve_pos_grid(
    n: int,
    combos: set[tuple[str, ...]],
    *,
    max_solutions: int = 1,
) -> list[list[tuple[str, ...]]]:
    """Backtracking solver: each row and column must be in *combos*."""
    valid = {c for c in combos if len(c) == n}
    if not valid:
        return []

    solutions: list[list[tuple[str, ...]]] = []
    grid: list[list[str | None]] = [[None] * n for _ in range(n)]

    def ok_through_row(upto: int) -> bool:
        """Validate rows 0..upto and column prefixes through row *upto*."""
        for i in range(upto + 1):
            if tuple(grid[i]) not in valid:  # type: ignore[arg-type]
                return False
        for j in range(n):
            prefix: list[str] = []
            for i in range(upto + 1):
                cell = grid[i][j]
                if cell is None:
                    break
                prefix.append(cell)
            if prefix:
                pt = tuple(prefix)
                if not any(v[: len(pt)] == pt for v in valid):
                    return False
        return True

    def search(row: int) -> None:
        if len(solutions) >= max_solutions:
            return
        if row == n:
            for j in range(n):
                if tuple(grid[i][j] for i in range(n)) not in valid:
                    return
            solutions.append([tuple(grid[i]) for i in range(n)])  # type: ignore[arg-type]
            return
        candidates = list(valid)
        random.shuffle(candidates)
        for seq in candidates:
            grid[row] = list(seq)
            if ok_through_row(row):
                search(row + 1)
            if len(solutions) >= max_solutions:
                return
        grid[row] = [None] * n

    search(0)
    return solutions


def try_swordsmith(dictionary_path: Path, n: int, k: int) -> list[list[tuple[str, ...]]] | None:
    """Optional integration with swordsmith_cooking if installed."""
    try:
        import subprocess
        import sys

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "swordsmith_cooking",
                "--size",
                str(n),
                "--dict",
                str(dictionary_path),
                "--count",
                str(k),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            return None
        grids: list[list[tuple[str, ...]]] = []
        for block in result.stdout.strip().split("\n\n"):
            rows = [parse_pos_sequence(line) for line in block.splitlines() if line.strip()]
            if len(rows) == n:
                grids.append(rows)
        return grids or None
    except Exception:
        return None


def generate(
    n: int,
    k: int,
    dictionary_result_id: str | None = None,
    dictionary_path: str | None = None,
    *,
    persist: bool = True,
    report=None,
) -> dict:
    if dictionary_result_id:
        if report:
            report(10, "Loading dictionary from saved result…", "load")
        combos = load_dictionary_from_result(dictionary_result_id)
        dict_path = storage.artifact_path(dictionary_result_id, "dictionary.txt")
    elif dictionary_path:
        dict_path = Path(dictionary_path)
        combos = load_dictionary(dict_path)
    else:
        raise ValueError("Provide dictionary_result_id or dictionary_path")

    if report:
        report(25, f"Solving {n}×{n} grid ({len(combos)} entries)…", "solve")

    grids: list[list[list[str]]] = []
    if dict_path.exists():
        external = try_swordsmith(dict_path, n, k)
        if external:
            grids = [[list(row) for row in g] for g in external]

    if len(grids) < k:
        solutions = solve_pos_grid(n, combos, max_solutions=k - len(grids))
        for sol in solutions:
            grids.append([list(row) for row in sol])

    if report:
        report(90, f"Found {len(grids)} candidate grid(s)", "done")

    payload = {
        "n": n,
        "k": k,
        "requested": k,
        "found": len(grids),
        "grids": grids,
        "solver": "swordsmith" if len(grids) and dict_path.exists() else "backtracking",
    }
    if persist:
        refs = [dictionary_result_id] if dictionary_result_id else []
        payload["result_id"] = storage.save_result(
            "pos_grid",
            payload,
            label=f"{n}x{n} x{len(grids)}",
            refs=[r for r in refs if r],
        )
    return payload
