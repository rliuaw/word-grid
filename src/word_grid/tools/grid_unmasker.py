"""POS grid unmasker with step-by-step debugger."""

from __future__ import annotations

from dataclasses import dataclass, field

from word_grid.grid import WordGrid
from word_grid.ml.bert_models import mask_token_for
from word_grid.ml.pos_vocab import (
    is_punctuation_token,
    random_word_for_pos,
    warmup,
    word_matches_pos,
)
from word_grid.pos.tags import fill_priority
from word_grid.tools import single_pos
from word_grid.tools.unmasker import unmask


@dataclass
class DebugStep:
    index: int
    row: int
    col: int
    pos: str
    word: str
    method: str
    rule: str
    row_sentence: str
    col_sentence: str
    top_k: list[dict] = field(default_factory=list)


@dataclass
class UnmaskSession:
    pos_grid: list[list[str]]
    word_grid: list[list[str | None]]
    n: int
    model_id: str
    top_k: int
    steps: list[DebugStep] = field(default_factory=list)
    cursor: int = 0
    seeds_used: int = 0

    def filled_count(self, i: int, j: int) -> int:
        count = 0
        for c in range(self.n):
            if self.word_grid[i][c]:
                count += 1
            if self.word_grid[c][j]:
                count += 1
        return count - (1 if self.word_grid[i][j] else 0)

    def cell_order(self) -> list[tuple[int, int]]:
        cells = [(i, j) for i in range(self.n) for j in range(self.n)]

        def key(rc: tuple[int, int]) -> tuple:
            i, j = rc
            tag = self.pos_grid[i][j]
            return (fill_priority(tag), -self.filled_count(i, j), i, j)

        return sorted(cells, key=key)

    def row_words(self, i: int, j_skip: int | None = None) -> list[str]:
        out: list[str] = []
        for c in range(self.n):
            w = self.word_grid[i][c]
            if w:
                out.append(w)
            elif c == j_skip:
                out.append("__MASK__")
        return out

    def col_words(self, j: int, i_skip: int | None = None) -> list[str]:
        out: list[str] = []
        for r in range(self.n):
            w = self.word_grid[r][j]
            if w:
                out.append(w)
            elif r == i_skip:
                out.append("__MASK__")
        return out

    def build_masked_sentence(self, words: list[str], mask: str) -> str | None:
        if "__MASK__" not in words:
            return None
        return " ".join(mask if w == "__MASK__" else w for w in words)

    @property
    def seed_limit(self) -> int:
        return 2 * self.n

    def _filter_unmask_results(self, results: list[dict], pos: str) -> list[dict]:
        """Keep only non-punctuation tokens matching the cell's intended POS."""
        filtered: list[dict] = []
        for r in results:
            word = r.get("word", "").strip()
            if not word or is_punctuation_token(word, self.model_id):
                continue
            if word_matches_pos(word, pos, self.model_id):
                filtered.append(r)
        return filtered

    def _merge_bert_scores(
        self, scores: dict[str, float], results: list[dict], pos: str
    ) -> None:
        for r in self._filter_unmask_results(results, pos):
            w = r["word"].strip()
            scores[w] = min(scores.get(w, 1.0), r["score"])

    def fill_cell(self, i: int, j: int, report=None) -> DebugStep:
        tag = self.pos_grid[i][j]
        mask = mask_token_for(self.model_id)
        method = "pos_vocab"
        rule = "priority: prepositions > verbs > nouns"
        top_results: list[dict] = []

        if self.seeds_used < self.seed_limit:
            word = random_word_for_pos(tag, self.model_id) or single_pos.generate(
                tag, self.model_id, persist=False
            )["word"]
            self.seeds_used += 1
            rule = (
                f"seed phase ({self.seeds_used}/{self.seed_limit}): random POS vocab word; "
                "prepositions > verbs > nouns"
            )
        else:
            method = "bert_min_score"
            row_sent = self.build_masked_sentence(self.row_words(i, j), mask)
            col_sent = self.build_masked_sentence(self.col_words(j, i), mask)
            scores: dict[str, float] = {}
            fetch_k = max(self.top_k * 15, 50)
            if row_sent:
                row_results = unmask(
                    row_sent, self.model_id, fetch_k, persist=False, report=report
                )["results"]
                self._merge_bert_scores(scores, row_results, tag)
            if col_sent:
                col_results = unmask(
                    col_sent, self.model_id, fetch_k, persist=False, report=report
                )["results"]
                for r in self._filter_unmask_results(col_results, tag):
                    w = r["word"].strip()
                    scores[w] = (
                        min(scores[w], r["score"]) if w in scores else r["score"]
                    )
            ranked = sorted(scores.items(), key=lambda x: -x[1])[: self.top_k]
            top_results = [{"word": w, "score": s} for w, s in ranked]
            word = ranked[0][0] if ranked else (random_word_for_pos(tag, self.model_id) or "the")
            rule = (
                "BERT: total_score = min(score_row, score_column); "
                f"candidates filtered to POS={tag}, no punctuation; "
                f"constraint={self.filled_count(i, j)} shared words"
            )

        self.word_grid[i][j] = word
        step = DebugStep(
            index=len(self.steps),
            row=i,
            col=j,
            pos=tag,
            word=word,
            method=method,
            rule=rule,
            row_sentence=" ".join(self.row_words(i)),
            col_sentence=" ".join(self.col_words(j)),
            top_k=top_results,
        )
        self.steps.append(step)
        return step

    def run_all(self) -> None:
        for i, j in self.cell_order():
            if not self.word_grid[i][j]:
                self.fill_cell(i, j)
        self.cursor = len(self.steps)

    def to_word_grid(self) -> WordGrid:
        cells = [[w or "" for w in row] for row in self.word_grid]
        return WordGrid(cells=cells)

    def snapshot(self) -> dict:
        return {
            "pos_grid": self.pos_grid,
            "word_grid": self.word_grid,
            "cursor": self.cursor,
            "total_steps": len(self.steps),
            "steps": [
                {
                    "index": s.index,
                    "row": s.row,
                    "col": s.col,
                    "pos": s.pos,
                    "word": s.word,
                    "method": s.method,
                    "rule": s.rule,
                    "row_sentence": s.row_sentence,
                    "col_sentence": s.col_sentence,
                    "top_k": s.top_k,
                }
                for s in self.steps
            ],
        }


_SESSIONS: dict[str, UnmaskSession] = {}


def start_session(
    pos_grid: list[list[str]],
    model_id: str = "bert-base-uncased",
    top_k: int = 5,
    *,
    auto_run: bool = False,
    report=None,
) -> dict:
    if report:
        from word_grid.ml.progress import warmup_with_progress

        warmup_with_progress(model_id, report)
    else:
        warmup(model_id)
    n = len(pos_grid)
    for row in pos_grid:
        if len(row) != n:
            raise ValueError("POS grid must be square")
    session = UnmaskSession(
        pos_grid=pos_grid,
        word_grid=[[None] * n for _ in range(n)],
        n=n,
        model_id=model_id,
        top_k=top_k,
    )
    if auto_run:
        if report:
            report(90, "Auto-filling grid…", "fill")
        session.run_all()
    import uuid

    sid = uuid.uuid4().hex[:12]
    _SESSIONS[sid] = session
    out = {"session_id": sid, **session.snapshot()}
    return out


def step_forward(session_id: str, report=None) -> dict:
    session = _get(session_id)
    if report:
        report(15, "Filling next cell…", "step")
    order = session.cell_order()
    # find next empty cell in order
    for i, j in order:
        if not session.word_grid[i][j]:
            session.fill_cell(i, j, report=report)
            session.cursor = len(session.steps)
            break
    else:
        session.cursor = len(session.steps)
    return {"session_id": session_id, **session.snapshot()}


def step_backward(session_id: str) -> dict:
    session = _get(session_id)
    if session.cursor > 0:
        session.cursor -= 1
        last = session.steps[session.cursor]
        session.word_grid[last.row][last.col] = None
        session.steps = session.steps[: session.cursor]
        session.seeds_used = sum(
            1 for s in session.steps if s.method == "pos_vocab"
        )
    return {"session_id": session_id, **session.snapshot()}


def override_cell(session_id: str, row: int, col: int, word: str) -> dict:
    session = _get(session_id)
    session.word_grid[row][col] = word
    # truncate steps after this cell if re-overriding
    session.steps = [s for s in session.steps if s.index < session.cursor]
    session.cursor = min(session.cursor, len(session.steps))
    return {"session_id": session_id, **session.snapshot()}


def finalize(session_id: str, *, persist: bool = True, metadata: dict | None = None) -> dict:
    from word_grid.tools import gallery

    session = _get(session_id)
    grid = session.to_word_grid()
    meta = metadata or {}
    meta.update(
        {
            "model_id": session.model_id,
            "algorithm": "pos_grid_unmasker",
            "steps": len(session.steps),
        }
    )
    payload = {
        "session_id": session_id,
        "grid": grid.cells,
        "formatted": grid.format_grid(),
        "sentences": [{"label": l, "text": t} for l, t in grid.all_sentences()],
    }
    if persist:
        payload["gallery_id"] = gallery.save_grid(
            grid.cells,
            metadata=meta,
            steps=session.snapshot()["steps"],
        )
    return payload


def _get(session_id: str) -> UnmaskSession:
    if session_id not in _SESSIONS:
        raise KeyError(f"Unknown session: {session_id}")
    return _SESSIONS[session_id]
