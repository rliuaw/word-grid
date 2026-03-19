"""Reward function for GRPO training.

Wraps grid parsing + discriminator scoring into a callable that
conforms to the ``reward_funcs`` interface expected by TRL's
:class:`~trl.GRPOTrainer`.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any

from word_grid.config import DEFAULT_GRID_SIZE
from word_grid.discriminator import SentenceScorer
from word_grid.grid import WordGrid, parse_grid_from_text, strip_thinking_tokens

logger = logging.getLogger(__name__)

# Optional: import wandb for metric logging (may not be available everywhere)
try:
    import wandb as _wandb
except ImportError:
    _wandb = None  # type: ignore[assignment]


class GridRewardFunction:
    """Callable reward function for GRPO.

    Parameters
    ----------
    scorer:
        A :class:`SentenceScorer` instance (discriminator).
    grid_size:
        Expected NxN dimension.
    invalid_reward:
        Reward returned when the completion cannot be parsed into a
        valid grid.
    partial_bonus:
        Small bonus added proportionally when the grid is partially
        parseable (wrong dimensions but some valid rows).
    """

    # TRL's GRPOTrainer expects reward_funcs to have a __name__ attribute.
    __name__ = "grid_reward"

    def __init__(
        self,
        scorer: SentenceScorer,
        grid_size: int = DEFAULT_GRID_SIZE,
        invalid_reward: float = 0.0,
        partial_bonus: float = 0.1,
    ) -> None:
        self.scorer = scorer
        self.grid_size = grid_size
        self.invalid_reward = invalid_reward
        self.partial_bonus = partial_bonus
        self._call_count = 0

    # ------------------------------------------------------------------
    # Main entry point — compatible with TRL GRPOTrainer.reward_funcs
    # ------------------------------------------------------------------

    def __call__(
        self,
        completions: list[str],
        **kwargs: Any,
    ) -> list[float]:
        """Score a batch of model completions.

        Each completion should contain a word grid as plain text.
        Returns one float reward per completion.

        Sentences are **deduplicated** across the batch so each unique
        sentence is scored only once by the discriminator.
        """
        self._call_count += 1
        t0 = time.time()

        # Phase 1: parse all grids, collect unique sentences
        parsed: list[tuple[list[tuple[str, str]] | None, dict]] = []
        unique_sentences: dict[str, float] = {}  # sentence → score (filled later)
        n_valid = 0
        raw_snippets: list[str] = []  # for logging

        for text in completions:
            text = self._extract_text(text)
            text = strip_thinking_tokens(text)

            # Keep a short snippet for logging (Proposal F)
            raw_snippets.append(text[:120].replace("\n", "\\n"))

            grid = parse_grid_from_text(text, n=self.grid_size)
            if grid is None:
                parsed.append((None, {"valid": False, "reason": "unparseable"}))
            else:
                n_valid += 1
                sents = grid.all_sentences()
                parsed.append((sents, {"valid": True}))
                for _, sentence in sents:
                    unique_sentences.setdefault(sentence, 0.0)

        # Phase 2: score unique sentences (single pass through discriminator)
        t_disc = time.time()
        if unique_sentences:
            for sent in unique_sentences:
                unique_sentences[sent] = self.scorer.score_sentence(sent)
        disc_elapsed = time.time() - t_disc

        # Phase 3: compute rewards using cached scores
        rewards: list[float] = []
        for sents, info in parsed:
            if sents is None:
                rewards.append(self.invalid_reward)
                continue
            scores = [unique_sentences[s] for _, s in sents]
            # Geometric mean penalises any single weak sentence more than
            # arithmetic mean, mitigating reward hacking where the model
            # produces one excellent sentence to compensate for poor ones.
            geo_mean = (
                math.prod(scores) ** (1.0 / len(scores))
                if scores else 0.0
            )
            reward = geo_mean / 10.0  # normalise 0–10 → 0–1
            rewards.append(reward)
            logger.debug(
                "Grid OK — geo_mean_syntax=%.2f  sentences=%s",
                geo_mean,
                [s for _, s in sents[:3]],
            )

        total_elapsed = time.time() - t0
        mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
        n_unique = len(unique_sentences)
        n_total = len(completions)

        # ── Proposal F: louder console log with raw completion snippets ──
        print(
            f"🏆 reward_fn #{self._call_count}: "
            f"{n_valid}/{n_total} valid, "
            f"{n_unique} unique sents, "
            f"mean={mean_reward:.3f} "
            f"[min={min(rewards) if rewards else 0:.3f} max={max(rewards) if rewards else 0:.3f}], "
            f"disc={disc_elapsed:.1f}s, total={total_elapsed:.1f}s",
            flush=True,
        )
        # Log first 2 raw completion snippets per call for debugging
        for idx, snippet in enumerate(raw_snippets[:2]):
            print(
                f"   completion[{idx}]: {snippet!r:.100}  → reward={rewards[idx]:.3f}",
                flush=True,
            )

        logger.info(
            "reward_fn call #%d: %d/%d valid, %d unique sents, "
            "mean_reward=%.3f, disc=%.1fs, total=%.1fs",
            self._call_count, n_valid, n_total, n_unique,
            mean_reward, disc_elapsed, total_elapsed,
        )

        # Log to W&B if an active run exists
        if _wandb is not None and _wandb.run is not None:
            _wandb.log({
                "reward/mean": mean_reward,
                "reward/min": min(rewards) if rewards else 0.0,
                "reward/max": max(rewards) if rewards else 0.0,
                "reward/valid_grids": n_valid,
                "reward/parse_failures": n_total - n_valid,
                "reward/unique_sentences": n_unique,
                "reward/disc_time_s": disc_elapsed,
                "reward/total_time_s": total_elapsed,
            }, commit=False)  # commit=False so it bundles with the trainer's log

        return rewards

    @staticmethod
    def _extract_text(text: Any) -> str:
        """Normalise various input shapes from TRL into a plain string."""
        if isinstance(text, list):
            return text[-1]["content"] if text else ""
        if isinstance(text, dict):
            return text.get("content", str(text))
        return str(text)

    # ------------------------------------------------------------------
    # Utility for offline evaluation
    # ------------------------------------------------------------------

    def evaluate_grid(self, grid: WordGrid) -> tuple[float, dict[str, float]]:
        """Score an already-parsed grid.  Returns ``(geo_mean, {label: score})``."""
        all_sents = grid.all_sentences()
        scores: dict[str, float] = {}
        for label, sentence in all_sents:
            scores[label] = self.scorer.score_sentence(sentence)
        geo_mean = (
            math.prod(scores.values()) ** (1.0 / len(scores))
            if scores else 0.0
        )
        return geo_mean, scores
