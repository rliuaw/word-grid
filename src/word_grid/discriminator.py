"""Discriminator: score sentences using Mistral-Nemo-Instruct-2407 GGUF.

This module loads the quantised GGUF model via ``llama-cpp-python`` and
exposes a thin :class:`SentenceScorer` that rates individual English
sentences on syntactic quality (0–10).
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Sequence

from huggingface_hub import hf_hub_download

from word_grid.config import (
    DISCRIMINATOR_GGUF_FILE,
    DISCRIMINATOR_REPO,
    HF_TOKEN,
)
from word_grid.prompts import discriminator_messages

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy-import llama_cpp so the rest of the package stays importable on
# machines that don't have the C++ backend compiled.
# ---------------------------------------------------------------------------
try:
    from llama_cpp import Llama  # type: ignore[import-untyped]
except ImportError:
    Llama = None  # type: ignore[assignment,misc]


class SentenceScorer:
    """Score English sentences using a GGUF language model.

    Parameters
    ----------
    model_path:
        Filesystem path to a ``.gguf`` file.  If *None*, the file is
        downloaded from *repo_id* / *gguf_filename* on the HF Hub.
    repo_id:
        HuggingFace repo containing the GGUF weights.
    gguf_filename:
        Name of the ``.gguf`` file inside the repo.
    n_ctx:
        Context window (tokens) for the GGUF model.
    n_gpu_layers:
        How many layers to offload to GPU.  ``0`` = CPU-only, which is
        recommended when the GPU is occupied by the generator.
    temperature:
        Sampling temperature for the scorer.  ``0.0`` = greedy.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        repo_id: str = DISCRIMINATOR_REPO,
        gguf_filename: str = DISCRIMINATOR_GGUF_FILE,
        n_ctx: int = 512,
        n_gpu_layers: int = 0,
        temperature: float = 0.0,
        max_tokens: int = 30,
    ) -> None:
        if Llama is None:
            raise ImportError(
                "llama-cpp-python is required for the discriminator.  "
                "Install it with:  pip install llama-cpp-python"
            )

        if model_path is None:
            logger.info(
                "Downloading discriminator GGUF from %s / %s …",
                repo_id, gguf_filename,
            )
            # Use local_dir to avoid HF cache symlink issues on
            # cloud volumes (e.g. Modal).  The file is downloaded
            # directly into local_dir/<gguf_filename>.
            import os
            local_dir = os.environ.get("HF_HOME", None)
            if local_dir:
                local_dir = os.path.join(local_dir, "gguf")
            else:
                local_dir = os.path.join(
                    os.path.expanduser("~"), ".cache", "gguf"
                )
            target = Path(local_dir) / gguf_filename
            if target.exists() and target.stat().st_size > 1_000_000:
                logger.info("Using cached GGUF at %s", target)
                model_path = str(target)
            else:
                model_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=gguf_filename,
                    token=HF_TOKEN,
                    local_dir=local_dir,
                )
                logger.info("Downloaded GGUF to %s", model_path)

        logger.info("Loading discriminator from %s …", model_path)
        self.llm = Llama(
            model_path=str(model_path),
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )
        self.temperature = temperature
        self.max_tokens = max_tokens

        # ── Proposal D: verify GPU offload ──────────────────────────
        self._verify_gpu_offload(n_gpu_layers)

        # ── Proposal C: per-call timing stats ───────────────────────
        self._total_calls = 0
        self._total_time = 0.0
        self._verbose_first_n = 5  # log first N calls individually

    def _verify_gpu_offload(self, requested_layers: int) -> None:
        """Log llama-cpp-python version and GPU offload status."""
        try:
            import llama_cpp
            ver = getattr(llama_cpp, "__version__", "unknown")
            print(f"[disc] llama-cpp-python version: {ver}", flush=True)
        except Exception:
            pass

        # Check if the build supports GPU offload — hard error if not
        try:
            from llama_cpp import llama_supports_gpu_offload  # type: ignore[import]
            gpu_ok = llama_supports_gpu_offload()
            print(f"[disc] GPU offload supported: {gpu_ok}", flush=True)
            if requested_layers != 0 and not gpu_ok:
                raise RuntimeError(
                    f"n_gpu_layers={requested_layers} requested but "
                    "llama-cpp-python was NOT built with GPU/CUDA support! "
                    "Discriminator would fall back to CPU which is ~50× slower. "
                    "Rebuild with: CMAKE_ARGS=\"-DGGML_CUDA=on\" pip install "
                    "llama-cpp-python --no-binary llama-cpp-python"
                )
        except ImportError:
            # Older versions don't expose llama_supports_gpu_offload;
            # raise if GPU was explicitly requested — we can't verify.
            if requested_layers != 0:
                raise RuntimeError(
                    f"n_gpu_layers={requested_layers} requested but "
                    "llama_supports_gpu_offload() is not available in this "
                    "llama-cpp-python build. Cannot verify GPU support. "
                    "Upgrade llama-cpp-python or rebuild with CUDA."
                )

        print(
            f"[disc] n_gpu_layers requested: {requested_layers}",
            flush=True,
        )

    # ------------------------------------------------------------------ score

    def score_sentence(self, sentence: str) -> float:
        """Return a syntax score in ``[0.0, 10.0]`` for *sentence*.

        The discriminator is prompted via the chat template defined in
        :mod:`word_grid.prompts`.  If parsing the model's reply fails,
        a fallback heuristic tries to find the first float in the text;
        on total failure the method returns ``0.0``.
        """
        t0 = time.time()

        messages = discriminator_messages(sentence)
        response = self.llm.create_chat_completion(
            messages=messages,  # type: ignore[arg-type]
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        raw: str = response["choices"][0]["message"]["content"]  # type: ignore[index]
        score = self._parse_score(raw)

        elapsed = time.time() - t0
        self._total_calls += 1
        self._total_time += elapsed

        # Log first N calls individually for debugging, then every 50th
        if self._total_calls <= self._verbose_first_n:
            print(
                f"[disc] call #{self._total_calls}: {elapsed:.2f}s — "
                f"{sentence!r:.60} → {score:.1f} (raw: {raw!r:.40})",
                flush=True,
            )
        elif self._total_calls % 50 == 0:
            avg = self._total_time / self._total_calls
            print(
                f"[disc] call #{self._total_calls}: avg {avg:.2f}s/call, "
                f"cumulative {self._total_time:.0f}s",
                flush=True,
            )

        return score

    def score_sentences(self, sentences: Sequence[str]) -> list[float]:
        """Score multiple sentences sequentially."""
        return [self.score_sentence(s) for s in sentences]

    # ----------------------------------------------------------------- parse

    @staticmethod
    def _parse_score(text: str) -> float:
        """Extract a float score from the model's raw output.

        Strategies (tried in order):
        1. Parse full JSON  ``{"syntax_score": X.X}``
        2. Find a ``"syntax_score"`` key with regex
        3. Find the first standalone float in the text
        4. Return ``0.0`` as fallback
        """
        # Strip any reasoning wrapper
        text = re.sub(r"\[THINK\].*?\[/THINK\]", "", text, flags=re.DOTALL).strip()

        # Strategy 1: full JSON parse
        try:
            data = json.loads(text)
            if isinstance(data, dict) and "syntax_score" in data:
                return _clamp(float(data["syntax_score"]))
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

        # Strategy 2: regex for key
        m = re.search(r'"syntax_score"\s*:\s*([\d.]+)', text)
        if m:
            return _clamp(float(m.group(1)))

        # Strategy 3: first float
        m = re.search(r"\b(\d{1,2}(?:\.\d+)?)\b", text)
        if m:
            return _clamp(float(m.group(1)))

        logger.warning("Could not parse score from discriminator output: %r", text)
        return 0.0


def _clamp(v: float, lo: float = 0.0, hi: float = 10.0) -> float:
    return max(lo, min(hi, v))
