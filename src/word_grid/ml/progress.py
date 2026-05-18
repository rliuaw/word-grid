"""Progress helpers for model and dataset loading."""

from __future__ import annotations

from typing import Callable

ReportFn = Callable[[float, str, str], None]


def report_tokenizer_load(model_id: str, report: ReportFn) -> None:
    report(5, f"Loading tokenizer for {model_id}…", "tokenizer")


def report_model_load(model_id: str, report: ReportFn) -> None:
    report(35, f"Downloading / loading {model_id} weights…", "model")


def report_pipeline_ready(report: ReportFn) -> None:
    report(90, "Initializing inference pipeline…", "pipeline")


def warmup_with_progress(model_id: str, report: ReportFn) -> int:
    """Tag BERT vocab with incremental progress."""
    from word_grid.ml.pos_vocab import warmup

    report(10, "Downloading NLTK tagger data…", "nltk")
    report(20, f"Loading tokenizer ({model_id})…", "tokenizer")
    # warmup loads tokenizer + tags entire vocab
    total = warmup(model_id)
    report(85, f"Tagged {total:,} vocabulary tokens", "vocab")
    return total
