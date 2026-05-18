"""Pydantic request/response models."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class SinglePosRequest(BaseModel):
    pos: str
    model_id: str = "bert-base-uncased"


class PosCombinationsRequest(BaseModel):
    n: int = Field(ge=1, le=12)
    k: int = Field(ge=1, le=5000)
    algorithm: str = "tagger"
    max_sentences: int = Field(default=500, ge=50, le=10000)


class PosTestRequest(BaseModel):
    pos_sequence: str
    model_id: str = "bert-base-uncased"


class PosGridRequest(BaseModel):
    n: int = Field(ge=2, le=9)
    k: int = Field(ge=1, le=50)
    dictionary_result_id: str | None = None
    dictionary_path: str | None = None


class UnmaskerRequest(BaseModel):
    sentence: str
    model_id: str = "bert-base-uncased"
    top_k: int = Field(default=10, ge=1, le=50)


class GridUnmaskStartRequest(BaseModel):
    pos_grid: list[list[str]]
    model_id: str = "bert-base-uncased"
    top_k: int = Field(default=5, ge=1, le=30)
    auto_run: bool = False


class GridUnmaskOverrideRequest(BaseModel):
    session_id: str
    row: int
    col: int
    word: str


class BenchmarkScoreRequest(BaseModel):
    score: float = Field(ge=0, le=10)
    special: bool = False


class GridUnmaskFinalizeRequest(BaseModel):
    session_id: str
    metadata: dict[str, Any] | None = None
