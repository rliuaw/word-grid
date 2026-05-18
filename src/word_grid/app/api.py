"""FastAPI routes for all word-grid tools."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from word_grid.app.schemas import (
    BenchmarkScoreRequest,
    GridUnmaskFinalizeRequest,
    GridUnmaskOverrideRequest,
    GridUnmaskStartRequest,
    PosCombinationsRequest,
    PosGridRequest,
    PosTestRequest,
    SinglePosRequest,
    UnmaskerRequest,
)
from word_grid.jobs.manager import JobStatus, get_job, submit_job
from word_grid.ml.bert_models import AVAILABLE_MODELS
from word_grid.tools import (
    benchmark,
    gallery,
    grid_unmasker,
    pos_combinations,
    pos_grid,
    pos_test,
    single_pos,
    storage,
    unmasker,
)

STATIC_DIR = Path(__file__).parent / "static"
DIST_DIR = STATIC_DIR / "dist"


def _http_job_error(exc: Exception) -> HTTPException:
    if isinstance(exc, KeyError):
        return HTTPException(404, str(exc))
    if isinstance(exc, ValueError):
        return HTTPException(400, str(exc))
    return HTTPException(500, str(exc))


def create_app() -> FastAPI:
    app = FastAPI(
        title="Word Grid Explorer",
        description="Tools for constructing and analyzing N×N word grids",
        version="0.2.0",
    )

    @app.get("/api/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/api/models")
    def models() -> dict:
        return {"models": AVAILABLE_MODELS}

    @app.get("/api/algorithms")
    def algorithms() -> dict:
        return {
            "algorithms": [
                {"id": k, "label": v[0]} for k, v in pos_combinations.ALGORITHMS.items()
            ]
        }

    @app.get("/api/results")
    def results_list(tool: str | None = None) -> dict:
        return {"results": storage.list_results(tool)}

    @app.get("/api/results/options")
    def results_options(tool: str) -> dict:
        """Simplified list for dropdown selectors."""
        entries = storage.list_results(tool)
        options = []
        for e in entries:
            label = e.get("label") or e["id"]
            options.append(
                {
                    "id": e["id"],
                    "label": label,
                    "created_at": e.get("created_at"),
                    "tool": e.get("tool"),
                }
            )
        return {"options": options}

    @app.get("/api/results/{result_id}")
    def results_get(result_id: str) -> dict:
        try:
            return storage.load_result(result_id)
        except FileNotFoundError as e:
            raise HTTPException(404, str(e)) from e

    @app.get("/api/results/{result_id}/dictionary.txt")
    def results_dictionary(result_id: str) -> FileResponse:
        path = storage.artifact_path(result_id, "dictionary.txt")
        if not path.exists():
            raise HTTPException(404, "dictionary.txt not found")
        return FileResponse(path, media_type="text/plain", filename="dictionary.txt")

    # ---- Jobs / progress -------------------------------------------------

    @app.get("/api/jobs/{job_id}/progress")
    def job_progress(job_id: str) -> dict:
        try:
            job = get_job(job_id)
        except KeyError as e:
            raise HTTPException(404, str(e)) from e
        return job.to_progress_dict()

    @app.get("/api/jobs/{job_id}/result")
    def job_result(job_id: str) -> dict:
        try:
            job = get_job(job_id)
        except KeyError as e:
            raise HTTPException(404, str(e)) from e
        if job.status == JobStatus.FAILED:
            raise HTTPException(500, job.error or "Job failed")
        if job.status != JobStatus.COMPLETED:
            raise HTTPException(409, "Job not complete")
        return job.to_result_dict()

    # ---- Async tools -----------------------------------------------------

    @app.post("/api/tools/single-pos")
    def api_single_pos(req: SinglePosRequest) -> dict:
        return submit_job(
            "single_pos",
            single_pos.generate,
            req.pos,
            req.model_id,
        )

    @app.post("/api/tools/pos-combinations")
    def api_pos_combinations(req: PosCombinationsRequest) -> dict:
        return submit_job(
            "pos_combinations",
            pos_combinations.generate,
            req.n,
            req.k,
            req.algorithm,
            max_sentences=req.max_sentences,
        )

    @app.post("/api/tools/pos-test")
    def api_pos_test(req: PosTestRequest) -> dict:
        return submit_job(
            "pos_test",
            pos_test.generate,
            req.pos_sequence,
            req.model_id,
        )

    @app.post("/api/tools/pos-grid")
    def api_pos_grid(req: PosGridRequest) -> dict:
        return submit_job(
            "pos_grid",
            pos_grid.generate,
            req.n,
            req.k,
            dictionary_result_id=req.dictionary_result_id,
            dictionary_path=req.dictionary_path,
        )

    @app.post("/api/tools/unmasker")
    def api_unmasker(req: UnmaskerRequest) -> dict:
        return submit_job(
            "unmasker",
            unmasker.unmask,
            req.sentence,
            req.model_id,
            req.top_k,
        )

    @app.post("/api/tools/grid-unmask/start")
    def api_grid_unmask_start(req: GridUnmaskStartRequest) -> dict:
        return submit_job(
            "grid_unmask_start",
            grid_unmasker.start_session,
            req.pos_grid,
            req.model_id,
            req.top_k,
            auto_run=req.auto_run,
        )

    @app.post("/api/tools/grid-unmask/step-forward")
    def api_grid_unmask_forward(body: dict) -> dict:
        sid = body.get("session_id")
        if not sid:
            raise HTTPException(400, "session_id required")
        return submit_job("grid_unmask_step", grid_unmasker.step_forward, sid)

    @app.post("/api/tools/grid-unmask/step-backward")
    def api_grid_unmask_backward(body: dict) -> dict:
        sid = body.get("session_id")
        if not sid:
            raise HTTPException(400, "session_id required")
        return submit_job("grid_unmask_step", grid_unmasker.step_backward, sid)

    @app.post("/api/tools/grid-unmask/override")
    def api_grid_unmask_override(req: GridUnmaskOverrideRequest) -> dict:
        return submit_job(
            "grid_unmask_override",
            grid_unmasker.override_cell,
            req.session_id,
            req.row,
            req.col,
            req.word,
        )

    @app.post("/api/tools/grid-unmask/finalize")
    def api_grid_unmask_finalize(req: GridUnmaskFinalizeRequest) -> dict:
        return submit_job(
            "grid_unmask_finalize",
            grid_unmasker.finalize,
            req.session_id,
            metadata=req.metadata,
        )

    @app.get("/api/benchmark/{gallery_id}/prompt")
    def api_benchmark_prompt(gallery_id: str) -> dict:
        try:
            return benchmark.gemini_prompt(gallery_id)
        except FileNotFoundError as e:
            raise HTTPException(404, str(e)) from e

    @app.post("/api/benchmark/{gallery_id}/score")
    def api_benchmark_score(gallery_id: str, req: BenchmarkScoreRequest) -> dict:
        try:
            return benchmark.set_benchmark_score(
                gallery_id, req.score, special=req.special
            )
        except FileNotFoundError as e:
            raise HTTPException(404, str(e)) from e

    # ---- Gallery (sync reads) --------------------------------------------

    @app.get("/api/gallery")
    def api_gallery_list() -> dict:
        entries = gallery.list_grids()
        items = []
        for e in entries:
            rec = storage.load_result(e["id"])
            items.append(
                {
                    "id": e["id"],
                    "created_at": e["created_at"],
                    "n": rec["payload"].get("n"),
                    "cells": rec["payload"].get("cells"),
                    "metadata": rec["payload"].get("metadata", {}),
                }
            )
        return {"grids": items}

    @app.get("/api/gallery/{gallery_id}")
    def api_gallery_get(gallery_id: str) -> dict:
        try:
            return gallery.get_grid(gallery_id)
        except FileNotFoundError as e:
            raise HTTPException(404, str(e)) from e

    # ---- SPA static assets -----------------------------------------------

    if DIST_DIR.exists():
        assets_dir = DIST_DIR / "assets"
        if assets_dir.exists():
            app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

        @app.get("/")
        def spa_index() -> FileResponse:
            return FileResponse(DIST_DIR / "index.html")

        @app.get("/{full_path:path}")
        def spa_fallback(full_path: str) -> FileResponse:
            if full_path.startswith("api/"):
                raise HTTPException(404)
            file_path = DIST_DIR / full_path
            if file_path.is_file():
                return FileResponse(file_path)
            return FileResponse(DIST_DIR / "index.html")
    elif STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

        @app.get("/")
        def legacy_index() -> FileResponse:
            return FileResponse(STATIC_DIR / "index.html")

    return app


def run() -> None:
    import uvicorn

    uvicorn.run("word_grid.app.api:create_app", factory=True, host="127.0.0.1", port=8765)
