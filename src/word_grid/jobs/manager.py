"""Thread-pool jobs with pollable progress for long-running tool calls."""

from __future__ import annotations

import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable

_executor = ThreadPoolExecutor(max_workers=2)
_lock = threading.Lock()
_jobs: dict[str, "JobRecord"] = {}


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class JobProgress:
    percent: float = 0.0
    message: str = "Starting…"
    stage: str = "init"

    def to_dict(self) -> dict[str, Any]:
        return {
            "percent": round(self.percent, 1),
            "message": self.message,
            "stage": self.stage,
        }


@dataclass
class JobRecord:
    id: str
    tool: str
    status: JobStatus = JobStatus.PENDING
    progress: JobProgress = field(default_factory=JobProgress)
    result: Any = None
    error: str | None = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def reporter(self) -> Callable[[float, str, str], None]:
        def report(percent: float, message: str, stage: str) -> None:
            with _lock:
                self.progress.percent = max(0.0, min(100.0, percent))
                self.progress.message = message
                self.progress.stage = stage

        return report

    def to_progress_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.id,
            "tool": self.tool,
            "status": self.status.value,
            "created_at": self.created_at,
            "progress": self.progress.to_dict(),
            "error": self.error,
        }

    def to_result_dict(self) -> dict[str, Any]:
        d = self.to_progress_dict()
        if self.status == JobStatus.COMPLETED:
            d["result"] = self.result
        return d


def start_job(tool: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> str:
    """Run *fn* in a worker thread; *fn* may accept ``report=`` keyword."""
    job_id = uuid.uuid4().hex[:12]
    record = JobRecord(id=job_id, tool=tool)
    with _lock:
        _jobs[job_id] = record

    def run() -> None:
        with _lock:
            record.status = JobStatus.RUNNING
            record.progress.message = "Running…"
        try:
            report = record.reporter()
            if "report" in fn.__code__.co_varnames:
                result = fn(*args, report=report, **kwargs)
            else:
                report(5, "Working…", "run")
                result = fn(*args, **kwargs)
                report(100, "Done", "done")
            with _lock:
                record.result = result
                record.status = JobStatus.COMPLETED
                record.progress.percent = 100.0
                record.progress.message = "Complete"
                record.progress.stage = "done"
        except Exception as exc:
            with _lock:
                record.status = JobStatus.FAILED
                record.error = str(exc)
                record.progress.message = f"Failed: {exc}"
                record.progress.stage = "error"
            traceback.print_exc()

    _executor.submit(run)
    return job_id


def submit_job(tool: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> dict[str, str]:
    return {"job_id": start_job(tool, fn, *args, **kwargs)}


def get_job(job_id: str) -> JobRecord:
    with _lock:
        if job_id not in _jobs:
            raise KeyError(f"Unknown job: {job_id}")
        return _jobs[job_id]
