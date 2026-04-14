"""Generic run / step / event data models for the task controller."""

from __future__ import annotations

import enum
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class RunStatus(str, enum.Enum):
    """High-level lifecycle status of a task run."""

    draft = "draft"
    ready = "ready"
    running = "running"
    pause_requested = "pause_requested"
    paused = "paused"
    completed = "completed"
    failed = "failed"
    aborted = "aborted"


class StepStatus(str, enum.Enum):
    """Status of an individual step within a run."""

    pending = "pending"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"
    skipped = "skipped"


# ---------------------------------------------------------------------------
# Helper factories (kept private)
# ---------------------------------------------------------------------------

def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class RunStep(BaseModel):
    """A single discrete step inside a :class:`TaskRun`.

    Attributes
    ----------
    key:
        Stable, user-defined identifier for this step.  Other steps can
        reference this step's output via ``{"$ref": "<key>.output.<field>"}``.
        Unlike ``id`` (random UUID), ``key`` is deterministic and survives
        serialisation round-trips, making it suitable for cross-step bindings.
    """

    id: str = Field(default_factory=_uuid)
    key: Optional[str] = None
    name: str
    kind: str
    params: dict[str, Any] = Field(default_factory=dict)
    checkpoint: bool = True
    status: StepStatus = StepStatus.pending
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    output: Optional[dict[str, Any]] = None
    error: Optional[str] = None


class TaskRun(BaseModel):
    """Top-level representation of a task run."""

    id: str = Field(default_factory=_uuid)
    name: str
    status: RunStatus = RunStatus.draft
    steps: list[RunStep] = Field(default_factory=list)
    current_step_index: int = 0
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)
    metadata: dict[str, Any] = Field(default_factory=dict)
    sequence_id: Optional[str] = None
    eta_seconds: Optional[float] = None


class RunSequence(BaseModel):
    """A logical grouping of related :class:`TaskRun` instances."""

    id: str = Field(default_factory=_uuid)
    name: str
    status: str = "pending"  # pending, running, completed, failed, paused
    run_ids: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RunEvent(BaseModel):
    """An immutable log entry associated with a run."""

    id: str = Field(default_factory=_uuid)
    run_id: str
    type: str
    message: str
    timestamp: datetime = Field(default_factory=_now)
    payload: Optional[dict[str, Any]] = None
