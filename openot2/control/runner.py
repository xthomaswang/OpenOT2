"""Generic sequential task runner with safe-point pause/resume and ETA."""

from __future__ import annotations

import statistics
import uuid
from datetime import datetime, timezone
from typing import Any, Callable

from openot2.control.models import (
    RunEvent,
    RunStatus,
    RunStep,
    StepStatus,
    TaskRun,
)
from openot2.control.store import JsonRunStore


class HandlerNotFoundError(Exception):
    """Raised when no handler is registered for a step kind."""


class TaskRunner:
    """Execute a sequence of :class:`RunStep` items using pluggable handlers.

    Handlers are plain callables keyed by ``step.kind``.  The runner walks the
    step list sequentially, honouring **safe-point pause** semantics: when a
    pause is requested the current step is allowed to finish, and the run
    transitions to *paused* before the next **checkpointed** step begins.

    Parameters
    ----------
    store:
        Persistence back-end (load / save runs, append events).
    handlers:
        Optional initial mapping of ``kind -> handler``.
    """

    def __init__(
        self,
        store: JsonRunStore,
        handlers: dict[str, Callable] | None = None,
    ) -> None:
        self._store = store
        self._handlers: dict[str, Callable] = dict(handlers) if handlers else {}

    # ------------------------------------------------------------------
    # Handler registry
    # ------------------------------------------------------------------

    def register_handler(self, kind: str, handler: Callable) -> None:
        """Register *handler* for steps whose ``kind`` matches *kind*."""
        self._handlers[kind] = handler

    # ------------------------------------------------------------------
    # Pause / resume
    # ------------------------------------------------------------------

    def request_pause(self, run_id: str) -> None:
        """Signal that the run should pause at the next safe-point."""
        run = self._store.load_run(run_id)
        if run.status not in (RunStatus.running, RunStatus.ready):
            return
        run.status = RunStatus.pause_requested
        run.updated_at = _now()
        self._store.save_run(run)
        self._emit(run_id, "pause_requested", "Pause requested by user")

    def resume(self, run_id: str, context: Any = None) -> TaskRun:
        """Resume a previously paused run and continue execution."""
        run = self._store.load_run(run_id)
        if run.status != RunStatus.paused:
            raise RuntimeError(
                f"Cannot resume run {run_id}: status is {run.status.value}, expected paused"
            )
        run.status = RunStatus.running
        run.updated_at = _now()
        self._store.save_run(run)
        self._emit(run_id, "run_resumed", "Run resumed")
        return self._execute_loop(run, context)

    # ------------------------------------------------------------------
    # Main execution
    # ------------------------------------------------------------------

    def run_until_pause_or_done(self, run_id: str, context: Any = None) -> TaskRun:
        """Start (or continue) executing *run_id* until paused or finished."""
        run = self._store.load_run(run_id)
        if run.status not in (RunStatus.draft, RunStatus.ready):
            raise RuntimeError(
                f"Cannot start run {run_id}: status is {run.status.value}"
            )
        run.status = RunStatus.running
        run.updated_at = _now()
        self._store.save_run(run)
        self._emit(run_id, "run_started", "Run started")
        return self._execute_loop(run, context)

    # ------------------------------------------------------------------
    # ETA
    # ------------------------------------------------------------------

    def estimate_eta(self, run: TaskRun) -> float | None:
        """Return estimated remaining seconds based on completed step durations."""
        durations = [
            s.duration_seconds
            for s in run.steps
            if s.status == StepStatus.succeeded and s.duration_seconds is not None
        ]
        if not durations:
            return None
        median = statistics.median(durations)
        remaining = sum(
            1 for s in run.steps if s.status in (StepStatus.pending, StepStatus.running)
        )
        return median * remaining

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _execute_loop(self, run: TaskRun, context: Any) -> TaskRun:
        """Walk through steps starting at ``current_step_index``."""
        while run.current_step_index < len(run.steps):
            step = run.steps[run.current_step_index]

            # --- safe-point pause check (before starting a checkpointed step) ---
            if run.status == RunStatus.pause_requested and step.checkpoint:
                run.status = RunStatus.paused
                run.updated_at = _now()
                run.eta_seconds = self.estimate_eta(run)
                self._store.save_run(run)
                self._emit(run.id, "run_paused", f"Run paused before step {step.name}")
                return run

            # --- execute the step ---
            try:
                self._run_step(run, step, context)
            except Exception as exc:
                # Step failure -> mark run as failed
                step.status = StepStatus.failed
                step.finished_at = _now()
                step.error = str(exc)
                if step.started_at:
                    step.duration_seconds = (
                        step.finished_at - step.started_at
                    ).total_seconds()
                run.status = RunStatus.failed
                run.updated_at = _now()
                run.eta_seconds = None
                self._store.save_run(run)
                self._emit(
                    run.id,
                    "step_failed",
                    f"Step {step.name} failed: {exc}",
                    {"step_id": step.id, "error": str(exc)},
                )
                self._emit(run.id, "run_failed", f"Run failed at step {step.name}")
                return run

            # Advance index
            run.current_step_index += 1
            run.eta_seconds = self.estimate_eta(run)
            run.updated_at = _now()

            # Reload status from store to pick up external pause requests
            # (e.g. request_pause called from a handler or another thread)
            # before we overwrite the stored run.
            persisted = self._store.load_run(run.id)
            if persisted.status == RunStatus.pause_requested:
                run.status = RunStatus.pause_requested

            self._store.save_run(run)

        # All steps done
        run.status = RunStatus.completed
        run.eta_seconds = 0.0
        run.updated_at = _now()
        self._store.save_run(run)
        self._emit(run.id, "run_completed", "Run completed successfully")
        return run

    def _run_step(self, run: TaskRun, step: RunStep, context: Any) -> None:
        """Execute a single step via its registered handler."""
        handler = self._handlers.get(step.kind)
        if handler is None:
            raise HandlerNotFoundError(
                f"No handler registered for step kind '{step.kind}'"
            )

        step.status = StepStatus.running
        step.started_at = _now()
        # Only reset to running if not already pause_requested — a non-checkpoint
        # step should preserve the pause_requested flag for the next checkpoint.
        if run.status != RunStatus.pause_requested:
            run.status = RunStatus.running
        run.updated_at = _now()
        self._store.save_run(run)
        self._emit(
            run.id, "step_started", f"Step {step.name} started", {"step_id": step.id}
        )

        result = handler(step, context)

        step.status = StepStatus.succeeded
        step.finished_at = _now()
        step.duration_seconds = (step.finished_at - step.started_at).total_seconds()
        step.output = result if isinstance(result, dict) else None
        self._emit(
            run.id,
            "step_succeeded",
            f"Step {step.name} succeeded",
            {"step_id": step.id, "duration": step.duration_seconds},
        )

    def _emit(
        self,
        run_id: str,
        event_type: str,
        message: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Create and persist a :class:`RunEvent`."""
        event = RunEvent(
            id=uuid.uuid4().hex,
            run_id=run_id,
            type=event_type,
            message=message,
            timestamp=_now(),
            payload=payload,
        )
        self._store.append_event(event)


def _now() -> datetime:
    """UTC-aware current timestamp."""
    return datetime.now(timezone.utc)
