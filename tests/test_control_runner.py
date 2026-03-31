"""Tests for the generic TaskRunner (safe-point pause/resume, ETA, events)."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

import pytest

from openot2.control.models import (
    RunEvent,
    RunStatus,
    RunStep,
    StepStatus,
    TaskRun,
)
from openot2.control.store import JsonRunStore
from openot2.control.runner import HandlerNotFoundError, TaskRunner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now() -> datetime:
    return datetime.now(timezone.utc)


def _make_step(name: str, kind: str = "generic", checkpoint: bool = True) -> RunStep:
    return RunStep(id=uuid.uuid4().hex, name=name, kind=kind, checkpoint=checkpoint)


def _make_run(steps: list[RunStep] | None = None, name: str = "test-run") -> TaskRun:
    now = _now()
    return TaskRun(
        id=uuid.uuid4().hex,
        name=name,
        steps=steps or [],
        created_at=now,
        updated_at=now,
    )


def _ok_handler(step: RunStep, context: Any = None) -> dict:
    """Trivial handler that always succeeds."""
    return {"result": "ok"}


def _fail_handler(step: RunStep, context: Any = None) -> dict:
    raise RuntimeError("boom")


def _pause_after_first_handler(runner: TaskRunner, run_id: str):
    """Return a handler that requests a pause the first time it runs."""
    call_count = {"n": 0}

    def handler(step: RunStep, context: Any = None) -> dict:
        call_count["n"] += 1
        if call_count["n"] == 1:
            runner.request_pause(run_id)
        return {"call": call_count["n"]}

    return handler


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def store(tmp_path):
    return JsonRunStore(base_dir=tmp_path)


@pytest.fixture()
def runner(store):
    return TaskRunner(store=store, handlers={"generic": _ok_handler})


# ---------------------------------------------------------------------------
# Full lifecycle: run to completion
# ---------------------------------------------------------------------------

class TestFullLifecycle:

    def test_run_completes_all_steps(self, store, runner):
        steps = [_make_step(f"step-{i}") for i in range(3)]
        run = _make_run(steps=steps)
        store.create_run(run)

        result = runner.run_until_pause_or_done(run.id)

        assert result.status == RunStatus.completed
        assert result.current_step_index == 3
        for s in result.steps:
            assert s.status == StepStatus.succeeded
            assert s.duration_seconds is not None
            assert s.output == {"result": "ok"}

    def test_run_emits_expected_events(self, store, runner):
        steps = [_make_step("only-step")]
        run = _make_run(steps=steps)
        store.create_run(run)

        runner.run_until_pause_or_done(run.id)

        events = store.list_events(run.id)
        types = [e.type for e in events]
        assert "run_started" in types
        assert "step_started" in types
        assert "step_succeeded" in types
        assert "run_completed" in types

    def test_empty_run_completes_immediately(self, store, runner):
        run = _make_run(steps=[])
        store.create_run(run)

        result = runner.run_until_pause_or_done(run.id)
        assert result.status == RunStatus.completed

    def test_eta_zero_after_completion(self, store, runner):
        steps = [_make_step("s")]
        run = _make_run(steps=steps)
        store.create_run(run)

        result = runner.run_until_pause_or_done(run.id)
        assert result.eta_seconds == 0.0


# ---------------------------------------------------------------------------
# Pause / resume
# ---------------------------------------------------------------------------

class TestPauseResume:

    def test_pause_at_checkpoint(self, store):
        """Requesting pause during step-0 causes run to pause before step-1."""
        steps = [
            _make_step("step-0", checkpoint=True),
            _make_step("step-1", checkpoint=True),
            _make_step("step-2", checkpoint=True),
        ]
        run = _make_run(steps=steps)
        store.create_run(run)

        runner = TaskRunner(store=store)
        handler = _pause_after_first_handler(runner, run.id)
        runner.register_handler("generic", handler)

        result = runner.run_until_pause_or_done(run.id)

        assert result.status == RunStatus.paused
        # step-0 finished successfully, step-1 not yet started
        assert result.steps[0].status == StepStatus.succeeded
        assert result.steps[1].status == StepStatus.pending
        assert result.current_step_index == 1

    def test_resume_after_pause(self, store):
        steps = [
            _make_step("step-0", checkpoint=True),
            _make_step("step-1", checkpoint=True),
        ]
        run = _make_run(steps=steps)
        store.create_run(run)

        runner = TaskRunner(store=store)
        handler = _pause_after_first_handler(runner, run.id)
        runner.register_handler("generic", handler)

        paused = runner.run_until_pause_or_done(run.id)
        assert paused.status == RunStatus.paused

        # Replace handler with plain ok handler for resume
        runner.register_handler("generic", _ok_handler)
        resumed = runner.resume(run.id)

        assert resumed.status == RunStatus.completed
        assert all(s.status == StepStatus.succeeded for s in resumed.steps)

    def test_non_checkpoint_step_does_not_trigger_pause(self, store):
        """A non-checkpoint step must NOT cause a pause even if pause_requested."""
        steps = [
            _make_step("step-0", checkpoint=True),
            _make_step("step-1", checkpoint=False),  # should NOT pause here
            _make_step("step-2", checkpoint=True),    # should pause here
        ]
        run = _make_run(steps=steps)
        store.create_run(run)

        runner = TaskRunner(store=store)
        handler = _pause_after_first_handler(runner, run.id)
        runner.register_handler("generic", handler)

        result = runner.run_until_pause_or_done(run.id)

        assert result.status == RunStatus.paused
        assert result.steps[0].status == StepStatus.succeeded
        assert result.steps[1].status == StepStatus.succeeded  # ran despite pause_requested
        assert result.steps[2].status == StepStatus.pending     # paused before this one
        assert result.current_step_index == 2

    def test_resume_non_paused_raises(self, store, runner):
        run = _make_run(steps=[_make_step("s")])
        store.create_run(run)

        with pytest.raises(RuntimeError, match="expected paused"):
            runner.resume(run.id)

    def test_pause_events_emitted(self, store):
        steps = [_make_step("s0"), _make_step("s1")]
        run = _make_run(steps=steps)
        store.create_run(run)

        runner = TaskRunner(store=store)
        handler = _pause_after_first_handler(runner, run.id)
        runner.register_handler("generic", handler)

        runner.run_until_pause_or_done(run.id)

        events = store.list_events(run.id)
        types = [e.type for e in events]
        assert "pause_requested" in types
        assert "run_paused" in types


# ---------------------------------------------------------------------------
# ETA
# ---------------------------------------------------------------------------

class TestETA:

    def test_estimate_eta_no_completed(self, runner):
        run = _make_run(steps=[_make_step("s")])
        assert runner.estimate_eta(run) is None

    def test_estimate_eta_with_completed(self, runner):
        s0 = _make_step("s0")
        s0.status = StepStatus.succeeded
        s0.duration_seconds = 2.0

        s1 = _make_step("s1")
        s1.status = StepStatus.succeeded
        s1.duration_seconds = 4.0

        s2 = _make_step("s2")  # pending

        run = _make_run(steps=[s0, s1, s2])
        eta = runner.estimate_eta(run)
        # median of [2.0, 4.0] = 3.0; 1 remaining step -> 3.0
        assert eta == pytest.approx(3.0)

    def test_eta_updated_during_run(self, store, runner):
        steps = [_make_step(f"s{i}") for i in range(3)]
        run = _make_run(steps=steps)
        store.create_run(run)

        result = runner.run_until_pause_or_done(run.id)
        # After completion ETA should be 0
        assert result.eta_seconds == 0.0


# ---------------------------------------------------------------------------
# Failure handling
# ---------------------------------------------------------------------------

class TestFailure:

    def test_step_failure_marks_run_failed(self, store):
        runner = TaskRunner(store=store, handlers={"generic": _fail_handler})

        steps = [_make_step("bad-step")]
        run = _make_run(steps=steps)
        store.create_run(run)

        result = runner.run_until_pause_or_done(run.id)

        assert result.status == RunStatus.failed
        assert result.steps[0].status == StepStatus.failed
        assert result.steps[0].error == "boom"

    def test_failure_emits_events(self, store):
        runner = TaskRunner(store=store, handlers={"generic": _fail_handler})

        steps = [_make_step("bad")]
        run = _make_run(steps=steps)
        store.create_run(run)

        runner.run_until_pause_or_done(run.id)

        types = [e.type for e in store.list_events(run.id)]
        assert "step_failed" in types
        assert "run_failed" in types

    def test_missing_handler_raises(self, store):
        runner = TaskRunner(store=store)  # no handlers

        steps = [_make_step("s", kind="unknown")]
        run = _make_run(steps=steps)
        store.create_run(run)

        result = runner.run_until_pause_or_done(run.id)
        assert result.status == RunStatus.failed
        assert "No handler registered" in (result.steps[0].error or "")


# ---------------------------------------------------------------------------
# Handler registry
# ---------------------------------------------------------------------------

class TestHandlerRegistry:

    def test_register_and_use(self, store):
        runner = TaskRunner(store=store)
        runner.register_handler("custom", lambda step, ctx=None: {"custom": True})

        steps = [_make_step("s", kind="custom")]
        run = _make_run(steps=steps)
        store.create_run(run)

        result = runner.run_until_pause_or_done(run.id)
        assert result.status == RunStatus.completed
        assert result.steps[0].output == {"custom": True}

    def test_context_passed_to_handler(self, store):
        received = {}

        def capture(step: RunStep, context: Any = None) -> dict:
            received["ctx"] = context
            return {}

        runner = TaskRunner(store=store, handlers={"generic": capture})

        steps = [_make_step("s")]
        run = _make_run(steps=steps)
        store.create_run(run)

        runner.run_until_pause_or_done(run.id, context={"my": "data"})
        assert received["ctx"] == {"my": "data"}
