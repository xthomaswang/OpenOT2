"""Tests for step output binding ($ref resolution) in TaskRunner."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

import pytest

from openot2.control.models import RunStep, RunStatus, StepStatus, TaskRun
from openot2.control.runner import RefResolutionError, TaskRunner
from openot2.control.store import JsonRunStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now() -> datetime:
    return datetime.now(timezone.utc)


def _make_step(
    name: str,
    kind: str = "generic",
    key: str | None = None,
    params: dict | None = None,
    checkpoint: bool = True,
) -> RunStep:
    return RunStep(
        id=uuid.uuid4().hex,
        name=name,
        kind=kind,
        key=key,
        params=params or {},
        checkpoint=checkpoint,
    )


def _make_run(steps: list[RunStep], name: str = "test-run") -> TaskRun:
    now = _now()
    return TaskRun(
        id=uuid.uuid4().hex,
        name=name,
        steps=steps,
        created_at=now,
        updated_at=now,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def store(tmp_path):
    return JsonRunStore(base_dir=tmp_path)


# ---------------------------------------------------------------------------
# Basic $ref resolution
# ---------------------------------------------------------------------------

class TestRefResolution:
    """Verify that $ref placeholders in params are replaced before handler runs."""

    def test_simple_ref_resolved(self, store):
        """Step B reads step A's output via $ref."""
        def capture_handler(step: RunStep, ctx: Any = None) -> dict:
            return {"image_path": "/tmp/capture.jpg"}

        received = {}

        def process_handler(step: RunStep, ctx: Any = None) -> dict:
            received["params"] = dict(step.params)
            return {"done": True}

        runner = TaskRunner(store=store, handlers={
            "capture": capture_handler,
            "process": process_handler,
        })

        steps = [
            _make_step("capture", kind="capture", key="cap"),
            _make_step("process", kind="process", params={
                "input_image": {"$ref": "cap.output.image_path"},
            }),
        ]
        run = _make_run(steps)
        store.create_run(run)

        result = runner.run_until_pause_or_done(run.id)

        assert result.status == RunStatus.completed
        assert received["params"]["input_image"] == "/tmp/capture.jpg"

    def test_nested_ref_in_list(self, store):
        """$ref inside a list element is resolved."""
        def step_a(step, ctx=None):
            return {"value": 42}

        received = {}

        def step_b(step, ctx=None):
            received["params"] = dict(step.params)
            return {}

        runner = TaskRunner(store=store, handlers={
            "produce": step_a,
            "consume": step_b,
        })

        steps = [
            _make_step("a", kind="produce", key="a"),
            _make_step("b", kind="consume", params={
                "values": [1, {"$ref": "a.output.value"}, 3],
            }),
        ]
        run = _make_run(steps)
        store.create_run(run)

        result = runner.run_until_pause_or_done(run.id)

        assert result.status == RunStatus.completed
        assert received["params"]["values"] == [1, 42, 3]

    def test_nested_ref_in_dict(self, store):
        """$ref deeply nested inside dicts is resolved."""
        def step_a(step, ctx=None):
            return {"host": "robot.local", "port": 8080}

        received = {}

        def step_b(step, ctx=None):
            received["params"] = step.params
            return {}

        runner = TaskRunner(store=store, handlers={
            "discover": step_a,
            "connect": step_b,
        })

        steps = [
            _make_step("discover", kind="discover", key="disc"),
            _make_step("connect", kind="connect", params={
                "connection": {
                    "host": {"$ref": "disc.output.host"},
                    "port": {"$ref": "disc.output.port"},
                },
            }),
        ]
        run = _make_run(steps)
        store.create_run(run)

        result = runner.run_until_pause_or_done(run.id)

        assert result.status == RunStatus.completed
        assert received["params"]["connection"]["host"] == "robot.local"
        assert received["params"]["connection"]["port"] == 8080

    def test_deep_output_path(self, store):
        """$ref can drill into nested output dicts via dotted path."""
        def step_a(step, ctx=None):
            return {"result": {"nested": {"deep": "found"}}}

        received = {}

        def step_b(step, ctx=None):
            received["val"] = step.params["val"]
            return {}

        runner = TaskRunner(store=store, handlers={
            "produce": step_a,
            "consume": step_b,
        })

        steps = [
            _make_step("a", kind="produce", key="a"),
            _make_step("b", kind="consume", params={
                "val": {"$ref": "a.output.result.nested.deep"},
            }),
        ]
        run = _make_run(steps)
        store.create_run(run)

        result = runner.run_until_pause_or_done(run.id)

        assert result.status == RunStatus.completed
        assert received["val"] == "found"

    def test_no_ref_params_unchanged(self, store):
        """Params without $ref are passed through unchanged."""
        received = {}

        def handler(step, ctx=None):
            received["params"] = dict(step.params)
            return {"ok": True}

        runner = TaskRunner(store=store, handlers={"generic": handler})

        steps = [_make_step("s", params={"x": 1, "y": "hello", "z": [1, 2]})]
        run = _make_run(steps)
        store.create_run(run)

        runner.run_until_pause_or_done(run.id)

        assert received["params"] == {"x": 1, "y": "hello", "z": [1, 2]}

    def test_multiple_refs_same_step(self, store):
        """Multiple $ref values referencing the same step are all resolved."""
        def step_a(step, ctx=None):
            return {"x": 10, "y": 20}

        received = {}

        def step_b(step, ctx=None):
            received["params"] = dict(step.params)
            return {}

        runner = TaskRunner(store=store, handlers={
            "produce": step_a,
            "consume": step_b,
        })

        steps = [
            _make_step("a", kind="produce", key="a"),
            _make_step("b", kind="consume", params={
                "first": {"$ref": "a.output.x"},
                "second": {"$ref": "a.output.y"},
            }),
        ]
        run = _make_run(steps)
        store.create_run(run)

        result = runner.run_until_pause_or_done(run.id)

        assert result.status == RunStatus.completed
        assert received["params"]["first"] == 10
        assert received["params"]["second"] == 20

    def test_chain_of_refs(self, store):
        """Step C reads step B's output, which was produced using step A's output."""
        def step_a(step, ctx=None):
            return {"base": 100}

        def step_b(step, ctx=None):
            return {"doubled": step.params["val"] * 2}

        received = {}

        def step_c(step, ctx=None):
            received["final"] = step.params["result"]
            return {}

        runner = TaskRunner(store=store, handlers={
            "a": step_a,
            "b": step_b,
            "c": step_c,
        })

        steps = [
            _make_step("step-a", kind="a", key="sa"),
            _make_step("step-b", kind="b", key="sb", params={
                "val": {"$ref": "sa.output.base"},
            }),
            _make_step("step-c", kind="c", params={
                "result": {"$ref": "sb.output.doubled"},
            }),
        ]
        run = _make_run(steps)
        store.create_run(run)

        result = runner.run_until_pause_or_done(run.id)

        assert result.status == RunStatus.completed
        assert received["final"] == 200


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------

class TestRefResolutionErrors:

    def test_missing_step_key_fails(self, store):
        """$ref to a non-existent step key marks run as failed."""
        def handler(step, ctx=None):
            return {}

        runner = TaskRunner(store=store, handlers={"generic": handler})

        steps = [
            _make_step("s", params={"val": {"$ref": "missing.output.field"}}),
        ]
        run = _make_run(steps)
        store.create_run(run)

        result = runner.run_until_pause_or_done(run.id)

        assert result.status == RunStatus.failed
        assert "missing" in (result.steps[0].error or "")

    def test_missing_output_field_fails(self, store):
        """$ref to a non-existent field in step output fails."""
        def step_a(step, ctx=None):
            return {"exists": True}

        def step_b(step, ctx=None):
            return {}

        runner = TaskRunner(store=store, handlers={
            "produce": step_a,
            "consume": step_b,
        })

        steps = [
            _make_step("a", kind="produce", key="a"),
            _make_step("b", kind="consume", params={
                "val": {"$ref": "a.output.nonexistent"},
            }),
        ]
        run = _make_run(steps)
        store.create_run(run)

        result = runner.run_until_pause_or_done(run.id)

        assert result.status == RunStatus.failed
        assert "nonexistent" in (result.steps[1].error or "")

    def test_invalid_ref_format_fails(self, store):
        """$ref with wrong format (no 'output' segment) fails."""
        def handler(step, ctx=None):
            return {}

        runner = TaskRunner(store=store, handlers={"generic": handler})

        steps = [
            _make_step("s", params={"val": {"$ref": "bad_format"}}),
        ]
        run = _make_run(steps)
        store.create_run(run)

        result = runner.run_until_pause_or_done(run.id)

        assert result.status == RunStatus.failed
        assert "Invalid $ref format" in (result.steps[0].error or "")


# ---------------------------------------------------------------------------
# RunStep key field
# ---------------------------------------------------------------------------

class TestRunStepKey:

    def test_key_default_none(self):
        step = RunStep(name="s", kind="k")
        assert step.key is None

    def test_key_round_trip_json(self):
        step = RunStep(name="s", kind="k", key="my_key")
        payload = step.model_dump_json()
        restored = RunStep.model_validate_json(payload)
        assert restored.key == "my_key"

    def test_key_in_task_run_round_trip(self, store):
        run = TaskRun(
            name="keyed-run",
            steps=[
                RunStep(name="s1", kind="a", key="alpha"),
                RunStep(name="s2", kind="b", key="beta"),
                RunStep(name="s3", kind="c"),
            ],
        )
        store.create_run(run)
        loaded = store.load_run(run.id)
        assert loaded.steps[0].key == "alpha"
        assert loaded.steps[1].key == "beta"
        assert loaded.steps[2].key is None


# ---------------------------------------------------------------------------
# Output stored correctly through binding flow
# ---------------------------------------------------------------------------

class TestOutputStoreIntegration:

    def test_output_persisted_after_binding_run(self, store):
        """Step outputs are persisted to the store even when bindings are used."""
        def step_a(step, ctx=None):
            return {"image_path": "/tmp/img.jpg"}

        def step_b(step, ctx=None):
            return {"processed": True, "source": step.params["img"]}

        runner = TaskRunner(store=store, handlers={
            "capture": step_a,
            "process": step_b,
        })

        steps = [
            _make_step("capture", kind="capture", key="cap"),
            _make_step("process", kind="process", key="proc", params={
                "img": {"$ref": "cap.output.image_path"},
            }),
        ]
        run = _make_run(steps)
        store.create_run(run)

        runner.run_until_pause_or_done(run.id)

        loaded = store.load_run(run.id)
        assert loaded.status == RunStatus.completed
        assert loaded.steps[0].output == {"image_path": "/tmp/img.jpg"}
        assert loaded.steps[1].output == {"processed": True, "source": "/tmp/img.jpg"}


# ---------------------------------------------------------------------------
# Binding + pause/resume
# ---------------------------------------------------------------------------

class TestBindingWithPauseResume:

    def test_binding_works_after_resume(self, store):
        """Refs are resolved correctly even when the run was paused and resumed."""
        call_count = {"n": 0}

        def capture_handler(step, ctx=None):
            return {"path": "/data/frame.png"}

        runner = TaskRunner(store=store)

        def pause_then_capture(step, ctx=None):
            call_count["n"] += 1
            if call_count["n"] == 1:
                runner.request_pause(step.params.get("_run_id", ""))
            return {"path": "/data/frame.png"}

        received = {}

        def process_handler(step, ctx=None):
            received["img"] = step.params["img"]
            return {"done": True}

        # Use a simpler approach: capture produces output, then pause before process
        runner.register_handler("capture", capture_handler)
        runner.register_handler("process", process_handler)

        steps = [
            _make_step("capture", kind="capture", key="cap", checkpoint=True),
            _make_step("process", kind="process", checkpoint=True, params={
                "img": {"$ref": "cap.output.path"},
            }),
        ]
        run = _make_run(steps)
        store.create_run(run)

        # Run capture, then request pause
        runner.register_handler("capture", lambda step, ctx=None: (
            runner.request_pause(run.id),
            {"path": "/data/frame.png"},
        )[-1])

        paused = runner.run_until_pause_or_done(run.id)
        assert paused.status == RunStatus.paused
        assert paused.steps[0].status == StepStatus.succeeded
        assert paused.steps[0].output == {"path": "/data/frame.png"}

        # Resume — process step should resolve the $ref from capture's output
        resumed = runner.resume(run.id)

        assert resumed.status == RunStatus.completed
        assert received["img"] == "/data/frame.png"
