"""Tests for openot2.control.models – serialization, defaults, enums."""

import json
import uuid
from datetime import datetime, timezone

import pytest

from openot2.control.models import (
    RunEvent,
    RunStatus,
    RunStep,
    StepStatus,
    TaskRun,
)


# ---------------------------------------------------------------------------
# Enum behaviour
# ---------------------------------------------------------------------------

class TestRunStatus:
    def test_values_are_strings(self):
        for member in RunStatus:
            assert isinstance(member.value, str)

    def test_json_serializes_as_string(self):
        assert json.loads(json.dumps(RunStatus.running)) == "running"

    def test_minimum_members_present(self):
        expected = {
            "draft", "ready", "running", "pause_requested",
            "paused", "completed", "failed", "aborted",
        }
        assert expected <= {m.value for m in RunStatus}


class TestStepStatus:
    def test_values_are_strings(self):
        for member in StepStatus:
            assert isinstance(member.value, str)

    def test_minimum_members_present(self):
        expected = {"pending", "running", "succeeded", "failed", "skipped"}
        assert expected <= {m.value for m in StepStatus}


# ---------------------------------------------------------------------------
# RunStep
# ---------------------------------------------------------------------------

class TestRunStep:
    def test_defaults(self):
        step = RunStep(name="transfer", kind="liquid_handling")
        assert step.status == StepStatus.pending
        assert step.params == {}
        assert step.checkpoint is True
        assert step.started_at is None
        assert step.finished_at is None
        assert step.duration_seconds is None
        assert step.output is None
        assert step.error is None
        # id should be a valid uuid
        uuid.UUID(step.id)

    def test_round_trip_json(self):
        step = RunStep(
            name="measure",
            kind="sensor_read",
            params={"sensor": "temperature"},
            status=StepStatus.succeeded,
            output={"value": 37.2},
        )
        payload = step.model_dump_json()
        restored = RunStep.model_validate_json(payload)
        assert restored == step

    def test_custom_id_preserved(self):
        step = RunStep(id="my-step-1", name="a", kind="b")
        assert step.id == "my-step-1"


# ---------------------------------------------------------------------------
# TaskRun
# ---------------------------------------------------------------------------

class TestTaskRun:
    def test_defaults(self):
        run = TaskRun(name="demo-run")
        assert run.status == RunStatus.draft
        assert run.steps == []
        assert run.current_step_index == 0
        assert isinstance(run.created_at, datetime)
        assert isinstance(run.updated_at, datetime)
        assert run.metadata == {}
        assert run.eta_seconds is None
        uuid.UUID(run.id)

    def test_round_trip_json(self):
        run = TaskRun(
            name="full-run",
            status=RunStatus.running,
            steps=[
                RunStep(name="s1", kind="prep"),
                RunStep(name="s2", kind="execute"),
            ],
            current_step_index=1,
            metadata={"operator": "test-suite"},
            eta_seconds=120.5,
        )
        payload = run.model_dump_json()
        restored = TaskRun.model_validate_json(payload)
        assert restored.name == run.name
        assert restored.status == RunStatus.running
        assert len(restored.steps) == 2
        assert restored.steps[0].name == "s1"
        assert restored.eta_seconds == 120.5

    def test_created_at_is_utc(self):
        run = TaskRun(name="tz-check")
        assert run.created_at.tzinfo is not None

    def test_model_dump_contains_enum_values(self):
        run = TaskRun(name="x")
        d = run.model_dump()
        assert d["status"] == "draft"


# ---------------------------------------------------------------------------
# RunEvent
# ---------------------------------------------------------------------------

class TestRunEvent:
    def test_defaults(self):
        evt = RunEvent(run_id="r1", type="info", message="hello")
        uuid.UUID(evt.id)
        assert isinstance(evt.timestamp, datetime)
        assert evt.payload is None

    def test_round_trip_json(self):
        evt = RunEvent(
            run_id="r1",
            type="step_completed",
            message="Step 1 done",
            payload={"step_index": 0},
        )
        payload = evt.model_dump_json()
        restored = RunEvent.model_validate_json(payload)
        assert restored == evt

    def test_timestamp_is_utc(self):
        evt = RunEvent(run_id="r1", type="t", message="m")
        assert evt.timestamp.tzinfo is not None
