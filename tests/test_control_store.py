"""Tests for openot2.control.store.JsonRunStore."""

from pathlib import Path

import pytest

from openot2.control.models import (
    RunEvent,
    RunStatus,
    RunStep,
    StepStatus,
    TaskRun,
)
from openot2.control.store import JsonRunStore


@pytest.fixture()
def store(tmp_path: Path) -> JsonRunStore:
    return JsonRunStore(base_dir=tmp_path)


# ---------------------------------------------------------------------------
# Run CRUD
# ---------------------------------------------------------------------------

class TestRunCRUD:
    def test_create_and_load(self, store: JsonRunStore):
        run = TaskRun(name="test-run")
        created = store.create_run(run)
        assert created.id == run.id

        loaded = store.load_run(run.id)
        assert loaded.name == "test-run"
        assert loaded.id == run.id

    def test_save_overwrites(self, store: JsonRunStore):
        run = TaskRun(name="original")
        store.create_run(run)

        run.status = RunStatus.running
        run.current_step_index = 2
        store.save_run(run)

        loaded = store.load_run(run.id)
        assert loaded.status == RunStatus.running
        assert loaded.current_step_index == 2

    def test_load_missing_raises(self, store: JsonRunStore):
        with pytest.raises(FileNotFoundError):
            store.load_run("nonexistent-id")

    def test_list_runs_empty(self, store: JsonRunStore):
        assert store.list_runs() == []

    def test_list_runs_multiple(self, store: JsonRunStore):
        for i in range(3):
            store.create_run(TaskRun(name=f"run-{i}"))
        runs = store.list_runs()
        assert len(runs) == 3
        names = {r.name for r in runs}
        assert names == {"run-0", "run-1", "run-2"}

    def test_run_with_steps_round_trip(self, store: JsonRunStore):
        run = TaskRun(
            name="with-steps",
            steps=[
                RunStep(name="s1", kind="prep", params={"key": "val"}),
                RunStep(name="s2", kind="exec", status=StepStatus.succeeded),
            ],
            current_step_index=1,
            metadata={"tag": "integration"},
        )
        store.create_run(run)
        loaded = store.load_run(run.id)
        assert len(loaded.steps) == 2
        assert loaded.steps[0].params == {"key": "val"}
        assert loaded.steps[1].status == StepStatus.succeeded
        assert loaded.metadata["tag"] == "integration"


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

class TestEvents:
    def test_append_and_list(self, store: JsonRunStore):
        run_id = "evt-run-1"
        store.append_event(RunEvent(run_id=run_id, type="info", message="started"))
        store.append_event(RunEvent(run_id=run_id, type="info", message="step done"))

        events = store.list_events(run_id)
        assert len(events) == 2
        assert events[0].message == "started"
        assert events[1].message == "step done"

    def test_list_events_missing_run(self, store: JsonRunStore):
        assert store.list_events("no-such-run") == []

    def test_event_payload_preserved(self, store: JsonRunStore):
        run_id = "evt-run-2"
        store.append_event(
            RunEvent(
                run_id=run_id,
                type="result",
                message="ok",
                payload={"value": 42},
            )
        )
        events = store.list_events(run_id)
        assert events[0].payload == {"value": 42}


# ---------------------------------------------------------------------------
# Directory structure
# ---------------------------------------------------------------------------

class TestDirectoryLayout:
    def test_directories_created(self, store: JsonRunStore):
        assert (store.base_dir / "runs").is_dir()
        assert (store.base_dir / "events").is_dir()

    def test_run_file_is_json(self, store: JsonRunStore):
        run = TaskRun(name="file-check")
        store.create_run(run)
        path = store.base_dir / "runs" / f"{run.id}.json"
        assert path.exists()
        assert path.suffix == ".json"

    def test_event_file_is_jsonl(self, store: JsonRunStore):
        run_id = "jsonl-check"
        store.append_event(RunEvent(run_id=run_id, type="t", message="m"))
        path = store.base_dir / "events" / f"{run_id}.jsonl"
        assert path.exists()
        assert path.suffix == ".jsonl"
