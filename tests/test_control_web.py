"""Tests for the OpenOT2 web control app — fully hardware-free."""

from __future__ import annotations

import threading
import time
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from openot2.webapp.calibration import (
    CalibrationProfile,
    CalibrationSession,
    Offset,
    build_target,
    save_profile,
)
from openot2.webapp.handlers import OT2StepHandlers
from openot2.control.models import RunStatus, RunStep, TaskRun
from openot2.control.runner import TaskRunner
from openot2.control.store import JsonRunStore
from openot2.webapp.web import create_app
from openot2.webapp.app import WebApp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ok_handler(step: RunStep, context: Any = None) -> dict:
    return {"result": "ok"}


def _slow_handler(step: RunStep, context: Any = None) -> dict:
    """Handler that takes a short moment so background thread tests work."""
    time.sleep(0.05)
    return {"result": "ok"}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def store(tmp_path):
    return JsonRunStore(base_dir=tmp_path)


@pytest.fixture()
def runner(store):
    return TaskRunner(store=store, handlers={"generic": _ok_handler})


@pytest.fixture()
def app(store, runner):
    return create_app(store=store, runner=runner, client=None)


@pytest.fixture()
def client(app):
    return TestClient(app)


# ---------------------------------------------------------------------------
# HTML pages
# ---------------------------------------------------------------------------


class TestHTMLPages:
    def test_index_returns_200(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]

    def test_index_shows_run(self, store, client):
        store.create_run(TaskRun(name="my-run"))
        resp = client.get("/")
        assert resp.status_code == 200
        assert "my-run" in resp.text

    def test_run_detail_returns_200(self, store, client):
        run = TaskRun(name="detail-run")
        store.create_run(run)
        resp = client.get(f"/runs/{run.id}")
        assert resp.status_code == 200
        assert "detail-run" in resp.text

    def test_run_detail_404(self, client):
        resp = client.get("/runs/nonexistent-id")
        assert resp.status_code == 404

    def test_calibration_returns_200(self, client):
        resp = client.get("/calibration")
        assert resp.status_code == 200
        assert "Calibration" in resp.text

    def test_calibration_no_profile_shows_setup(self, client):
        """When no profile is loaded, the setup forms are shown."""
        resp = client.get("/calibration")
        assert resp.status_code == 200
        assert "Load existing profile" in resp.text
        assert "Create new profile" in resp.text

    def test_calibration_page_shows_initial_profile(self, store, runner):
        profile = CalibrationProfile(
            name="autoloaded",
            targets=[build_target("src", "3", "A1", "aspirate")],
        )
        app = create_app(
            store=store,
            runner=runner,
            client=None,
            initial_calibration_profile=profile,
            initial_calibration_session=CalibrationSession(profile_id=profile.id, status="active"),
        )
        tc = TestClient(app)
        resp = tc.get("/calibration")
        assert resp.status_code == 200
        assert "autoloaded" in resp.text


# ---------------------------------------------------------------------------
# JSON API — runs
# ---------------------------------------------------------------------------


class TestRunAPI:
    def test_create_run(self, client):
        resp = client.post(
            "/api/runs",
            json={
                "name": "test-run",
                "steps": [
                    {"name": "step1", "kind": "generic"},
                    {"name": "step2", "kind": "generic"},
                ],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "test-run"
        assert len(data["steps"]) == 2
        assert data["status"] == "draft"

    def test_get_run(self, store, client):
        run = TaskRun(name="get-me", steps=[RunStep(name="s1", kind="generic")])
        store.create_run(run)

        resp = client.get(f"/api/runs/{run.id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "get-me"
        assert data["id"] == run.id

    def test_get_run_404(self, client):
        resp = client.get("/api/runs/nonexistent")
        assert resp.status_code == 404

    def test_get_events(self, store, client):
        run = TaskRun(name="evt-run")
        store.create_run(run)
        resp = client.get(f"/api/runs/{run.id}/events")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_start_run(self, store, client):
        run = TaskRun(
            name="start-me",
            steps=[RunStep(name="s1", kind="generic")],
        )
        store.create_run(run)

        resp = client.post(f"/api/runs/{run.id}/start")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "started"

        # Wait for background thread to finish
        time.sleep(0.3)

        # Verify the run completed
        loaded = store.load_run(run.id)
        assert loaded.status.value == "completed"

    def test_start_run_404(self, client):
        resp = client.post("/api/runs/nonexistent/start")
        assert resp.status_code == 404

    def test_pause_running_run(self, store, client):
        run = TaskRun(name="pause-me", status=RunStatus.running)
        store.create_run(run)

        resp = client.post(f"/api/runs/{run.id}/pause")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "pause_requested"

    def test_pause_run_404(self, client):
        resp = client.post("/api/runs/nonexistent/pause")
        assert resp.status_code == 404

    def test_resume_run_404(self, client):
        resp = client.post("/api/runs/nonexistent/resume")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# JSON API — run state validation
# ---------------------------------------------------------------------------


class TestRunStateValidation:
    def test_start_completed_run_returns_409(self, store, client):
        run = TaskRun(name="done", status=RunStatus.completed)
        store.create_run(run)
        resp = client.post(f"/api/runs/{run.id}/start")
        assert resp.status_code == 409
        assert "completed" in resp.json()["detail"]

    def test_start_failed_run_returns_409(self, store, client):
        run = TaskRun(name="broken", status=RunStatus.failed)
        store.create_run(run)
        resp = client.post(f"/api/runs/{run.id}/start")
        assert resp.status_code == 409

    def test_start_running_run_returns_409(self, store, client):
        run = TaskRun(name="active", status=RunStatus.running)
        store.create_run(run)
        resp = client.post(f"/api/runs/{run.id}/start")
        assert resp.status_code == 409

    def test_resume_draft_run_returns_409(self, store, client):
        run = TaskRun(name="not-paused")
        store.create_run(run)
        resp = client.post(f"/api/runs/{run.id}/resume")
        assert resp.status_code == 409
        assert "draft" in resp.json()["detail"]

    def test_resume_completed_run_returns_409(self, store, client):
        run = TaskRun(name="already-done", status=RunStatus.completed)
        store.create_run(run)
        resp = client.post(f"/api/runs/{run.id}/resume")
        assert resp.status_code == 409

    def test_resume_running_run_returns_409(self, store, client):
        run = TaskRun(name="still-running", status=RunStatus.running)
        store.create_run(run)
        resp = client.post(f"/api/runs/{run.id}/resume")
        assert resp.status_code == 409

    def test_pause_draft_run_returns_409(self, store, client):
        run = TaskRun(name="not-started")
        store.create_run(run)
        resp = client.post(f"/api/runs/{run.id}/pause")
        assert resp.status_code == 409
        assert "draft" in resp.json()["detail"]

    def test_pause_ready_run_returns_409(self, store, client):
        run = TaskRun(name="not-yet-running", status=RunStatus.ready)
        store.create_run(run)
        resp = client.post(f"/api/runs/{run.id}/pause")
        assert resp.status_code == 409
        assert "ready" in resp.json()["detail"]

    def test_pause_completed_run_returns_409(self, store, client):
        run = TaskRun(name="already-done", status=RunStatus.completed)
        store.create_run(run)
        resp = client.post(f"/api/runs/{run.id}/pause")
        assert resp.status_code == 409

    def test_pause_failed_run_returns_409(self, store, client):
        run = TaskRun(name="broken", status=RunStatus.failed)
        store.create_run(run)
        resp = client.post(f"/api/runs/{run.id}/pause")
        assert resp.status_code == 409


# ---------------------------------------------------------------------------
# JSON API — calibration input validation
# ---------------------------------------------------------------------------


class TestCalibrationInputValidation:
    def test_load_profile_malformed_json(self, client, tmp_path):
        fp = tmp_path / "bad.json"
        fp.write_text("{ not valid json !!!", encoding="utf-8")
        resp = client.post("/api/calibration/load-profile", json={"path": str(fp)})
        assert resp.status_code == 400
        assert "Invalid profile file" in resp.json()["detail"]

    def test_load_profile_invalid_structure(self, client, tmp_path):
        fp = tmp_path / "wrong.json"
        fp.write_text('{"foo": "bar"}', encoding="utf-8")
        resp = client.post("/api/calibration/load-profile", json={"path": str(fp)})
        assert resp.status_code == 400
        assert "Invalid profile file" in resp.json()["detail"]

    def test_create_profile_invalid_target(self, client):
        resp = client.post("/api/calibration/create-profile", json={
            "name": "bad-profile",
            "targets": [{"bad_field": "oops"}],
        })
        assert resp.status_code == 400
        assert "Invalid target definition" in resp.json()["detail"]

    def test_create_profile_missing_required_fields(self, client):
        resp = client.post("/api/calibration/create-profile", json={
            "name": "incomplete",
            "targets": [{"name": "t1"}],
        })
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# JSON API — calibration (no client)
# ---------------------------------------------------------------------------


class TestCalibrationNoClient:
    def test_preview_503_without_client(self, client):
        resp = client.post(
            "/api/calibration/preview",
            json={"target_index": 0},
        )
        assert resp.status_code in (400, 503)

    def test_test_aspirate_503_without_client(self, client):
        resp = client.post(
            "/api/calibration/test-aspirate",
            json={"target_index": 0},
        )
        assert resp.status_code in (400, 503)

    def test_test_dispense_503_without_client(self, client):
        resp = client.post(
            "/api/calibration/test-dispense",
            json={"source_index": 0, "dest_index": 1},
        )
        assert resp.status_code in (400, 503)

    def test_save_profile_400_without_profile(self, client):
        resp = client.post(
            "/api/calibration/save-profile",
            json={"path": "/tmp/test.json"},
        )
        assert resp.status_code == 400

    def test_nudge_400_without_profile(self, client):
        resp = client.post(
            "/api/calibration/nudge",
            json={"target_index": 0, "axis": "x", "delta": 0.5},
        )
        assert resp.status_code == 400

    def test_list_profiles_empty(self, client):
        resp = client.get("/api/calibration/profiles")
        assert resp.status_code == 200
        assert resp.json() == []


# ---------------------------------------------------------------------------
# JSON API — calibration profile lifecycle (load/create via API)
# ---------------------------------------------------------------------------


class TestCalibrationProfileLifecycle:
    def test_load_profile_from_file(self, store, runner, tmp_path):
        """Load a profile via API, then calibration page shows targets."""
        profile = CalibrationProfile(
            name="loaded-profile",
            targets=[
                build_target("src", "3", "A1", "aspirate", volume=50.0),
                build_target("dst", "5", "B2", "dispense", volume=50.0),
            ],
        )
        fp = tmp_path / "profiles" / "test.json"
        save_profile(profile, fp)

        app = create_app(store=store, runner=runner, client=None)
        tc = TestClient(app)

        # Before loading, calibration shows setup
        resp = tc.get("/calibration")
        assert "Load existing profile" in resp.text

        # Load via API
        resp = tc.post("/api/calibration/load-profile", json={"path": str(fp)})
        assert resp.status_code == 200
        assert resp.json()["profile"] == "loaded-profile"
        assert resp.json()["targets"] == 2

        # Now calibration page shows targets
        resp = tc.get("/calibration")
        assert "src" in resp.text
        assert "dst" in resp.text

        # Nudge now works
        resp = tc.post(
            "/api/calibration/nudge",
            json={"target_index": 0, "axis": "z", "delta": 1.0},
        )
        assert resp.status_code == 200

    def test_load_profile_not_found(self, client):
        resp = client.post(
            "/api/calibration/load-profile",
            json={"path": "/nonexistent/profile.json"},
        )
        assert resp.status_code == 404

    def test_create_profile_via_api(self, store, runner):
        app = create_app(store=store, runner=runner, client=None)
        tc = TestClient(app)

        resp = tc.post("/api/calibration/create-profile", json={
            "name": "new-profile",
            "targets": [
                {"name": "src", "labware_slot": "3", "well": "A1", "action": "aspirate", "volume": 50},
                {"name": "dst", "labware_slot": "5", "well": "B2", "action": "dispense", "volume": 50},
            ],
        })
        assert resp.status_code == 200
        assert resp.json()["profile"] == "new-profile"
        assert resp.json()["targets"] == 2

        # Now calibration page shows targets
        resp = tc.get("/calibration")
        assert "src" in resp.text
        assert "dst" in resp.text

    def test_get_current_profile(self, store, runner):
        app = create_app(store=store, runner=runner, client=None)
        tc = TestClient(app)

        # No profile
        resp = tc.get("/api/calibration/profile")
        assert resp.status_code == 404

        # Create one
        tc.post("/api/calibration/create-profile", json={
            "name": "p1",
            "targets": [
                {"name": "t1", "labware_slot": "3", "well": "A1", "action": "aspirate"},
            ],
        })
        resp = tc.get("/api/calibration/profile")
        assert resp.status_code == 200
        assert resp.json()["name"] == "p1"


# ---------------------------------------------------------------------------
# JSON API — calibration with profile and mock client
# ---------------------------------------------------------------------------


class TestCalibrationWithClient:
    @pytest.fixture()
    def mock_client(self):
        c = MagicMock()
        c.get_labware_id.side_effect = lambda s: f"lw-{s}"
        return c

    @pytest.fixture()
    def app_with_client(self, store, runner, mock_client):
        app = create_app(store=store, runner=runner, client=mock_client)
        # Create profile via API
        tc = TestClient(app)
        tc.post("/api/calibration/create-profile", json={
            "name": "test-profile",
            "targets": [
                {"name": "tips", "labware_slot": "10", "well": "A1", "action": "pick_up_tip", "pipette_mount": "left"},
                {"name": "source", "labware_slot": "3", "well": "A1", "action": "aspirate", "volume": 50, "pipette_mount": "right"},
                {"name": "dest", "labware_slot": "5", "well": "B2", "action": "dispense", "volume": 50, "pipette_mount": "right"},
            ],
        })
        return app

    @pytest.fixture()
    def tc(self, app_with_client):
        return TestClient(app_with_client)

    def test_nudge_updates_offset(self, tc):
        resp = tc.post(
            "/api/calibration/nudge",
            json={"target_index": 0, "axis": "x", "delta": 1.5},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["offset"]["x"] == pytest.approx(1.5)
        assert data["offset"]["y"] == pytest.approx(0.0)

    def test_nudge_invalid_index(self, tc):
        resp = tc.post(
            "/api/calibration/nudge",
            json={"target_index": 99, "axis": "x", "delta": 1.0},
        )
        assert resp.status_code == 400

    def test_preview_calls_client(self, tc, mock_client):
        resp = tc.post("/api/calibration/preview", json={"target_index": 0})
        assert resp.status_code == 200
        mock_client.use_pipette.assert_called_once_with("left")
        mock_client.move_to_well.assert_called_once()

    def test_test_aspirate_calls_client(self, tc, mock_client):
        resp = tc.post("/api/calibration/test-aspirate", json={"target_index": 1})
        assert resp.status_code == 200
        mock_client.use_pipette.assert_called_once_with("right")
        mock_client.aspirate.assert_called_once()

    def test_test_pick_up_tip_calls_client(self, tc, mock_client):
        resp = tc.post("/api/calibration/test-pick-up-tip", json={"target_index": 0})
        assert resp.status_code == 200
        mock_client.use_pipette.assert_called_once_with("left")
        mock_client.pick_up_tip.assert_called_once()
        mock_client.drop_tip_in_trash.assert_not_called()

    def test_test_drop_tip_calls_client(self, tc, mock_client):
        resp = tc.post("/api/calibration/test-drop-tip", json={"target_index": 0})
        assert resp.status_code == 200
        mock_client.use_pipette.assert_called_once_with("left")
        mock_client.drop_tip.assert_called_once()

    def test_test_dispense_source_to_dest(self, tc, mock_client):
        """Dispense endpoint passes source and dest targets to calibration."""
        resp = tc.post("/api/calibration/test-dispense", json={
            "source_index": 1,
            "dest_index": 2,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["source"] == "source"
        assert data["dest"] == "dest"
        # Should have called aspirate (from source) then dispense (to dest) then blow_out
        mock_client.use_pipette.assert_called_once_with("right")
        mock_client.aspirate.assert_called_once()
        mock_client.dispense.assert_called_once()
        mock_client.blow_out.assert_called_once()

    def test_save_profile(self, tc, tmp_path):
        path = str(tmp_path / "saved_profile.json")
        resp = tc.post(
            "/api/calibration/save-profile",
            json={"path": path},
        )
        assert resp.status_code == 200
        assert resp.json()["path"] == path

    def test_list_profiles_with_profile(self, tc):
        resp = tc.get("/api/calibration/profiles")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["name"] == "test-profile"

    def test_calibration_page_shows_targets(self, tc):
        resp = tc.get("/calibration")
        assert resp.status_code == 200
        assert "source" in resp.text
        assert "dest" in resp.text

    def test_calibration_page_has_dispense_selectors(self, tc):
        """Calibration page has source/dest selectors for dispense test."""
        resp = tc.get("/calibration")
        assert resp.status_code == 200
        assert "dispense-source" in resp.text
        assert "dispense-dest-label" in resp.text


class TestHandlerCalibrationOffsets:
    def test_transfer_passes_profile_offsets_to_ops(self):
        client = MagicMock()
        client.get_labware_id.side_effect = lambda s: f"lw-{s}"
        ops = MagicMock()
        handlers = OT2StepHandlers(client=client, ops=ops)
        profile = CalibrationProfile(
            name="robot",
            targets=[
                build_target("tip", "10", "A1", "pick_up_tip", pipette_mount="left", offset=Offset(z=-3.0)),
                build_target("src", "7", "A1", "aspirate", pipette_mount="left", offset=Offset(x=1.0)),
                build_target("dst", "1", "A1", "dispense", pipette_mount="left", offset=Offset(y=2.0)),
                build_target("clean", "4", "A6", "aspirate", pipette_mount="left", offset=Offset(z=-1.0)),
            ],
        )
        handlers.set_calibration_profile(profile)
        handlers._active_mount = "left"

        step = RunStep(
            name="transfer",
            kind="transfer",
            params={
                "tiprack_slot": "10",
                "source_slot": "7",
                "dest_slot": "1",
                "cleaning_slot": "4",
                "tip_well": "A1",
                "source_well": "A1",
                "dest_well": "A1",
                "rinse_well": "A6",
                "volume": 50,
            },
        )

        handlers.handle_transfer(step)

        kwargs = ops.transfer.call_args.kwargs
        assert kwargs["tip_offset"] == (0.0, 0.0, -3.0)
        assert kwargs["source_offset"] == (1.0, 0.0, 0.0)
        assert kwargs["dest_offset"] == (0.0, 2.0, 0.0)
        assert kwargs["cleaning_offset"] == (0.0, 0.0, -1.0)


# ---------------------------------------------------------------------------
# Minimal TaskPlugin stub for testing generic task routes
# ---------------------------------------------------------------------------


class _StubPlugin:
    """Minimal TaskPlugin implementation for testing."""

    name = "stub_task"

    def load_config(self, path: str):
        return {"path": path, "target_color": [128, 0, 255]}

    def build_deck_config(self, config):
        return {"pipettes": {}, "labware": {}}

    def initial_state(self, config, mode: str):
        return {
            "mode": mode,
            "iteration": 0,
            "phase": "random",
            "history": [],
            "terminal": False,
        }

    def build_plan(self, config, state, mode: str):
        return {
            "mode": mode,
            "total_iterations": 3,
            "iterations": [
                {"iteration": 1, "phase": "random", "status": "planned"},
                {"iteration": 2, "phase": "random", "status": "planned"},
                {"iteration": 3, "phase": "bo", "status": "planned"},
            ],
        }

    def build_iteration_run(self, config, state, iteration, mode):
        return TaskRun(
            name=f"iter_{iteration}",
            steps=[RunStep(name="step1", kind="generic")],
        )

    def build_calibration_run(self, config, state):
        return TaskRun(
            name="calibration",
            steps=[RunStep(name="cal_step", kind="generic")],
        )

    def build_tip_check_run(self, config, state):
        return TaskRun(
            name="tip_check",
            steps=[RunStep(name="tip_step", kind="generic")],
        )

    def apply_run_result(self, config, state, run, mode):
        new = dict(state)
        new["iteration"] = state.get("iteration", 0) + 1
        new["history"] = list(state.get("history", [])) + [{"run_id": run.id}]
        return new

    def build_calibration_targets(self, config):
        return [
            {"name": "tip", "slot": "10", "well": "A1", "action": "pick_up_tip"},
            {"name": "src", "slot": "7", "well": "A1", "action": "aspirate"},
        ]

    def status_payload(self, config, state):
        return {
            "state": "idle" if not state.get("terminal") else "stopped",
            "iteration": state.get("iteration", 0),
            "mode": state.get("mode", "quick"),
        }

    def web_extension(self, config):
        return _StubWebExtension()


class _StubWebExtension:
    """Minimal TaskWebExtension for testing."""

    def extra_routes(self):
        return None

    def ui_payload(self, config, state):
        return {"task_name": "stub", "custom_data": 42}

    def extra_status(self, config, state):
        return {"extra_field": "from_extension"}


# ---------------------------------------------------------------------------
# Generic task plugin routes
# ---------------------------------------------------------------------------


class TestTaskPluginRoutes:
    """Test the generic /api/task/* routes backed by a TaskPlugin."""

    @pytest.fixture()
    def plugin(self):
        return _StubPlugin()

    @pytest.fixture()
    def task_app(self, tmp_path, plugin):
        store = JsonRunStore(base_dir=tmp_path)
        runner = TaskRunner(store=store, handlers={"generic": _ok_handler})
        wa = WebApp.__new__(WebApp)
        # Manually set required attributes (skip __init__ to avoid robot connection)
        wa.store = store
        wa.runner = runner
        wa.plugin = plugin
        wa.config_path = "/fake/config.yaml"
        wa._task_config = plugin.load_config("/fake/config.yaml")
        wa._task_state = None
        wa.client = None
        wa.ops = None
        wa.handlers = MagicMock()
        wa._sub_apps = []
        wa._nav_links = []
        wa.calibration_profile = None
        wa.calibration_session = None
        wa.deck = None
        wa.data_dir = tmp_path
        return wa

    @pytest.fixture()
    def tc(self, task_app):
        app = create_app(
            store=task_app.store,
            runner=task_app.runner,
            client=None,
            plugin=task_app.plugin,
            webapp=task_app,
        )
        return TestClient(app)

    def test_task_info(self, tc):
        resp = tc.get("/api/task/info")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "stub_task"
        assert data["has_config"] is True

    def test_task_plan(self, tc):
        resp = tc.get("/api/task/plan?mode=quick")
        assert resp.status_code == 200
        data = resp.json()
        assert data["mode"] == "quick"
        assert data["total_iterations"] == 3
        assert len(data["iterations"]) == 3

    def test_task_status(self, tc):
        resp = tc.get("/api/task/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["state"] == "idle"
        assert data["iteration"] == 0
        # extra_status from web extension should be merged
        assert data["extra_field"] == "from_extension"

    def test_task_status_merges_extension(self, tc):
        """extra_status from TaskWebExtension is merged into status."""
        resp = tc.get("/api/task/status")
        data = resp.json()
        assert "extra_field" in data
        assert data["extra_field"] == "from_extension"

    def test_task_calibration_targets(self, tc):
        resp = tc.get("/api/task/calibration-targets")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["targets"]) == 2
        assert data["targets"][0]["name"] == "tip"

    def test_task_start(self, tc):
        resp = tc.post("/api/task/start", json={"mode": "quick"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "started"
        assert "run_id" in data

    def test_task_start_terminal_returns_409(self, tc, task_app):
        task_app.task_state = {"terminal": True, "iteration": 5}
        resp = tc.post("/api/task/start")
        assert resp.status_code == 409

    def test_task_reset(self, tc, task_app):
        task_app.task_state = {"terminal": True, "iteration": 5}
        resp = tc.post("/api/task/reset", json={"mode": "full"})
        assert resp.status_code == 200
        assert task_app.task_state["terminal"] is False
        assert task_app.task_state["mode"] == "full"

    def test_task_stop_no_active_run(self, tc):
        resp = tc.post("/api/task/stop")
        assert resp.status_code == 409

    def test_task_pause_no_active_run(self, tc):
        resp = tc.post("/api/task/pause")
        assert resp.status_code == 409

    def test_task_resume_no_active_run(self, tc):
        resp = tc.post("/api/task/resume")
        assert resp.status_code == 409

    def test_task_calibrate(self, tc):
        resp = tc.post("/api/task/calibrate")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "started"
        assert "run_id" in data

    def test_task_tip_check(self, tc):
        resp = tc.post("/api/task/tip-check")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"

    def test_task_ui_payload(self, tc):
        resp = tc.get("/api/task/ui")
        assert resp.status_code == 200
        data = resp.json()
        assert data["task_name"] == "stub"
        assert data["custom_data"] == 42


class TestTaskLifecycle:
    """Integration tests for start -> pause -> resume -> completion flow.

    Verifies the bug fix: active_run_id is written into task state on
    start and cleared on completion/failure/stop.
    """

    @pytest.fixture()
    def plugin(self):
        return _StubPlugin()

    @pytest.fixture()
    def blocking_event(self):
        return threading.Event()

    @pytest.fixture()
    def task_app(self, tmp_path, plugin, blocking_event):
        """WebApp with a handler that blocks until blocking_event is set."""

        def _blocking_handler(step: RunStep, context: Any = None) -> dict:
            blocking_event.wait(timeout=10)
            return {"result": "ok"}

        store = JsonRunStore(base_dir=tmp_path)
        runner = TaskRunner(store=store, handlers={"generic": _blocking_handler})
        wa = WebApp.__new__(WebApp)
        wa.store = store
        wa.runner = runner
        wa.plugin = plugin
        wa.config_path = "/fake/config.yaml"
        wa._task_config = plugin.load_config("/fake/config.yaml")
        wa._task_state = None
        wa.client = None
        wa.ops = None
        wa.handlers = MagicMock()
        wa._sub_apps = []
        wa._nav_links = []
        wa.calibration_profile = None
        wa.calibration_session = None
        wa.deck = None
        wa.data_dir = tmp_path
        return wa

    @pytest.fixture()
    def tc(self, task_app):
        app = create_app(
            store=task_app.store,
            runner=task_app.runner,
            client=None,
            plugin=task_app.plugin,
            webapp=task_app,
        )
        return TestClient(app)

    def test_start_stores_active_run_id(self, tc, task_app, blocking_event):
        """start should write active_run_id into task state."""
        resp = tc.post("/api/task/start", json={"mode": "quick"})
        assert resp.status_code == 200
        run_id = resp.json()["run_id"]

        # active_run_id should be set in task state immediately
        assert task_app.task_state is not None
        assert task_app.task_state.get("active_run_id") == run_id

        # Let the run finish
        blocking_event.set()
        time.sleep(0.3)

    def test_pause_succeeds_after_start(self, tc, task_app, blocking_event):
        """pause should succeed when a run is active (the original bug)."""
        resp = tc.post("/api/task/start", json={"mode": "quick"})
        assert resp.status_code == 200

        # Pause while the blocking handler is still waiting
        resp = tc.post("/api/task/pause")
        assert resp.status_code == 200
        assert resp.json()["status"] == "pause_requested"

        # Let the run finish (it will pause at next step boundary)
        blocking_event.set()
        time.sleep(0.3)

    def test_active_run_id_cleared_after_completion(self, tc, task_app, blocking_event):
        """active_run_id should be cleared when the run completes."""
        resp = tc.post("/api/task/start", json={"mode": "quick"})
        assert resp.status_code == 200
        assert task_app.task_state.get("active_run_id") is not None

        # Let the run finish
        blocking_event.set()
        time.sleep(0.5)

        # active_run_id should be cleared
        assert task_app.task_state.get("active_run_id") is None

    def test_active_run_id_cleared_on_failure(self, tc, tmp_path, plugin):
        """active_run_id should be cleared if the run fails."""

        def _failing_handler(step: RunStep, context: Any = None) -> dict:
            raise RuntimeError("step failed")

        store = JsonRunStore(base_dir=tmp_path)
        runner = TaskRunner(store=store, handlers={"generic": _failing_handler})
        wa = WebApp.__new__(WebApp)
        wa.store = store
        wa.runner = runner
        wa.plugin = plugin
        wa.config_path = "/fake/config.yaml"
        wa._task_config = plugin.load_config("/fake/config.yaml")
        wa._task_state = None
        wa.client = None
        wa.ops = None
        wa.handlers = MagicMock()
        wa._sub_apps = []
        wa._nav_links = []
        wa.calibration_profile = None
        wa.calibration_session = None
        wa.deck = None
        wa.data_dir = tmp_path

        app = create_app(
            store=wa.store,
            runner=wa.runner,
            client=None,
            plugin=wa.plugin,
            webapp=wa,
        )
        fail_tc = TestClient(app)

        resp = fail_tc.post("/api/task/start", json={"mode": "quick"})
        assert resp.status_code == 200
        assert wa.task_state.get("active_run_id") is not None

        # Wait for background thread to finish (handler raises immediately)
        time.sleep(0.5)

        # active_run_id should be cleared on failure
        assert wa.task_state.get("active_run_id") is None

    def test_stop_clears_active_run_id(self, tc, task_app, blocking_event):
        """stop should clear active_run_id from state."""
        resp = tc.post("/api/task/start", json={"mode": "quick"})
        assert resp.status_code == 200
        assert task_app.task_state.get("active_run_id") is not None

        resp = tc.post("/api/task/stop")
        assert resp.status_code == 200
        assert resp.json()["status"] == "stop_requested"

        # active_run_id should be cleared immediately
        assert task_app.task_state.get("active_run_id") is None
        assert task_app.task_state.get("terminal") is True

        # Let the run finish
        blocking_event.set()
        time.sleep(0.3)

    def test_resume_uses_stored_active_run_id(self, tc, task_app, blocking_event):
        """resume should use the active_run_id stored in state."""
        # Build a run with two steps so the runner can pause between them
        import types

        original_build = task_app.plugin.build_iteration_run

        def _build_two_steps(self_plugin, config, state, iteration, mode):
            return TaskRun(
                name=f"iter_{iteration}",
                steps=[
                    RunStep(name="step1", kind="generic"),
                    RunStep(name="step2", kind="generic"),
                ],
            )

        task_app.plugin.build_iteration_run = types.MethodType(
            _build_two_steps, task_app.plugin,
        )

        resp = tc.post("/api/task/start", json={"mode": "quick"})
        assert resp.status_code == 200
        run_id = resp.json()["run_id"]

        # Request pause while step1 is blocked
        resp = tc.post("/api/task/pause")
        assert resp.status_code == 200

        # Let step1 finish — runner should pause before step2
        blocking_event.set()
        time.sleep(0.5)

        # If the run paused, active_run_id should still be set
        state = task_app.task_state
        run = task_app.store.load_run(run_id)
        if run.status == RunStatus.paused:
            assert state.get("active_run_id") == run_id

            # Now resume — it should use the stored active_run_id
            blocking_event.clear()  # block step2

            resp = tc.post("/api/task/resume")
            assert resp.status_code == 200

            # Let step2 finish
            blocking_event.set()
            time.sleep(0.5)

            # After completion, active_run_id should be cleared
            assert task_app.task_state.get("active_run_id") is None


class TestTaskPluginNotLoaded:
    """When no plugin is provided, task routes should not exist."""

    def test_task_info_404(self, client):
        resp = client.get("/api/task/info")
        assert resp.status_code in (404, 405)

    def test_task_status_404(self, client):
        resp = client.get("/api/task/status")
        assert resp.status_code in (404, 405)

    def test_task_plan_404(self, client):
        resp = client.get("/api/task/plan")
        assert resp.status_code in (404, 405)
