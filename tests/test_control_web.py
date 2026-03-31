"""Tests for the OpenOT2 web control app — fully hardware-free."""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from webapp.calibration import (
    CalibrationProfile,
    Offset,
    build_target,
    save_profile,
)
from openot2.control.models import RunStatus, RunStep, TaskRun
from openot2.control.runner import TaskRunner
from openot2.control.store import JsonRunStore
from webapp.web import create_app


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
                {"name": "source", "labware_slot": "3", "well": "A1", "action": "aspirate", "volume": 50},
                {"name": "dest", "labware_slot": "5", "well": "B2", "action": "dispense", "volume": 50},
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
        mock_client.move_to_well.assert_called_once()

    def test_test_aspirate_calls_client(self, tc, mock_client):
        resp = tc.post("/api/calibration/test-aspirate", json={"target_index": 0})
        assert resp.status_code == 200
        mock_client.aspirate.assert_called_once()

    def test_test_dispense_source_to_dest(self, tc, mock_client):
        """Dispense endpoint passes source and dest targets to calibration."""
        resp = tc.post("/api/calibration/test-dispense", json={
            "source_index": 0,
            "dest_index": 1,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["source"] == "source"
        assert data["dest"] == "dest"
        # Should have called aspirate (from source) then dispense (to dest) then blow_out
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
        assert "dispense-dest" in resp.text
