"""Minimal web UI and JSON API for the OpenOT2 task controller.

Provides:
- Server-rendered HTML pages (run list, run detail, calibration)
- JSON API endpoints for run management and calibration
- Polling-based progress updates (no websockets)

Requires the ``web`` optional extras::

    pip install openot2[web]
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from webapp.calibration import (
    CalibrationProfile,
    CalibrationSession,
    CalibrationTarget,
    load_profile,
    nudge_offset,
    preview_target,
    save_profile,
    test_aspirate,
    test_dispense,
)
from openot2.control.models import RunStatus, RunStep, TaskRun
from openot2.control.runner import TaskRunner
from openot2.control.store import JsonRunStore

# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------


class CreateRunRequest(BaseModel):
    """Body for POST /api/runs."""

    name: str
    steps: list[dict[str, Any]]


class NudgeRequest(BaseModel):
    """Body for POST /api/calibration/nudge."""

    target_index: int
    axis: str
    delta: float


class TargetIndexRequest(BaseModel):
    """Body for calibration endpoints that only need a target index."""

    target_index: int
    volume: float | None = None


class DispenseRequest(BaseModel):
    """Body for POST /api/calibration/test-dispense (source + dest)."""

    source_index: int
    dest_index: int
    volume: float | None = None


class SaveProfileRequest(BaseModel):
    """Body for POST /api/calibration/save-profile."""

    path: str


class LoadProfileRequest(BaseModel):
    """Body for POST /api/calibration/load-profile."""

    path: str


class CreateProfileRequest(BaseModel):
    """Body for POST /api/calibration/create-profile."""

    name: str
    targets: list[dict[str, Any]]


# ---------------------------------------------------------------------------
# Template directory (sibling of this file)
# ---------------------------------------------------------------------------

_TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"

# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(
    store: JsonRunStore,
    runner: TaskRunner,
    client: Any | None = None,
) -> FastAPI:
    """Create a :class:`FastAPI` application wired to the given store and runner.

    Parameters
    ----------
    store:
        JSON-file persistence back-end.
    runner:
        Task runner instance (with handlers already registered).
    client:
        Optional :class:`OT2Client` for calibration endpoints.  When
        ``None``, calibration preview / test endpoints return 503.
    """
    app = FastAPI(title="OpenOT2 Control")
    templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

    # Mutable app-level state
    app.state.store = store
    app.state.runner = runner
    app.state.client = client
    app.state.active_runs: set[str] = set()
    app.state.calibration_session: CalibrationSession | None = None
    app.state.calibration_profile: CalibrationProfile | None = None

    # ------------------------------------------------------------------
    # HTML pages
    # ------------------------------------------------------------------

    @app.get("/", response_class=HTMLResponse)
    async def index_page(request: Request):
        runs = store.list_runs()
        return templates.TemplateResponse(request, "index.html", {"runs": runs})

    @app.get("/runs/{run_id}", response_class=HTMLResponse)
    async def run_detail_page(request: Request, run_id: str):
        try:
            run = store.load_run(run_id)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Run not found")
        events = store.list_events(run_id)
        return templates.TemplateResponse(
            request, "run_detail.html", {"run": run, "events": events},
        )

    @app.get("/calibration", response_class=HTMLResponse)
    async def calibration_page(request: Request):
        profile = app.state.calibration_profile
        session = app.state.calibration_session
        return templates.TemplateResponse(
            request,
            "calibration.html",
            {
                "profile": profile,
                "session": session,
                "has_client": client is not None,
            },
        )

    # ------------------------------------------------------------------
    # JSON API — runs
    # ------------------------------------------------------------------

    @app.get("/api/runs")
    async def list_runs_api():
        """Return all runs as a JSON array."""
        runs = store.list_runs()
        return [r.model_dump(mode="json") for r in runs]

    @app.post("/api/runs")
    async def create_run(body: CreateRunRequest):
        steps = [RunStep(**s) for s in body.steps]
        run = TaskRun(name=body.name, steps=steps)
        store.create_run(run)
        return run.model_dump(mode="json")

    @app.get("/api/runs/{run_id}")
    async def get_run(run_id: str):
        try:
            run = store.load_run(run_id)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Run not found")
        return run.model_dump(mode="json")

    @app.get("/api/runs/{run_id}/events")
    async def get_events(run_id: str):
        events = store.list_events(run_id)
        return [e.model_dump(mode="json") for e in events]

    @app.post("/api/runs/{run_id}/start")
    async def start_run(run_id: str):
        try:
            run = store.load_run(run_id)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Run not found")

        if run.status not in (RunStatus.draft, RunStatus.ready):
            raise HTTPException(
                status_code=409,
                detail=f"Cannot start run with status '{run.status.value}' — expected draft or ready",
            )

        if run_id in app.state.active_runs:
            raise HTTPException(status_code=409, detail="Run is already active")

        def _bg():
            try:
                runner.run_until_pause_or_done(run_id)
            finally:
                app.state.active_runs.discard(run_id)

        app.state.active_runs.add(run_id)
        threading.Thread(target=_bg, daemon=True).start()
        return {"status": "started", "run_id": run_id}

    @app.post("/api/runs/{run_id}/pause")
    async def pause_run(run_id: str):
        try:
            run = store.load_run(run_id)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Run not found")

        if run.status != RunStatus.running:
            raise HTTPException(
                status_code=409,
                detail=f"Cannot pause run with status '{run.status.value}' — only running runs can be paused",
            )
        runner.request_pause(run_id)
        return {"status": "pause_requested", "run_id": run_id}

    @app.post("/api/runs/{run_id}/resume")
    async def resume_run(run_id: str):
        try:
            run = store.load_run(run_id)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Run not found")

        if run.status != RunStatus.paused:
            raise HTTPException(
                status_code=409,
                detail=f"Cannot resume run with status '{run.status.value}' — expected paused",
            )

        if run_id in app.state.active_runs:
            raise HTTPException(status_code=409, detail="Run is already active")

        def _bg():
            try:
                runner.resume(run_id)
            finally:
                app.state.active_runs.discard(run_id)

        app.state.active_runs.add(run_id)
        threading.Thread(target=_bg, daemon=True).start()
        return {"status": "resumed", "run_id": run_id}

    @app.post("/api/runs/{run_id}/abort")
    async def abort_run(run_id: str):
        """Abort a running or paused run."""
        try:
            run = store.load_run(run_id)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Run not found")
        if run.status not in (RunStatus.running, RunStatus.paused, RunStatus.pause_requested):
            raise HTTPException(
                status_code=409,
                detail=f"Cannot abort run with status '{run.status.value}'",
            )
        run.status = RunStatus.aborted
        run.updated_at = datetime.now(timezone.utc)
        store.save_run(run)
        return {"status": "aborted", "run_id": run_id}

    # ------------------------------------------------------------------
    # JSON API — system status
    # ------------------------------------------------------------------

    @app.get("/api/status")
    async def get_status():
        """Return system status: robot connection, loaded labware, etc."""
        result: dict[str, Any] = {"robot_connected": client is not None}
        if client:
            try:
                health = client.health(timeout=5)
                result["robot_name"] = health.get("name", "OT-2")
                result["robot_healthy"] = True
            except Exception:
                result["robot_healthy"] = False
            result["labware_slots"] = dict(client.labware_by_slot) if client.labware_by_slot else {}
            result["active_pipette"] = getattr(client, '_pipette_id', None)
        return result

    # ------------------------------------------------------------------
    # HTML — setup wizard
    # ------------------------------------------------------------------

    @app.get("/setup", response_class=HTMLResponse)
    async def setup_page(request: Request):
        return templates.TemplateResponse(request, "setup.html", {
            "has_client": client is not None,
        })

    # ------------------------------------------------------------------
    # HTML — hardware setup
    # ------------------------------------------------------------------

    @app.get("/hardware", response_class=HTMLResponse)
    async def hardware_page(request: Request):
        return templates.TemplateResponse(request, "hardware.html", {
            "has_client": client is not None,
        })

    # ------------------------------------------------------------------
    # JSON API — hardware precheck
    # ------------------------------------------------------------------

    @app.post("/api/hardware/check-robot")
    async def check_robot():
        """Check robot connectivity. Uses the connected client if available."""
        if client is not None:
            try:
                health = client.health(timeout=5)
                return {
                    "reachable": True,
                    "name": health.get("name", ""),
                    "api_version": health.get("api_version", ""),
                }
            except Exception as exc:
                return {"reachable": False, "error": str(exc)}
        return {"reachable": False, "error": "No robot configured. Start with --robot <IP>"}

    @app.post("/api/hardware/check-robot-ip")
    async def check_robot_ip(body: dict):
        """Check robot connectivity by IP (for manual testing)."""
        ip = body.get("ip", "").strip()
        if not ip:
            raise HTTPException(status_code=400, detail="IP address required")
        from openot2.precheck import check_robot_connection
        status = check_robot_connection(ip, timeout=5.0)
        return {
            "reachable": status.reachable,
            "name": status.name,
            "api_version": status.api_version,
            "error": status.error,
        }

    @app.post("/api/hardware/scan-cameras")
    async def scan_cameras():
        """Scan for USB cameras."""
        try:
            from openot2.precheck import probe_cameras
            cameras = probe_cameras(max_id=10)
            return {
                "cameras": [
                    {
                        "device_id": c.device_id,
                        "width": c.width,
                        "height": c.height,
                        "backend": c.backend,
                    }
                    for c in cameras
                ]
            }
        except Exception as exc:
            return {"cameras": [], "error": str(exc)}

    @app.post("/api/hardware/test-camera")
    async def test_camera(body: dict):
        """Capture a single test frame from a camera and return it as base64 JPEG."""
        import base64
        device_id = body.get("device_id", 0)
        try:
            from vision.camera import USBCamera
            import cv2
            cam = USBCamera(
                camera_id=device_id,
                width=body.get("width", 1920),
                height=body.get("height", 1080),
                warmup_frames=body.get("warmup_frames", 10),
            )
            with cam:
                frame = cam.capture()
            if frame is None:
                return {"ok": False, "error": "Capture returned None"}
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            b64 = base64.b64encode(buf.tobytes()).decode("ascii")
            return {"ok": True, "image": b64, "width": frame.shape[1], "height": frame.shape[0]}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    @app.post("/api/hardware/full-precheck")
    async def full_precheck(body: dict):
        """Run full precheck (robot + cameras)."""
        ip = body.get("ip", "").strip()
        if not ip and client is not None:
            ip = getattr(client, '_base_url', '').replace('http://', '').split(':')[0]
        from openot2.precheck import check_robot_connection, probe_cameras
        robot = {"reachable": False, "error": "No IP provided"} if not ip else None
        if ip:
            status = check_robot_connection(ip, timeout=5.0)
            robot = {
                "reachable": status.reachable,
                "name": status.name,
                "api_version": status.api_version,
                "error": status.error,
            }
        try:
            cameras = probe_cameras(max_id=10)
            cam_list = [{"device_id": c.device_id, "width": c.width, "height": c.height} for c in cameras]
        except Exception as exc:
            cam_list = []
        return {"robot": robot, "cameras": cam_list}

    # ------------------------------------------------------------------
    # JSON API — calibration
    # ------------------------------------------------------------------

    def _require_client():
        if client is None:
            raise HTTPException(
                status_code=503,
                detail="No OT2Client configured — calibration actions require a connected robot",
            )
        return client

    def _require_profile() -> CalibrationProfile:
        p = app.state.calibration_profile
        if p is None:
            raise HTTPException(status_code=400, detail="No calibration profile loaded")
        return p

    def _get_target(profile: CalibrationProfile, index: int) -> CalibrationTarget:
        if index < 0 or index >= len(profile.targets):
            raise HTTPException(status_code=400, detail="Target index out of range")
        return profile.targets[index]

    @app.post("/api/calibration/preview")
    async def calibration_preview(body: TargetIndexRequest):
        c = _require_client()
        profile = _require_profile()
        target = _get_target(profile, body.target_index)
        preview_target(c, target)
        return {"status": "ok", "target": target.name}

    @app.post("/api/calibration/nudge")
    async def calibration_nudge(body: NudgeRequest):
        profile = _require_profile()
        target = _get_target(profile, body.target_index)
        new_offset = nudge_offset(target.offset, body.axis, body.delta)
        profile.targets[body.target_index].offset = new_offset
        return {
            "status": "ok",
            "target": target.name,
            "offset": new_offset.model_dump(),
        }

    @app.post("/api/calibration/test-aspirate")
    async def calibration_test_aspirate(body: TargetIndexRequest):
        c = _require_client()
        profile = _require_profile()
        target = _get_target(profile, body.target_index)
        test_aspirate(c, target, volume=body.volume)
        return {"status": "ok", "target": target.name, "action": "aspirate"}

    @app.post("/api/calibration/test-dispense")
    async def calibration_test_dispense(body: DispenseRequest):
        c = _require_client()
        profile = _require_profile()
        source = _get_target(profile, body.source_index)
        dest = _get_target(profile, body.dest_index)
        test_dispense(c, dest, source_target=source, volume=body.volume)
        return {
            "status": "ok",
            "source": source.name,
            "dest": dest.name,
            "action": "dispense",
        }

    @app.post("/api/calibration/save-profile")
    async def calibration_save_profile(body: SaveProfileRequest):
        profile = _require_profile()
        save_profile(profile, Path(body.path))
        return {"status": "ok", "path": body.path}

    @app.get("/api/calibration/profiles")
    async def list_profiles():
        profile = app.state.calibration_profile
        if profile is None:
            return []
        return [profile.model_dump(mode="json")]

    @app.post("/api/calibration/load-profile")
    async def calibration_load_profile(body: LoadProfileRequest):
        path = Path(body.path)
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"Profile file not found: {body.path}")
        try:
            profile = load_profile(path)
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid profile file: {exc}",
            )
        app.state.calibration_profile = profile
        app.state.calibration_session = CalibrationSession(profile_id=profile.id, status="active")
        return {"status": "ok", "profile": profile.name, "targets": len(profile.targets)}

    @app.post("/api/calibration/create-profile")
    async def calibration_create_profile(body: CreateProfileRequest):
        try:
            targets = [CalibrationTarget(**t) for t in body.targets]
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid target definition: {exc}",
            )
        profile = CalibrationProfile(name=body.name, targets=targets)
        app.state.calibration_profile = profile
        app.state.calibration_session = CalibrationSession(profile_id=profile.id, status="active")
        return {"status": "ok", "profile": profile.name, "targets": len(profile.targets)}

    @app.get("/api/calibration/profile")
    async def get_current_profile():
        profile = app.state.calibration_profile
        if profile is None:
            raise HTTPException(status_code=404, detail="No profile loaded")
        return profile.model_dump(mode="json")

    return app
