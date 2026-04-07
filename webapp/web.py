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
from openot2.control.models import RunSequence, RunStatus, RunStep, TaskRun
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


class CreateSequenceRequest(BaseModel):
    """Body for POST /api/sequences."""

    name: str
    metadata: dict[str, Any] = {}


class AddSequenceRunRequest(BaseModel):
    """Body for POST /api/sequences/{seq_id}/runs."""

    name: str
    steps: list[dict[str, Any]]


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
    nav_links: list[dict] | None = None,
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
    nav_links:
        Optional list of extra navigation links for the sidebar.
        Each dict should have ``title``, ``path``, and optionally ``icon``.
    """
    app = FastAPI(title="OpenOT2 Control")
    templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

    # Mutable app-level state
    app.state.store = store
    app.state.runner = runner
    app.state.client = client
    app.state.selected_camera = None  # {"device_id": int, "width": int, "height": int}
    app.state.active_runs: set[str] = set()
    app.state.calibration_session: CalibrationSession | None = None
    app.state.calibration_profile: CalibrationProfile | None = None

    # ------------------------------------------------------------------
    # HTML pages
    # ------------------------------------------------------------------

    @app.get("/", response_class=HTMLResponse)
    async def index_page(request: Request):
        runs = store.list_runs()
        return templates.TemplateResponse(request, "index.html", {"runs": runs, "extra_nav": nav_links})

    @app.get("/runs/{run_id}", response_class=HTMLResponse)
    async def run_detail_page(request: Request, run_id: str):
        try:
            run = store.load_run(run_id)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Run not found")
        events = store.list_events(run_id)
        return templates.TemplateResponse(
            request, "run_detail.html", {"run": run, "events": events, "extra_nav": nav_links},
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
                "has_client": app.state.client is not None,
                "extra_nav": nav_links,
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
    # JSON API — sequences
    # ------------------------------------------------------------------

    @app.post("/api/sequences")
    async def create_sequence(body: CreateSequenceRequest):
        """Create a new run sequence."""
        seq = RunSequence(name=body.name, metadata=body.metadata)
        store.create_sequence(seq)
        return seq.model_dump(mode="json")

    @app.get("/api/sequences")
    async def list_sequences():
        """Return all sequences as a JSON array."""
        seqs = store.list_sequences()
        return [s.model_dump(mode="json") for s in seqs]

    @app.get("/api/sequences/{seq_id}")
    async def get_sequence(seq_id: str):
        """Return a sequence with its full runs."""
        try:
            seq = store.load_sequence(seq_id)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Sequence not found")
        runs = []
        for run_id in seq.run_ids:
            try:
                runs.append(store.load_run(run_id).model_dump(mode="json"))
            except FileNotFoundError:
                pass
        data = seq.model_dump(mode="json")
        data["runs"] = runs
        return data

    @app.post("/api/sequences/{seq_id}/runs")
    async def add_run_to_sequence(seq_id: str, body: AddSequenceRunRequest):
        """Create a run and add it to a sequence."""
        try:
            seq = store.load_sequence(seq_id)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Sequence not found")
        steps = [RunStep(**s) for s in body.steps]
        run = TaskRun(name=body.name, steps=steps, sequence_id=seq_id)
        store.create_run(run)
        seq.run_ids.append(run.id)
        seq.updated_at = datetime.now(timezone.utc)
        store.save_sequence(seq)
        return run.model_dump(mode="json")

    # ------------------------------------------------------------------
    # JSON API — system status
    # ------------------------------------------------------------------

    @app.get("/api/status")
    async def get_status():
        """Return system status: robot connection, loaded labware, etc."""
        c = app.state.client
        result: dict[str, Any] = {"robot_connected": c is not None}
        if c:
            try:
                health = c.health(timeout=5)
                result["robot_name"] = health.get("name", "OT-2")
                result["robot_healthy"] = True
            except Exception:
                result["robot_healthy"] = False
            result["labware_slots"] = dict(c.labware_by_slot) if c.labware_by_slot else {}
            result["active_pipette"] = getattr(c, '_pipette_id', None)
        return result

    # ------------------------------------------------------------------
    # HTML — setup wizard
    # ------------------------------------------------------------------

    @app.get("/setup", response_class=HTMLResponse)
    async def setup_page(request: Request):
        return templates.TemplateResponse(request, "setup.html", {
            "has_client": app.state.client is not None,
            "extra_nav": nav_links,
        })

    # ------------------------------------------------------------------
    # HTML — prompt generator
    # ------------------------------------------------------------------

    @app.get("/generate", response_class=HTMLResponse)
    async def generate_page(request: Request):
        return templates.TemplateResponse(request, "generate.html", {"extra_nav": nav_links})

    @app.post("/api/task/validate")
    async def validate_task(body: dict):
        """Validate user-pasted LLM output: deck YAML, steps JSON, template HTML."""
        errors = []
        warnings = []

        # Validate deck YAML
        deck_yaml = body.get("deck_yaml", "").strip()
        if deck_yaml:
            try:
                import yaml
                deck_data = yaml.safe_load(deck_yaml)
                if not isinstance(deck_data, dict):
                    errors.append({"field": "deck_yaml", "message": "YAML must be a dict"})
                else:
                    if "pipettes" not in deck_data and "labware" not in deck_data:
                        warnings.append({"field": "deck_yaml", "message": "Missing 'pipettes' or 'labware' keys"})
            except Exception as exc:
                errors.append({"field": "deck_yaml", "message": f"Invalid YAML: {exc}"})
        else:
            errors.append({"field": "deck_yaml", "message": "Deck config is required"})

        # Validate steps JSON
        steps_json = body.get("steps_json", "").strip()
        if steps_json:
            try:
                import json
                steps = json.loads(steps_json)
                if not isinstance(steps, list):
                    errors.append({"field": "steps_json", "message": "Steps must be a JSON array"})
                else:
                    for i, step in enumerate(steps):
                        if not isinstance(step, dict):
                            errors.append({"field": "steps_json", "message": f"Step {i} must be a dict"})
                        elif "kind" not in step and "type" not in step:
                            errors.append({"field": "steps_json", "message": f"Step {i} missing 'kind' field"})
                        elif "name" not in step:
                            warnings.append({"field": "steps_json", "message": f"Step {i} missing 'name' field"})
            except Exception as exc:
                errors.append({"field": "steps_json", "message": f"Invalid JSON: {exc}"})
        else:
            errors.append({"field": "steps_json", "message": "Steps definition is required"})

        # Validate template HTML (basic check)
        template_html = body.get("template_html", "").strip()
        if template_html:
            if "{% extends" not in template_html and "<html" not in template_html.lower():
                warnings.append({"field": "template_html", "message": "Template should extend base.html or be valid HTML"})

        return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}

    # ------------------------------------------------------------------
    # JSON API — saved tasks
    # ------------------------------------------------------------------

    _TASKS_DIR = Path(".config/tasks")

    def _normalize_steps(raw_steps: list) -> list:
        """Normalize step dicts: support 'type' as alias for 'kind'."""
        steps = []
        for s in raw_steps:
            d = dict(s)
            if "type" in d and "kind" not in d:
                d["kind"] = d.pop("type")
            if "name" not in d:
                d["name"] = d.get("kind", "step")
            steps.append(d)
        return steps

    @app.post("/api/task/apply")
    async def apply_task(body: dict):
        """Save task to .config/tasks/{timestamp}_{name}/ and create a run."""
        import json as _json
        import re
        import yaml

        deck_yaml = body.get("deck_yaml", "").strip()
        steps_json = body.get("steps_json", "").strip()
        template_html = body.get("template_html", "").strip()
        run_name = body.get("run_name", "Custom Task").strip() or "Custom Task"

        # Parse
        try:
            yaml.safe_load(deck_yaml)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid deck YAML: {exc}")
        try:
            raw_steps = _json.loads(steps_json)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid steps JSON: {exc}")

        # Build task directory: {timestamp}_{sanitized_name}
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", run_name.lower()).strip("_")[:40]
        task_id = f"{ts}_{safe_name}"
        task_dir = _TASKS_DIR / task_id
        task_dir.mkdir(parents=True, exist_ok=True)

        # Save files
        (task_dir / "deck.yaml").write_text(deck_yaml)
        normalized = _normalize_steps(raw_steps)
        (task_dir / "steps.json").write_text(_json.dumps(normalized, indent=2))
        if template_html:
            (task_dir / "template.html").write_text(template_html)
        meta = {
            "name": run_name,
            "task_id": task_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "step_count": len(normalized),
            "has_template": bool(template_html),
        }
        (task_dir / "meta.json").write_text(_json.dumps(meta, indent=2))

        # Also save as active deck config
        active_dir = Path(".config")
        active_dir.mkdir(exist_ok=True)
        (active_dir / "deck.yaml").write_text(deck_yaml)

        # Save custom template if provided
        if template_html:
            (_TEMPLATES_DIR / "custom_task.html").write_text(template_html)

        # Create run
        steps = [RunStep(**s) for s in normalized]
        run = TaskRun(name=run_name, steps=steps)
        store.create_run(run)

        return {
            "status": "ok",
            "task_id": task_id,
            "run_id": run.id,
            "steps_count": len(steps),
        }

    @app.get("/api/tasks")
    async def list_tasks():
        """List all saved tasks."""
        import json as _json
        tasks = []
        if _TASKS_DIR.exists():
            for d in sorted(_TASKS_DIR.iterdir(), reverse=True):
                meta_path = d / "meta.json"
                if d.is_dir() and meta_path.exists():
                    try:
                        meta = _json.loads(meta_path.read_text())
                        meta["task_id"] = d.name
                        tasks.append(meta)
                    except Exception:
                        pass
        return tasks

    @app.get("/api/tasks/{task_id}")
    async def get_task(task_id: str):
        """Get full task details: meta, deck, steps."""
        import json as _json
        import yaml
        task_dir = _TASKS_DIR / task_id
        if not task_dir.exists():
            raise HTTPException(status_code=404, detail="Task not found")
        meta = _json.loads((task_dir / "meta.json").read_text())
        deck_yaml = (task_dir / "deck.yaml").read_text()
        deck = yaml.safe_load(deck_yaml)
        steps = _json.loads((task_dir / "steps.json").read_text())
        template = ""
        tpl_path = task_dir / "template.html"
        if tpl_path.exists():
            template = tpl_path.read_text()
        return {
            "meta": meta,
            "deck": deck,
            "deck_yaml": deck_yaml,
            "steps": steps,
            "template": template,
        }

    @app.post("/api/tasks/{task_id}/run")
    async def run_saved_task(task_id: str):
        """Create a new run from a saved task."""
        import json as _json
        task_dir = _TASKS_DIR / task_id
        if not task_dir.exists():
            raise HTTPException(status_code=404, detail="Task not found")
        meta = _json.loads((task_dir / "meta.json").read_text())
        raw_steps = _json.loads((task_dir / "steps.json").read_text())
        steps = [RunStep(**s) for s in raw_steps]
        run = TaskRun(name=meta.get("name", "Saved Task"), steps=steps)
        store.create_run(run)
        return {"status": "ok", "run_id": run.id}

    @app.delete("/api/tasks/{task_id}")
    async def delete_task(task_id: str):
        """Delete a saved task."""
        import shutil
        task_dir = _TASKS_DIR / task_id
        if not task_dir.exists():
            raise HTTPException(status_code=404, detail="Task not found")
        shutil.rmtree(task_dir)
        return {"status": "ok"}

    @app.get("/tasks", response_class=HTMLResponse)
    async def tasks_page(request: Request):
        return templates.TemplateResponse(request, "tasks.html", {"extra_nav": nav_links})

    @app.get("/api/tasks/{task_id}/calibration-targets")
    async def get_calibration_targets(task_id: str):
        """Extract calibration targets from a task in protocol order.

        Walks through steps sequentially, extracts every slot reference,
        and groups actions by slot.  Slots are ordered by their first
        appearance in the protocol.  Same-slot actions are merged into
        one calibration card.
        """
        import json as _json
        import yaml

        task_dir = _TASKS_DIR / task_id
        if not task_dir.exists():
            raise HTTPException(status_code=404, detail="Task not found")

        deck = yaml.safe_load((task_dir / "deck.yaml").read_text())
        steps = _json.loads((task_dir / "steps.json").read_text())
        labware = deck.get("labware", {})

        # Map: kind → which param keys hold slot references and what action they imply
        SLOT_PARAMS = {
            "pick_up_tip":  [("slot", "pick_up_tip")],
            "drop_tip":     [],  # trash, no labware calibration needed
            "aspirate":     [("slot", "aspirate")],
            "dispense":     [("slot", "dispense")],
            "move":         [("slot", "move")],
            "blow_out":     [],
            "home":         [],
            "use_pipette":  [],
            "transfer":     [
                ("tiprack_slot", "pick_up_tip"),
                ("source_slot", "aspirate"),
                ("dest_slot", "dispense"),
                ("cleaning_slot", "rinse"),
            ],
            "mix":          [
                ("tiprack_slot", "pick_up_tip"),
                ("plate_slot", "aspirate/dispense"),
                ("cleaning_slot", "rinse"),
            ],
        }

        # Walk steps in order, collect slots
        seen_slots: dict[str, dict] = {}  # slot -> {order, actions, labware_name}
        order_counter = 0

        for step_idx, step in enumerate(steps):
            kind = step.get("kind", step.get("type", ""))
            params = step.get("params", {})
            slot_refs = SLOT_PARAMS.get(kind, [])

            for param_key, action_type in slot_refs:
                slot = params.get(param_key)
                if not slot:
                    continue
                slot = str(slot)
                if slot not in seen_slots:
                    seen_slots[slot] = {
                        "slot": slot,
                        "labware": labware.get(slot, "unknown"),
                        "actions": set(),
                        "order": order_counter,
                        "first_step": step_idx,
                    }
                    order_counter += 1
                seen_slots[slot]["actions"].add(action_type)

        # Build result sorted by protocol order
        targets = sorted(seen_slots.values(), key=lambda x: x["order"])
        return [
            {
                "slot": t["slot"],
                "labware": t["labware"],
                "actions": sorted(t["actions"]),
                "calibration_well": "A1",
                "first_step": t["first_step"],
            }
            for t in targets
        ]

    # ------------------------------------------------------------------
    # HTML — hardware setup
    # ------------------------------------------------------------------

    @app.get("/hardware", response_class=HTMLResponse)
    async def hardware_page(request: Request):
        return templates.TemplateResponse(request, "hardware.html", {
            "has_client": app.state.client is not None,
            "extra_nav": nav_links,
        })

    # ------------------------------------------------------------------
    # JSON API — hardware precheck
    # ------------------------------------------------------------------

    @app.post("/api/hardware/check-robot")
    async def check_robot():
        """Check robot connectivity. Uses the connected client if available."""
        c = app.state.client
        if c is not None:
            try:
                health = c.health(timeout=5)
                return {
                    "reachable": True,
                    "name": health.get("name", ""),
                    "api_version": health.get("api_version", ""),
                }
            except Exception as exc:
                return {"reachable": False, "error": str(exc)}
        return {"reachable": False, "error": "No robot configured. Use Hardware page to connect."}

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
        if not ip and app.state.client is not None:
            ip = getattr(app.state.client, '_base_url', '').replace('http://', '').split(':')[0]
        if not ip:
            ip = "169.254.8.56"
        from openot2.precheck import check_robot_connection, probe_cameras
        robot = None
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

    @app.post("/api/hardware/connect")
    async def connect_robot(body: dict):
        """Connect to a robot by IP and remember the connection."""
        ip = body.get("ip", "").strip()
        if not ip:
            ip = "169.254.8.56"
        from openot2.client import OT2Client
        try:
            c = OT2Client(ip, timeout=30)
            health = c.health(timeout=10)
            app.state.client = c
            return {
                "connected": True,
                "name": health.get("name", ""),
                "api_version": health.get("api_version", ""),
                "ip": ip,
            }
        except Exception as exc:
            return {"connected": False, "error": str(exc)}

    @app.post("/api/hardware/select-camera")
    async def select_camera(body: dict):
        """Remember the selected camera for this session."""
        # Stop any existing live camera first
        _stop_live_camera()
        app.state.selected_camera = {
            "device_id": body.get("device_id", 0),
            "width": body.get("width", 1920),
            "height": body.get("height", 1080),
        }
        return {"ok": True, "selected": app.state.selected_camera}

    @app.get("/api/hardware/selected-camera")
    async def get_selected_camera():
        """Return the currently selected camera, if any."""
        return {"selected": app.state.selected_camera}

    # Live camera support — keep camera open between frames
    app.state._live_camera = None
    app.state._live_camera_id = None

    def _stop_live_camera():
        if app.state._live_camera is not None:
            try:
                app.state._live_camera.release()
            except Exception:
                pass
            app.state._live_camera = None
            app.state._live_camera_id = None

    @app.post("/api/hardware/live-frame")
    async def live_frame(body: dict):
        """Grab a frame from the kept-open camera. Much faster than test-camera."""
        import base64
        device_id = body.get("device_id", 0)
        try:
            import cv2
            from vision.camera import USBCamera
            # Open camera if not already open or if device changed
            if app.state._live_camera is None or app.state._live_camera_id != device_id:
                _stop_live_camera()
                cam = USBCamera(
                    camera_id=device_id,
                    width=body.get("width", 1920),
                    height=body.get("height", 1080),
                    warmup_frames=5,
                )
                # Do initial warmup capture
                cam.capture()
                app.state._live_camera = cam
                app.state._live_camera_id = device_id

            cap = app.state._live_camera._cap
            if cap is None or not cap.isOpened():
                _stop_live_camera()
                return {"ok": False, "error": "Camera lost"}

            ret, frame = cap.read()
            if not ret or frame is None:
                return {"ok": False, "error": "Failed to read frame"}

            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            b64 = base64.b64encode(buf.tobytes()).decode("ascii")
            return {"ok": True, "image": b64, "width": frame.shape[1], "height": frame.shape[0]}
        except Exception as exc:
            _stop_live_camera()
            return {"ok": False, "error": str(exc)}

    @app.post("/api/hardware/live-stop")
    async def live_stop():
        """Release the live camera."""
        _stop_live_camera()
        return {"ok": True}

    # ------------------------------------------------------------------
    # JSON API — calibration
    # ------------------------------------------------------------------

    def _require_client():
        if app.state.client is None:
            raise HTTPException(
                status_code=503,
                detail="No OT2Client configured — calibration actions require a connected robot",
            )
        return app.state.client

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
