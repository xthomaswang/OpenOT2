"""Reusable OT-2 step handlers for the web controller.

:class:`OT2StepHandlers` encapsulates all handler logic as instance
methods — no global state.  Call :meth:`register_all` to wire every
built-in handler into a :class:`TaskRunner`, or cherry-pick individual
handlers.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openot2.client import OT2Client
    from openot2.control.models import RunStep
    from openot2.control.runner import TaskRunner
    from openot2.operations import OT2Operations

logger = logging.getLogger("openot2.webapp.handlers")


class OT2StepHandlers:
    """Step handlers backed by an :class:`OT2Client` and :class:`OT2Operations`.

    When *client* is ``None`` (UI-only mode), handlers that require the
    robot return ``{"skipped": True}`` instead of raising.

    Parameters
    ----------
    client:
        Connected :class:`OT2Client`, or ``None`` for dry-run mode.
    ops:
        :class:`OT2Operations` wrapping *client*, or ``None``.
    """

    def __init__(
        self,
        client: OT2Client | None = None,
        ops: OT2Operations | None = None,
    ) -> None:
        self.client = client
        self.ops = ops
        self.progress_callback = None

    def _emit_progress(
        self,
        step: RunStep,
        context,
        *,
        action: str,
        detail: str,
        cycle: int | None = None,
        total_cycles: int | None = None,
    ) -> None:
        """Forward live sub-step progress to an external observer."""
        if not self.progress_callback:
            return
        run_id = context.get("run_id") if isinstance(context, dict) else None
        payload = {
            "run_id": run_id,
            "step_id": step.id,
            "step_name": step.name,
            "step_kind": step.kind,
            "action": action,
            "detail": detail,
        }
        if cycle is not None:
            payload["cycle"] = cycle
        if total_cycles is not None:
            payload["total_cycles"] = total_cycles
        self.progress_callback(payload)

    # ------------------------------------------------------------------
    # Bulk registration
    # ------------------------------------------------------------------

    #: Maps step ``kind`` strings to the method name that handles them.
    HANDLER_MAP: dict[str, str] = {
        "home": "handle_home",
        "aspirate": "handle_aspirate",
        "dispense": "handle_dispense",
        "move": "handle_move",
        "pick_up_tip": "handle_pick_up_tip",
        "drop_tip": "handle_drop_tip",
        "blow_out": "handle_blow_out",
        "use_pipette": "handle_use_pipette",
        "transfer": "handle_transfer",
        "mix": "handle_mix",
        "capture": "handle_capture",
        "predict": "handle_predict",
        "wait": "handle_wait",
    }

    def register_all(self, runner: TaskRunner) -> None:
        """Register every built-in handler with *runner*."""
        for kind, method_name in self.HANDLER_MAP.items():
            runner.register_handler(kind, getattr(self, method_name))

    # ------------------------------------------------------------------
    # Primitive handlers
    # ------------------------------------------------------------------

    def handle_home(self, step: RunStep, context=None) -> dict:
        self._emit_progress(step, context, action="home", detail="homing robot")
        if self.client:
            self.client.home()
        return {"action": "home"}

    def handle_aspirate(self, step: RunStep, context=None) -> dict:
        if not self.client:
            return {"skipped": True, "reason": "no client"}
        p = step.params
        labware_id = self.client.get_labware_id(p["slot"])
        self.client.aspirate(
            volume=p["volume"],
            labware_id=labware_id,
            well=p.get("well", "A1"),
        )
        return {"aspirated": p["volume"]}

    def handle_dispense(self, step: RunStep, context=None) -> dict:
        if not self.client:
            return {"skipped": True, "reason": "no client"}
        p = step.params
        labware_id = self.client.get_labware_id(p["slot"])
        self.client.dispense(
            volume=p["volume"],
            labware_id=labware_id,
            well=p.get("well", "A1"),
        )
        return {"dispensed": p["volume"]}

    def handle_move(self, step: RunStep, context=None) -> dict:
        if not self.client:
            return {"skipped": True, "reason": "no client"}
        p = step.params
        labware_id = self.client.get_labware_id(p["slot"])
        offset = tuple(p["offset"]) if "offset" in p else None
        self.client.move_to_well(
            labware_id=labware_id,
            well=p.get("well", "A1"),
            offset=offset,
        )
        return {"moved_to": p.get("well", "A1")}

    def handle_pick_up_tip(self, step: RunStep, context=None) -> dict:
        if not self.client:
            return {"skipped": True, "reason": "no client"}
        p = step.params
        labware_id = self.client.get_labware_id(p["slot"])
        self.client.pick_up_tip(labware_id=labware_id, well=p.get("well", "A1"))
        return {"tip": "picked_up"}

    def handle_drop_tip(self, step: RunStep, context=None) -> dict:
        if not self.client:
            return {"skipped": True, "reason": "no client"}
        self.client.drop_tip_in_trash()
        return {"tip": "dropped"}

    def handle_blow_out(self, step: RunStep, context=None) -> dict:
        if not self.client:
            return {"skipped": True, "reason": "no client"}
        self.client.blow_out()
        return {"action": "blow_out"}

    def handle_use_pipette(self, step: RunStep, context=None) -> dict:
        if not self.client:
            return {"skipped": True, "reason": "no client"}
        mount = step.params["mount"]
        self._emit_progress(step, context, action="use_pipette", detail=f"mount={mount}")
        self.client.use_pipette(mount)
        return {"active_pipette": mount}

    # ------------------------------------------------------------------
    # Composite handlers (require OT2Operations)
    # ------------------------------------------------------------------

    def handle_transfer(self, step: RunStep, context=None) -> dict:
        if not self.ops:
            return {"skipped": True, "reason": "no client"}
        p = step.params
        client = self.client
        self.ops.transfer(
            tiprack_id=client.get_labware_id(p["tiprack_slot"]),
            source_id=client.get_labware_id(p["source_slot"]),
            dest_id=client.get_labware_id(p["dest_slot"]),
            tip_well=p.get("tip_well", "A1"),
            source_well=p.get("source_well", "A1"),
            dest_well=p.get("dest_well", "A1"),
            volume=p["volume"],
            cleaning_id=(
                client.get_labware_id(p["cleaning_slot"])
                if "cleaning_slot" in p
                else None
            ),
            rinse_col=p.get("rinse_well"),
            rinse_cycles=p.get("rinse_cycles"),
            rinse_volume=p.get("rinse_volume"),
            progress_callback=lambda payload: self._emit_progress(
                step,
                context,
                action=payload["action"],
                detail=payload["detail"],
                cycle=payload.get("cycle"),
                total_cycles=payload.get("total_cycles"),
            ),
        )
        return {"transferred": p["volume"], "to": p.get("dest_well")}

    def handle_mix(self, step: RunStep, context=None) -> dict:
        if not self.ops:
            return {"skipped": True, "reason": "no client"}
        p = step.params
        client = self.client
        self.ops.mix(
            tiprack_id=client.get_labware_id(p["tiprack_slot"]),
            labware_id=client.get_labware_id(p["plate_slot"]),
            tip_well=p.get("tip_well", "A4"),
            mix_well=p.get("mix_well", "A1"),
            cycles=p.get("cycles", 3),
            volume=p.get("volume", 150),
            cleaning_id=(
                client.get_labware_id(p["cleaning_slot"])
                if "cleaning_slot" in p
                else None
            ),
            rinse_col=p.get("rinse_well"),
            rinse_cycles=p.get("rinse_cycles"),
            rinse_volume=p.get("rinse_volume"),
            progress_callback=lambda payload: self._emit_progress(
                step,
                context,
                action=payload["action"],
                detail=payload["detail"],
                cycle=payload.get("cycle"),
                total_cycles=payload.get("total_cycles"),
            ),
        )
        return {"mixed": p.get("mix_well"), "cycles": p.get("cycles", 3)}

    # ------------------------------------------------------------------
    # Vision / utility handlers
    # ------------------------------------------------------------------

    def handle_capture(self, step: RunStep, context=None) -> dict:
        """Capture an image from a USB camera.

        Params:
            camera_id (int, optional): Camera device index. Default 0.
            width (int, optional): Frame width. Default 1920.
            height (int, optional): Frame height. Default 1080.
            warmup_frames (int, optional): Frames to discard. Default 10.
            save_dir (str, optional): Directory to save image. Default "captures".
            label (str, optional): Label for the filename.

        Returns:
            {"image_path": str, "width": int, "height": int}
        """
        import time
        from pathlib import Path

        p = step.params
        camera_id = p.get("camera_id", 0)
        width = p.get("width", 1920)
        height = p.get("height", 1080)
        warmup = p.get("warmup_frames", 10)
        save_dir = Path(p.get("save_dir", "captures"))
        label = p.get("label", "capture")

        try:
            self._emit_progress(
                step, context, action="capture", detail=f"camera {camera_id}"
            )
            from vision.camera import USBCamera
            import cv2

            cam = USBCamera(
                camera_id=camera_id,
                width=width,
                height=height,
                warmup_frames=warmup,
            )
            with cam:
                frame = cam.capture()

            if frame is None:
                return {"ok": False, "error": "Capture returned None"}

            save_dir.mkdir(parents=True, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            filename = f"{label}_{ts}.jpg"
            path = save_dir / filename
            cv2.imwrite(str(path), frame)

            logger.info("Captured image: %s (%dx%d)", path, frame.shape[1], frame.shape[0])
            return {
                "ok": True,
                "image_path": str(path),
                "width": frame.shape[1],
                "height": frame.shape[0],
            }
        except Exception as exc:
            logger.error("Capture failed: %s", exc)
            return {"ok": False, "error": str(exc)}

    def handle_predict(self, step: RunStep, context=None) -> dict:
        """Run a vision model prediction on an image.

        Params:
            image_path (str, optional): Path to image. If omitted, captures
                a new frame using camera params.
            model_path (str, optional): Path to model weights.
            model_type (str, optional): "yolo" or "supergradients". Default "yolo".
            num_classes (int, optional): Number of classes. Default 2.
            conf (float, optional): Confidence threshold. Default 0.5.
            expected_class (str, optional): Class name to check for.
            min_detections (int, optional): Minimum detections to pass. Default 1.
            camera_id (int, optional): Camera index if capturing. Default 0.

        Returns:
            {"result": bool, "detections": int, "labels": [...], "confidences": [...]}

        The ``result`` boolean is the key output — it drives conditional
        branching in the GraphRunner (on_true / on_false).
        """
        p = step.params

        # Get or capture image
        image_path = p.get("image_path")
        if not image_path:
            cap_result = self.handle_capture(step, context)
            if not cap_result.get("ok"):
                return {"result": False, "error": cap_result.get("error", "capture failed")}
            image_path = cap_result["image_path"]

        model_type = p.get("model_type", "yolo")
        model_path = p.get("model_path")
        conf = p.get("conf", 0.5)
        expected_class = p.get("expected_class")
        min_detections = p.get("min_detections", 1)

        if not model_path:
            logger.warning("predict: no model_path, returning result=True (passthrough)")
            return {"result": True, "detections": 0, "reason": "no model configured"}

        try:
            if model_type == "yolo":
                from vision import YOLOAdapter
                model = YOLOAdapter(model_path=model_path, num_classes=p.get("num_classes", 2))
            else:
                from vision import SuperGradientsAdapter
                model = SuperGradientsAdapter(model_path=model_path, num_classes=p.get("num_classes", 2))

            prediction = model.predict(image_path, conf=conf)

            if expected_class:
                filtered = prediction.filter_by_class(expected_class)
                count = len(filtered.labels)
            else:
                count = len(prediction.labels)

            passed = count >= min_detections
            logger.info(
                "predict: %d detections (need %d), result=%s",
                count, min_detections, passed,
            )
            return {
                "result": passed,
                "detections": count,
                "labels": prediction.labels[:10],
                "confidences": [round(c, 3) for c in prediction.confidences[:10]],
            }
        except Exception as exc:
            logger.error("Predict failed: %s", exc)
            return {"result": False, "error": str(exc)}

    def handle_wait(self, step: RunStep, context=None) -> dict:
        """Wait for a specified number of seconds.

        Params:
            seconds (float, REQUIRED): Duration to wait.

        Returns:
            {"waited": float}
        """
        import time

        seconds = step.params.get("seconds", 1)
        self._emit_progress(step, context, action="wait", detail=f"{seconds:.1f}s")
        logger.info("Waiting %.1f seconds...", seconds)
        time.sleep(seconds)
        return {"waited": seconds}
