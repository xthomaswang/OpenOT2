"""Task-based protocol executor with integrated vision checks."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
from pydantic import BaseModel, field_validator

from openot2.client import OT2Client
from openot2.utils import LabwareMap, create_output_path, load_labware_from_config
from openot2.vision.base_types import PredictionResult, VisionModel
from openot2.vision.camera import Camera
from openot2.vision.analyzers import (
    CalibrationFn,
    LiquidAnalyzer,
    LiquidCheckResult,
    TipAnalyzer,
    TipCheckResult,
)
from openot2.protocol.recovery import ErrorRecovery, RecoveryContext

logger = logging.getLogger("openot2.protocol")


# ---------------------------------------------------------------------------
# Internal Pydantic models for config validation
# ---------------------------------------------------------------------------

class _VisionCheckConfig(BaseModel):
    type: str  # "tip" or "liquid"
    expected_tips: int = 8
    conf: float = 0.6


class _TaskConfig(BaseModel):
    type: str  # "pickup", "transfer", "drop"
    well: Optional[str] = None
    source_slot: Optional[str] = None
    source_well: Optional[str] = None
    dest_slot: Optional[str] = None
    dest_well: Optional[str] = None
    volume: Optional[float] = None
    origin: str = "bottom"
    offset: Optional[Tuple[float, float, float]] = None
    check: Optional[_VisionCheckConfig] = None
    on_fail: Optional[List[dict]] = None


class _ProtocolConfig(BaseModel):
    labware: dict
    settings: dict
    tasks: List[_TaskConfig]


# ---------------------------------------------------------------------------
# ProtocolExecutor
# ---------------------------------------------------------------------------

class ProtocolExecutor:
    """Execute task-based protocols with integrated vision checks.

    Args:
        client: :class:`OT2Client` for robot communication.
        model: :class:`VisionModel` for detection (required for vision checks).
        camera: :class:`Camera` for image capture (required for vision checks).
        calibration_fn: Volume-to-height calibration (required for liquid checks).
        tip_analyzer: Custom :class:`TipAnalyzer` (uses default if *None*).
        liquid_analyzer: Custom :class:`LiquidAnalyzer` (uses default if *None*).
        output_dir: Directory for saving captured images.
    """

    def __init__(
        self,
        client: OT2Client,
        model: Optional[VisionModel] = None,
        camera: Optional[Camera] = None,
        calibration_fn: Optional[CalibrationFn] = None,
        tip_analyzer: Optional[TipAnalyzer] = None,
        liquid_analyzer: Optional[LiquidAnalyzer] = None,
        output_dir: str = "output",
    ) -> None:
        self._client = client
        self._model = model
        self._camera = camera
        self._calibration_fn = calibration_fn
        self._tip_analyzer = tip_analyzer or TipAnalyzer()
        self._liquid_analyzer = liquid_analyzer or LiquidAnalyzer()
        self._output_dir = output_dir
        self._recovery = ErrorRecovery(client)

    def execute(self, config: Dict[str, Any]) -> bool:
        """Execute a complete task-based protocol.

        Args:
            config: Protocol configuration dict with keys
                ``labware``, ``settings``, ``tasks``.

        Returns:
            *True* if protocol completed successfully.
        """
        # Validate config
        validated = _ProtocolConfig.model_validate(config)

        logger.info("=" * 60)
        logger.info("  STARTING TASK-BASED PROTOCOL")
        logger.info("=" * 60)

        # Setup
        run_id = self._client.create_run()
        logger.info("Run ID: %s", run_id)

        labware_map = load_labware_from_config(self._client, validated.labware)

        settings = validated.settings
        img_labware_id = labware_map.imaging_labware_id
        img_well = settings.get("imaging_well", "A1")
        img_offset = settings.get("imaging_offset", (0, 0, 50))
        self._output_dir = settings.get("base_dir", self._output_dir)

        current_tip_well: Optional[str] = None

        # Execute tasks
        for i, task in enumerate(validated.tasks):
            logger.info("Task %d: %s", i + 1, task.type.upper())

            if task.type == "pickup":
                success = self._execute_pickup(
                    task, labware_map, run_id,
                    img_labware_id, img_well, img_offset,
                )
                if success:
                    current_tip_well = task.well
                else:
                    logger.error("Protocol stopping: pickup failure.")
                    break

            elif task.type == "transfer":
                success = self._execute_transfer(
                    task, labware_map, run_id, current_tip_well,
                    img_labware_id, img_well, img_offset,
                )
                if not success:
                    logger.error("Protocol stopping: transfer failure.")
                    break

            elif task.type == "drop":
                tiprack_id = labware_map.tiprack_id
                self._client.drop_tip(labware_id=tiprack_id, well=task.well or "A1")
                current_tip_well = None
                logger.info("Tips dropped.")

        logger.info("Protocol execution finished.")
        self._client.home()
        return True

    # ------------------------------------------------------------------
    # Task handlers
    # ------------------------------------------------------------------

    def _execute_pickup(
        self,
        task: _TaskConfig,
        lw: LabwareMap,
        run_id: str,
        img_labware_id: Optional[str],
        img_well: str,
        img_offset: tuple,
    ) -> bool:
        well = task.well or "A1"
        self._client.pick_up_tip(labware_id=lw.tiprack_id, well=well)

        # Vision check
        if task.check and task.check.type == "tip":
            prediction = self._capture_and_predict(
                run_id, f"pickup_{well}",
                img_labware_id, img_well, img_offset,
                task.check.conf,
            )
            if prediction is None:
                logger.warning("Vision check skipped (no model/camera).")
                return True

            result = self._tip_analyzer.analyze(prediction, task.check.expected_tips)
            if result.passed:
                logger.info("Tip pickup verified.")
                return True
            else:
                logger.error("Missing tips: %s", result.missing_positions)
                ctx = RecoveryContext(
                    run_id=run_id,
                    pipette_id=self._client.pipette_id,
                    tiprack_id=lw.tiprack_id,
                    well_name=well,
                    vision_result=result,
                )
                plan = task.on_fail or ErrorRecovery.default_pickup_recovery()
                self._recovery.execute(plan, ctx)
                return False

        return True

    def _execute_transfer(
        self,
        task: _TaskConfig,
        lw: LabwareMap,
        run_id: str,
        current_tip_well: Optional[str],
        img_labware_id: Optional[str],
        img_well: str,
        img_offset: tuple,
    ) -> bool:
        src_id = lw.sources.get(task.source_slot, "")
        dest_id = lw.dispenses.get(task.dest_slot, "") or lw.dispense_labware_id or ""

        # Aspirate
        self._client.aspirate(
            volume=task.volume,
            labware_id=src_id,
            well=task.source_well,
            origin=task.origin,
            offset=task.offset,
        )

        # Vision check
        if task.check and task.check.type == "liquid":
            self._client.move_to_well(img_labware_id, well=img_well, offset=img_offset)

            prediction = self._capture_and_predict(
                run_id, f"asp_{task.source_well}",
                img_labware_id, img_well, img_offset,
                task.check.conf,
            )
            if prediction is None:
                logger.warning("Vision check skipped (no model/camera).")
            elif self._calibration_fn is None:
                logger.warning("Vision check skipped (no calibration function).")
            else:
                result = self._liquid_analyzer.analyze(
                    prediction, task.volume, self._calibration_fn, task.check.expected_tips,
                )
                if result.passed:
                    logger.info("Liquid level verified.")
                else:
                    logger.error("Liquid check failed: %s", result.error_message)
                    ctx = RecoveryContext(
                        run_id=run_id,
                        pipette_id=self._client.pipette_id,
                        source_labware_id=src_id,
                        source_well=task.source_well,
                        dest_labware_id=dest_id,
                        dest_well=task.dest_well,
                        tiprack_id=lw.tiprack_id,
                        pick_well=current_tip_well,
                        volume=task.volume,
                        vision_result=result,
                    )
                    plan = task.on_fail or ErrorRecovery.default_liquid_recovery()
                    self._recovery.execute(plan, ctx)
                    return False

        # Dispense
        self._client.dispense(
            volume=task.volume,
            labware_id=dest_id,
            well=task.dest_well,
            origin="bottom",
        )
        self._client.blow_out(labware_id=dest_id, well=task.dest_well)
        return True

    # ------------------------------------------------------------------
    # Vision helpers
    # ------------------------------------------------------------------

    def _capture_and_predict(
        self,
        run_id: str,
        step_name: str,
        imaging_labware_id: Optional[str],
        imaging_well: str,
        imaging_offset: tuple,
        conf: float,
    ) -> Optional[PredictionResult]:
        """Move to imaging position, capture an image, and run inference."""
        if self._model is None or self._camera is None:
            return None

        # Move to imaging position
        if imaging_labware_id:
            self._client.move_to_well(
                imaging_labware_id, well=imaging_well, offset=imaging_offset,
            )

        # Capture
        frame = self._camera.capture()
        if frame is None:
            logger.error("Image capture failed.")
            return None

        # Save image
        image_path = create_output_path(self._output_dir, run_id, step_name)
        cv2.imwrite(image_path, frame)
        logger.info("Image saved: %s", image_path)

        # Predict
        prediction = self._model.predict(image_path, conf=conf)

        # Save annotated image
        if prediction.annotated_image is not None:
            ann_path = image_path.replace(".jpg", "_prediction.jpg")
            cv2.imwrite(ann_path, prediction.annotated_image)
            logger.info("Prediction image saved: %s", ann_path)

        return prediction
