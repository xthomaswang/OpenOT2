"""Built-in model adapters for YOLO (ultralytics) and SuperGradients."""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

from openot2.vision.base_types import PredictionResult, VisionModel

logger = logging.getLogger("openot2.vision.models")


class YOLOAdapter(VisionModel):
    """Adapter for `Ultralytics YOLO <https://docs.ultralytics.com/>`_ models.

    Args:
        model_path: Path to ``.pt`` weights file.
        num_classes: Number of classes the model detects.
        device: Inference device (``"cpu"``, ``"cuda"``, ``"mps"``).
                If *None*, auto-detected.

    Raises:
        ImportError: If ``ultralytics`` is not installed.
    """

    def __init__(
        self,
        model_path: str,
        num_classes: int = 2,
        device: Optional[str] = None,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "ultralytics is required for YOLOAdapter. "
                "Install with: pip install openot2[yolo]"
            ) from None

        self._model = YOLO(model_path)
        self._device = device
        self._num_classes = num_classes
        # class_names populated on first predict from model.names
        self._class_names_list: List[str] = []

    def predict(self, image_path: str, conf: float = 0.4) -> PredictionResult:
        kwargs = {"conf": conf, "verbose": False}
        if self._device:
            kwargs["device"] = self._device

        results = self._model(image_path, **kwargs)
        result = results[0]

        # Populate class names from model metadata
        if not self._class_names_list:
            self._class_names_list = [result.names[i] for i in sorted(result.names)]

        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return PredictionResult(
                labels=np.array([], dtype=int),
                bboxes_xyxy=np.empty((0, 4)),
                class_names=self._class_names_list,
                confidences=np.array([]),
                source_image_path=image_path,
            )

        return PredictionResult(
            labels=boxes.cls.cpu().numpy().astype(int),
            bboxes_xyxy=boxes.xyxy.cpu().numpy(),
            class_names=self._class_names_list,
            confidences=boxes.conf.cpu().numpy(),
            source_image_path=image_path,
            annotated_image=result.plot(),
        )

    @property
    def class_names(self) -> List[str]:
        return self._class_names_list


class SuperGradientsAdapter(VisionModel):
    """Adapter for `SuperGradients <https://docs.deci.ai/super-gradients/>`_ models.

    All access to the private ``_images_prediction_lst`` API is isolated here.

    Args:
        model_name: SG architecture name (e.g. ``"yolo_nas_l"``).
        num_classes: Number of classes.
        checkpoint_path: Path to ``.pth`` weights.

    Raises:
        ImportError: If ``super-gradients`` is not installed.
    """

    def __init__(
        self,
        model_name: str = "yolo_nas_l",
        num_classes: int = 2,
        checkpoint_path: Optional[str] = None,
    ) -> None:
        try:
            from super_gradients.training import models
        except ImportError:
            raise ImportError(
                "super-gradients is required for SuperGradientsAdapter. "
                "Install with: pip install openot2[supergradients]"
            ) from None

        self._model = models.get(
            model_name,
            num_classes=num_classes,
            checkpoint_path=checkpoint_path,
        )
        self._class_names_list: List[str] = []

    def predict(self, image_path: str, conf: float = 0.4) -> PredictionResult:
        predictions = self._model.predict(image_path, conf=conf)

        # Navigate SG internal structure
        if hasattr(predictions, "_images_prediction_lst"):
            pred_list = predictions._images_prediction_lst
        else:
            pred_list = [predictions]

        if not pred_list:
            return PredictionResult(
                labels=np.array([], dtype=int),
                bboxes_xyxy=np.empty((0, 4)),
                class_names=self._class_names_list,
                confidences=np.array([]),
                source_image_path=image_path,
            )

        image_pred = pred_list[0]
        prediction = image_pred.prediction

        # Populate class names
        if not self._class_names_list:
            self._class_names_list = list(image_pred.class_names)

        labels = prediction.labels.astype(int)
        bboxes = prediction.bboxes_xyxy
        confidences = (
            prediction.confidence
            if hasattr(prediction, "confidence")
            else np.ones(len(labels))
        )

        # Annotated image
        annotated = None
        if hasattr(image_pred, "draw"):
            try:
                annotated = image_pred.draw()
            except Exception:
                pass

        return PredictionResult(
            labels=labels,
            bboxes_xyxy=bboxes,
            class_names=self._class_names_list,
            confidences=confidences,
            source_image_path=image_path,
            annotated_image=annotated,
        )

    @property
    def class_names(self) -> List[str]:
        return self._class_names_list
