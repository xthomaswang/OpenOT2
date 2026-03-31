"""Abstract base class for vision models and standardized prediction result."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class PredictionResult:
    """Framework-agnostic prediction output.

    All adapters must convert their native predictions into this format.

    Attributes:
        labels: Integer class indices per detection, shape ``(N,)``.
        bboxes_xyxy: Bounding boxes ``[x1, y1, x2, y2]``, shape ``(N, 4)``.
        class_names: Mapping from class index to human name.
        confidences: Confidence scores, shape ``(N,)``.
        source_image_path: Path to the analysed image.
        annotated_image: Optional annotated image array (BGR).
    """

    labels: np.ndarray
    bboxes_xyxy: np.ndarray
    class_names: List[str]
    confidences: np.ndarray
    source_image_path: Optional[str] = None
    annotated_image: Optional[np.ndarray] = None

    @property
    def num_detections(self) -> int:
        return len(self.labels)

    def filter_by_class(self, class_name: str) -> "PredictionResult":
        """Return a new result containing only detections of *class_name*."""
        if class_name not in self.class_names:
            return PredictionResult(
                labels=np.array([], dtype=int),
                bboxes_xyxy=np.empty((0, 4)),
                class_names=self.class_names,
                confidences=np.array([]),
                source_image_path=self.source_image_path,
            )
        idx = self.class_names.index(class_name)
        mask = self.labels == idx
        return PredictionResult(
            labels=self.labels[mask],
            bboxes_xyxy=self.bboxes_xyxy[mask],
            class_names=self.class_names,
            confidences=self.confidences[mask],
            source_image_path=self.source_image_path,
        )

    def sort_by_x(self) -> "PredictionResult":
        """Return a new result sorted left-to-right by bbox ``x_min``."""
        if self.num_detections == 0:
            return self
        order = np.argsort(self.bboxes_xyxy[:, 0])
        return PredictionResult(
            labels=self.labels[order],
            bboxes_xyxy=self.bboxes_xyxy[order],
            class_names=self.class_names,
            confidences=self.confidences[order],
            source_image_path=self.source_image_path,
            annotated_image=self.annotated_image,
        )


class VisionModel(ABC):
    """Abstract base class for object detection models.

    Implement :meth:`predict` to integrate any detection framework.
    """

    @abstractmethod
    def predict(self, image_path: str, conf: float = 0.4) -> PredictionResult:
        """Run inference on an image file.

        Args:
            image_path: Path to the image.
            conf: Confidence threshold.

        Returns:
            Standardized :class:`PredictionResult`.
        """
        ...

    @property
    @abstractmethod
    def class_names(self) -> List[str]:
        """Class names this model detects."""
        ...
