"""Camera abstraction with platform-aware USB camera support."""

from __future__ import annotations

import logging
import platform
from abc import ABC, abstractmethod
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger("openot2.vision.camera")


class Camera(ABC):
    """Abstract camera interface."""

    @abstractmethod
    def capture(self) -> Optional[np.ndarray]:
        """Capture a single frame. Returns BGR array or *None*."""
        ...

    @abstractmethod
    def release(self) -> None:
        """Release camera resources."""
        ...

    def __enter__(self) -> "Camera":
        return self

    def __exit__(self, *exc) -> None:
        self.release()


class USBCamera(Camera):
    """USB camera with platform auto-detection and exposure stabilization.

    Args:
        camera_id: OpenCV camera device index.
        width: Frame width.
        height: Frame height.
        warmup_frames: Frames to discard for exposure stabilization.
        backend: OpenCV backend override. If *None*, auto-detects per OS.
    """

    def __init__(
        self,
        camera_id: int = 0,
        width: int = 640,
        height: int = 480,
        warmup_frames: int = 10,
        backend: Optional[int] = None,
    ) -> None:
        self._warmup_frames = warmup_frames
        resolved_backend = backend if backend is not None else self._detect_backend()

        self._cap = cv2.VideoCapture(camera_id, resolved_backend)
        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open camera {camera_id}")

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        logger.info("Camera %d opened (%dx%d, backend=%s)", camera_id, width, height,
                     resolved_backend)

    @staticmethod
    def _detect_backend() -> int:
        """Select OpenCV backend based on OS."""
        system = platform.system()
        if system == "Darwin":
            return cv2.CAP_AVFOUNDATION
        elif system == "Linux":
            return cv2.CAP_V4L2
        elif system == "Windows":
            return cv2.CAP_DSHOW
        return cv2.CAP_ANY

    def capture(self) -> Optional[np.ndarray]:
        """Capture a single stabilized frame."""
        if self._cap is None or not self._cap.isOpened():
            logger.error("Camera not available")
            return None

        # Discard warmup frames for exposure stabilization
        for _ in range(self._warmup_frames):
            self._cap.read()

        ret, frame = self._cap.read()
        if not ret:
            logger.error("Failed to capture frame")
            return None

        return frame

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.info("Camera released")
