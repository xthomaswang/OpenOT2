"""Camera abstraction with platform-aware USB camera support."""

from __future__ import annotations

import logging
import platform
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

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

    def preview(self, title: str = "Camera Preview") -> np.ndarray:
        """Capture a frame and display it in an OpenCV window.

        Press any key to close the window. Returns the captured BGR frame.
        """
        frame = self.capture()
        if frame is None:
            raise RuntimeError("Failed to capture frame for preview")
        cv2.imshow(title, frame)
        cv2.waitKey(0)
        cv2.destroyWindow(title)
        return frame

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.info("Camera released")

    @staticmethod
    def precheck_cameras(
        expected_id: int = 0,
        max_id: int = 10,
        width: int = 1920,
        height: int = 1080,
        warmup_frames: int = 5,
    ) -> List[Dict]:
        """Scan all cameras, show a preview from each, and print guidance.

        Displays a matplotlib figure with one frame per detected camera.
        The camera matching *expected_id* is highlighted so the user can
        verify it shows the correct overhead view.

        Args:
            expected_id: The device index currently set in the config.
            max_id: Maximum device index to probe.
            width: Capture width for preview frames.
            height: Capture height for preview frames.
            warmup_frames: Frames to discard before capturing the preview.

        Returns:
            List of camera info dicts (same format as :meth:`list_cameras`).
        """
        import matplotlib.pyplot as plt

        cameras = USBCamera.list_cameras(max_id=max_id)

        print(f"Found {len(cameras)} camera(s):")
        for cam in cameras:
            marker = " <-- config" if cam["id"] == expected_id else ""
            print(f"  Camera {cam['id']}: {cam['width']}x{cam['height']}{marker}")

        if not cameras:
            print("\nNo cameras found! Check USB connection.")
            return cameras

        fig, axes = plt.subplots(1, len(cameras), figsize=(6 * len(cameras), 4))
        if len(cameras) == 1:
            axes = [axes]

        for i, cam_info in enumerate(cameras):
            try:
                cam = USBCamera(
                    camera_id=cam_info["id"],
                    width=width,
                    height=height,
                    warmup_frames=warmup_frames,
                )
                with cam:
                    frame = cam.capture()
                if frame is not None:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    axes[i].imshow(frame_rgb)
                    title = f"Camera {cam_info['id']}"
                    if cam_info["id"] == expected_id:
                        title += " (selected)"
                    axes[i].set_title(title, fontsize=12, fontweight="bold")
                else:
                    axes[i].text(0.5, 0.5, "Capture failed",
                                 ha="center", va="center", fontsize=14)
                    axes[i].set_title(f"Camera {cam_info['id']} (FAILED)", fontsize=12)
            except Exception as e:
                axes[i].text(0.5, 0.5, f"Error:\n{e}",
                             ha="center", va="center", fontsize=10, wrap=True)
                axes[i].set_title(f"Camera {cam_info['id']} (ERROR)", fontsize=12)
            axes[i].axis("off")

        plt.suptitle("Verify the overhead camera — update device_id if needed",
                      fontsize=11)
        plt.tight_layout()
        plt.show()

        if not any(c["id"] == expected_id for c in cameras):
            print(f"\nWARNING: Config expects camera {expected_id} but it's not available.")
            print(f"Available IDs: {[c['id'] for c in cameras]}")
        else:
            print(f"\nConfig uses camera {expected_id}. "
                  "If that's not the plate view, update camera.device_id in experiment.yaml.")

        return cameras

    @staticmethod
    def list_cameras(max_id: int = 10) -> List[Dict]:
        """Scan for available USB cameras.

        Args:
            max_id: Maximum device index to probe (0 to max_id-1).

        Returns:
            List of dicts with keys: ``id``, ``width``, ``height``, ``backend``.
        """
        backend = USBCamera._detect_backend()
        found: List[Dict] = []
        for i in range(max_id):
            cap = cv2.VideoCapture(i, backend)
            if cap.isOpened():
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                info = {"id": i, "width": w, "height": h, "backend": backend}
                found.append(info)
                logger.info("Found camera %d: %dx%d", i, w, h)
                cap.release()
            else:
                cap.release()
        if not found:
            logger.warning("No cameras found (probed 0-%d)", max_id - 1)
        return found
