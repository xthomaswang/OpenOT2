"""Vision subsystem: model adapters, camera, and analyzers."""

from vision.base_types import PredictionResult, VisionModel
from vision.camera import Camera, USBCamera
from vision.analyzers import (
    TipAnalyzer,
    TipCheckResult,
    LiquidAnalyzer,
    LiquidCheckResult,
    build_calibration_from_csv,
)

__all__ = [
    "PredictionResult",
    "VisionModel",
    "Camera",
    "USBCamera",
    "list_cameras",
    "precheck_cameras",
    "TipAnalyzer",
    "TipCheckResult",
    "LiquidAnalyzer",
    "LiquidCheckResult",
    "build_calibration_from_csv",
]


def list_cameras(max_id: int = 10):
    """Shortcut for :meth:`USBCamera.list_cameras`."""
    return USBCamera.list_cameras(max_id=max_id)


def precheck_cameras(expected_id: int = 0, **kwargs):
    """Shortcut for :meth:`USBCamera.precheck_cameras`."""
    return USBCamera.precheck_cameras(expected_id=expected_id, **kwargs)


# Lazy adapter imports to avoid forcing ML framework installation
def YOLOAdapter(*args, **kwargs):  # noqa: N802
    """Lazy-loaded :class:`YOLOAdapter` (requires ``pip install openot2[yolo]``)."""
    from vision.models import YOLOAdapter as _Cls
    return _Cls(*args, **kwargs)


def SuperGradientsAdapter(*args, **kwargs):  # noqa: N802
    """Lazy-loaded :class:`SuperGradientsAdapter` (requires ``pip install openot2[supergradients]``)."""
    from vision.models import SuperGradientsAdapter as _Cls
    return _Cls(*args, **kwargs)
