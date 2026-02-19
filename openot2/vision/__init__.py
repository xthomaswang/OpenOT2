"""Vision subsystem: model adapters, camera, and analyzers."""

from openot2.vision.base_types import PredictionResult, VisionModel
from openot2.vision.camera import Camera, USBCamera
from openot2.vision.analyzers import (
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
    "TipAnalyzer",
    "TipCheckResult",
    "LiquidAnalyzer",
    "LiquidCheckResult",
    "build_calibration_from_csv",
]


# Lazy adapter imports to avoid forcing ML framework installation
def YOLOAdapter(*args, **kwargs):  # noqa: N802
    """Lazy-loaded :class:`YOLOAdapter` (requires ``pip install openot2[yolo]``)."""
    from openot2.vision.models import YOLOAdapter as _Cls
    return _Cls(*args, **kwargs)


def SuperGradientsAdapter(*args, **kwargs):  # noqa: N802
    """Lazy-loaded :class:`SuperGradientsAdapter` (requires ``pip install openot2[supergradients]``)."""
    from openot2.vision.models import SuperGradientsAdapter as _Cls
    return _Cls(*args, **kwargs)
