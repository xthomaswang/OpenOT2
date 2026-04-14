"""Tip and liquid level analyzers working on standardized PredictionResult."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

import numpy as np

from openot2.vision.base_types import PredictionResult

logger = logging.getLogger("openot2.vision.analyzers")

CalibrationFn = Callable[[float], float]
"""Type alias: volume (uL) → expected height (percent)."""


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TipCheckResult:
    """Result of a tip presence check."""

    passed: bool
    tip_presence: List[int]
    missing_positions: List[int]
    tip_count: int
    expected_tips: int


@dataclass
class LiquidCheckResult:
    """Result of a liquid level check."""

    passed: bool
    detected_levels: List[float]
    channel_pass_status: List[bool]
    expected_height_percent: float
    expected_volume: float
    tip_count: int
    liquid_count: int
    error_message: Optional[str] = None


# ---------------------------------------------------------------------------
# Analyzers
# ---------------------------------------------------------------------------

class TipAnalyzer:
    """Analyze tip presence from a :class:`PredictionResult`.

    Args:
        tip_class_name: Class name for tips in the model output.
    """

    def __init__(self, tip_class_name: str = "Tip") -> None:
        self._tip_class = tip_class_name

    def analyze(
        self,
        prediction: PredictionResult,
        expected_tips: int = 8,
    ) -> TipCheckResult:
        """Check if all expected tips are present.

        Algorithm:
            1. Filter prediction for tip class, sort left-to-right.
            2. Estimate horizontal spacing from outermost tips.
            3. Check each expected position for a nearby detection.
        """
        tips = prediction.filter_by_class(self._tip_class).sort_by_x()
        tip_count = tips.num_detections

        # No tips at all
        if tip_count == 0:
            return TipCheckResult(
                passed=False,
                tip_presence=[0] * expected_tips,
                missing_positions=list(range(1, expected_tips + 1)),
                tip_count=0,
                expected_tips=expected_tips,
            )

        # Exact match
        if tip_count == expected_tips:
            return TipCheckResult(
                passed=True,
                tip_presence=[1] * expected_tips,
                missing_positions=[],
                tip_count=tip_count,
                expected_tips=expected_tips,
            )

        # Infer missing tips via spacing
        centers_x = (tips.bboxes_xyxy[:, 0] + tips.bboxes_xyxy[:, 2]) / 2.0
        spacing = (centers_x[-1] - centers_x[0]) / max(expected_tips - 1, 1)

        presence = [1] * expected_tips
        missing: List[int] = []

        for i in range(expected_tips):
            expected_x = centers_x[0] + i * spacing
            found = any(abs(cx - expected_x) <= spacing / 2 for cx in centers_x)
            if not found:
                presence[i] = 0
                missing.append(i + 1)

        return TipCheckResult(
            passed=len(missing) == 0,
            tip_presence=presence,
            missing_positions=missing,
            tip_count=tip_count,
            expected_tips=expected_tips,
        )


class LiquidAnalyzer:
    """Analyze liquid levels from a :class:`PredictionResult`.

    Args:
        tip_class_name: Class name for tips.
        liquid_class_name: Class name for liquid.
        tolerance_percent: Allowed deviation from expected height.
    """

    def __init__(
        self,
        tip_class_name: str = "Tip",
        liquid_class_name: str = "Liquid",
        tolerance_percent: float = 5.0,
    ) -> None:
        self._tip_class = tip_class_name
        self._liquid_class = liquid_class_name
        self._tolerance = tolerance_percent

    def analyze(
        self,
        prediction: PredictionResult,
        expected_volume: float,
        calibration_fn: CalibrationFn,
        expected_tips: int = 8,
    ) -> LiquidCheckResult:
        """Check liquid levels against expected volume.

        Algorithm:
            1. Filter and sort tips & liquids left-to-right.
            2. Match each liquid to its nearest tip by x-center.
            3. Compute ``liquid_height / tip_height * 100`` per channel.
            4. Compare to ``calibration_fn(expected_volume)`` within tolerance.
        """
        tips = prediction.filter_by_class(self._tip_class).sort_by_x()
        liquids = prediction.filter_by_class(self._liquid_class).sort_by_x()

        tip_count = tips.num_detections
        expected_height = calibration_fn(expected_volume)

        # No tips detected
        if tip_count == 0:
            return LiquidCheckResult(
                passed=False,
                detected_levels=[],
                channel_pass_status=[],
                expected_height_percent=expected_height,
                expected_volume=expected_volume,
                tip_count=0,
                liquid_count=0,
                error_message="No tips detected.",
            )

        # Calculate liquid levels by matching liquids to nearest tips
        detected_levels: List[float] = []
        tip_boxes = tips.bboxes_xyxy
        liq_boxes = liquids.bboxes_xyxy

        for liq_box in liq_boxes:
            liq_cx = (liq_box[0] + liq_box[2]) / 2.0

            # Find nearest tip
            tip_cxs = (tip_boxes[:, 0] + tip_boxes[:, 2]) / 2.0
            nearest_idx = int(np.argmin(np.abs(tip_cxs - liq_cx)))
            nearest_tip = tip_boxes[nearest_idx]

            liquid_h = float(liq_box[3] - liq_box[1])
            tip_h = float(nearest_tip[3] - nearest_tip[1])

            pct = (liquid_h / tip_h * 100.0) if tip_h > 0 else 0.0
            detected_levels.append(pct)

        liquid_count = len(detected_levels)

        # Evaluate pass/fail per channel
        channel_status = [abs(lvl - expected_height) <= self._tolerance for lvl in detected_levels]
        all_ok = len(channel_status) > 0 and all(channel_status)
        count_mismatch = liquid_count != tip_count
        tips_correct = tip_count == expected_tips

        error_msg: Optional[str] = None

        if not all_ok:
            failed = [i for i, ok in enumerate(channel_status) if not ok]
            error_msg = f"Levels out of range on channels {failed}. Expected {expected_height:.2f}%."
            passed = False
        elif count_mismatch:
            strict = all(abs(lvl - expected_height) <= self._tolerance for lvl in detected_levels)
            if tips_correct and strict:
                passed = True
                logger.warning(
                    "Tip count %d matches, but only %d liquids detected. "
                    "All within tolerance — proceeding.",
                    tip_count, liquid_count,
                )
            else:
                passed = False
                error_msg = (
                    f"Liquid count mismatch ({liquid_count}/{tip_count}) "
                    "and safety criteria not met."
                )
        else:
            passed = True
            logger.info("All levels valid and counts match.")

        if error_msg:
            logger.error(error_msg)

        return LiquidCheckResult(
            passed=passed,
            detected_levels=detected_levels,
            channel_pass_status=channel_status,
            expected_height_percent=expected_height,
            expected_volume=expected_volume,
            tip_count=tip_count,
            liquid_count=liquid_count,
            error_message=error_msg,
        )


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def build_calibration_from_csv(
    csv_path: str,
    degree: int = 3,
    channel_columns: Optional[Sequence[str]] = None,
) -> CalibrationFn:
    """Build a volume-to-height calibration function from CSV data.

    The CSV should contain a ``Volume`` column and per-channel height columns.
    A polynomial regression of the given *degree* is fit on the mean channel
    height.

    Args:
        csv_path: Path to calibration CSV.
        degree: Polynomial degree (default 3).
        channel_columns: Column names for channels.
            Default: ``Ch1`` … ``Ch8``.

    Returns:
        Callable mapping volume (uL) → expected height (percent).

    Raises:
        ImportError: If ``pandas`` or ``scikit-learn`` are not installed.
    """
    try:
        import pandas as pd
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import PolynomialFeatures
    except ImportError:
        raise ImportError(
            "pandas and scikit-learn are required for calibration. "
            "Install with: pip install openot2[calibration]"
        ) from None

    cols = list(channel_columns or [f"Ch{i}" for i in range(1, 9)])

    df = pd.read_csv(csv_path)
    df["MeanHeight"] = df[cols].mean(axis=1)

    X = df["Volume"].values.reshape(-1, 1)
    y = df["MeanHeight"].values

    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)

    model = LinearRegression()
    model.fit(X_poly, y)

    def _predict(volume_ul: float) -> float:
        vol = np.array([[volume_ul]])
        return float(model.predict(poly.transform(vol))[0])

    return _predict
