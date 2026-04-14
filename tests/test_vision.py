"""Tests for vision module: PredictionResult, analyzers, calibration."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from openot2.vision.base_types import PredictionResult
from openot2.vision.analyzers import (
    LiquidAnalyzer,
    TipAnalyzer,
    build_calibration_from_csv,
)


# ---------------------------------------------------------------------------
# PredictionResult
# ---------------------------------------------------------------------------

class TestPredictionResult:
    def test_num_detections(self):
        pr = PredictionResult(
            labels=np.array([0, 1, 0]),
            bboxes_xyxy=np.array([[0, 0, 10, 10], [20, 20, 30, 30], [40, 40, 50, 50]]),
            class_names=["Tip", "Liquid"],
            confidences=np.array([0.9, 0.8, 0.7]),
        )
        assert pr.num_detections == 3

    def test_filter_by_class(self):
        pr = PredictionResult(
            labels=np.array([0, 1, 0, 1]),
            bboxes_xyxy=np.array([
                [0, 0, 10, 10], [20, 20, 30, 30],
                [40, 40, 50, 50], [60, 60, 70, 70],
            ]),
            class_names=["Tip", "Liquid"],
            confidences=np.array([0.9, 0.8, 0.7, 0.6]),
        )
        tips = pr.filter_by_class("Tip")
        assert tips.num_detections == 2
        assert all(tips.labels == 0)

        liquids = pr.filter_by_class("Liquid")
        assert liquids.num_detections == 2

    def test_filter_by_unknown_class(self):
        pr = PredictionResult(
            labels=np.array([0]),
            bboxes_xyxy=np.array([[0, 0, 10, 10]]),
            class_names=["Tip"],
            confidences=np.array([0.9]),
        )
        result = pr.filter_by_class("Unknown")
        assert result.num_detections == 0

    def test_sort_by_x(self):
        pr = PredictionResult(
            labels=np.array([0, 0, 0]),
            bboxes_xyxy=np.array([
                [100, 0, 110, 10],
                [0, 0, 10, 10],
                [50, 0, 60, 10],
            ]),
            class_names=["Tip"],
            confidences=np.array([0.7, 0.9, 0.8]),
        )
        sorted_pr = pr.sort_by_x()
        assert sorted_pr.bboxes_xyxy[0, 0] == 0.0
        assert sorted_pr.bboxes_xyxy[1, 0] == 50.0
        assert sorted_pr.bboxes_xyxy[2, 0] == 100.0
        # Confidences follow the sort
        assert sorted_pr.confidences[0] == 0.9
        assert sorted_pr.confidences[1] == 0.8

    def test_sort_empty(self):
        pr = PredictionResult(
            labels=np.array([], dtype=int),
            bboxes_xyxy=np.empty((0, 4)),
            class_names=["Tip"],
            confidences=np.array([]),
        )
        sorted_pr = pr.sort_by_x()
        assert sorted_pr.num_detections == 0


# ---------------------------------------------------------------------------
# TipAnalyzer
# ---------------------------------------------------------------------------

class TestTipAnalyzer:
    def test_all_tips_present(self, sample_tip_prediction):
        analyzer = TipAnalyzer()
        result = analyzer.analyze(sample_tip_prediction, expected_tips=8)
        assert result.passed is True
        assert result.tip_count == 8
        assert result.missing_positions == []
        assert all(p == 1 for p in result.tip_presence)

    def test_missing_tips(self, sample_tip_prediction_missing):
        analyzer = TipAnalyzer()
        result = analyzer.analyze(sample_tip_prediction_missing, expected_tips=8)
        assert result.passed is False
        assert result.tip_count == 6
        assert len(result.missing_positions) == 2
        assert 3 in result.missing_positions  # 1-based position
        assert 6 in result.missing_positions

    def test_no_tips(self):
        pr = PredictionResult(
            labels=np.array([], dtype=int),
            bboxes_xyxy=np.empty((0, 4)),
            class_names=["Tip", "Liquid"],
            confidences=np.array([]),
        )
        analyzer = TipAnalyzer()
        result = analyzer.analyze(pr, expected_tips=8)
        assert result.passed is False
        assert result.tip_count == 0
        assert result.missing_positions == list(range(1, 9))

    def test_custom_class_name(self):
        pr = PredictionResult(
            labels=np.array([0, 0]),
            bboxes_xyxy=np.array([[0, 0, 10, 10], [50, 0, 60, 10]]),
            class_names=["pipette_tip"],
            confidences=np.array([0.9, 0.9]),
        )
        analyzer = TipAnalyzer(tip_class_name="pipette_tip")
        result = analyzer.analyze(pr, expected_tips=2)
        assert result.passed is True


# ---------------------------------------------------------------------------
# LiquidAnalyzer
# ---------------------------------------------------------------------------

class TestLiquidAnalyzer:
    def _make_calibration(self, fixed_height: float = 50.0):
        """Return a calibration function that always returns fixed_height."""
        return lambda vol: fixed_height

    def test_pass_all_levels_ok(self, sample_liquid_prediction):
        cal = self._make_calibration(50.0)
        analyzer = LiquidAnalyzer(tolerance_percent=5.0)
        result = analyzer.analyze(sample_liquid_prediction, 100.0, cal, expected_tips=8)
        assert result.passed is True
        assert result.tip_count == 8
        assert result.liquid_count == 8

    def test_fail_levels_out_of_range(self, sample_liquid_prediction):
        # Expect 90% but actual is ~50%
        cal = self._make_calibration(90.0)
        analyzer = LiquidAnalyzer(tolerance_percent=5.0)
        result = analyzer.analyze(sample_liquid_prediction, 100.0, cal, expected_tips=8)
        assert result.passed is False
        assert result.error_message is not None

    def test_no_tips_detected(self):
        pr = PredictionResult(
            labels=np.array([], dtype=int),
            bboxes_xyxy=np.empty((0, 4)),
            class_names=["Tip", "Liquid"],
            confidences=np.array([]),
        )
        cal = self._make_calibration(50.0)
        analyzer = LiquidAnalyzer()
        result = analyzer.analyze(pr, 100.0, cal)
        assert result.passed is False
        assert result.error_message == "No tips detected."


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

class TestCalibration:
    def test_build_from_csv(self):
        """Test calibration with a simple CSV."""
        try:
            import pandas  # noqa: F401
            import sklearn  # noqa: F401
        except ImportError:
            pytest.skip("pandas/scikit-learn not installed")

        csv_content = "Volume,Ch1,Ch2,Ch3,Ch4,Ch5,Ch6,Ch7,Ch8\n"
        csv_content += "0,0,0,0,0,0,0,0,0\n"
        csv_content += "50,25,25,25,25,25,25,25,25\n"
        csv_content += "100,50,50,50,50,50,50,50,50\n"
        csv_content += "200,90,90,90,90,90,90,90,90\n"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            f.flush()
            csv_path = f.name

        try:
            cal_fn = build_calibration_from_csv(csv_path, degree=2)
            # At 100uL, should predict ~50%
            predicted = cal_fn(100)
            assert 40 < predicted < 60
            # At 0uL, should predict ~0%
            predicted_zero = cal_fn(0)
            assert -10 < predicted_zero < 10
        finally:
            os.unlink(csv_path)


# ---------------------------------------------------------------------------
# Camera (mock-based)
# ---------------------------------------------------------------------------

class TestUSBCamera:
    def test_detect_backend(self):
        """_detect_backend should return an int without crashing."""
        from openot2.vision.camera import USBCamera
        backend = USBCamera._detect_backend()
        assert isinstance(backend, int)
