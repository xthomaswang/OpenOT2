"""Mock-based tests for openot2.precheck — no real hardware needed."""

import unittest
from unittest.mock import patch, MagicMock

from openot2.precheck import (
    RobotStatus,
    CameraInfo,
    PrecheckReport,
    check_robot_connection,
    probe_cameras,
    run_device_precheck,
)


class TestRobotStatus(unittest.TestCase):
    def test_defaults(self):
        s = RobotStatus(reachable=True)
        self.assertTrue(s.reachable)
        self.assertEqual(s.name, "")
        self.assertEqual(s.error, "")


class TestPrecheckReport(unittest.TestCase):
    def test_all_ok_true(self):
        r = PrecheckReport(
            robot=RobotStatus(reachable=True),
            cameras=[CameraInfo(device_id=0, width=1920, height=1080)],
        )
        self.assertTrue(r.all_ok)

    def test_all_ok_no_camera(self):
        r = PrecheckReport(robot=RobotStatus(reachable=True), cameras=[])
        self.assertFalse(r.all_ok)

    def test_all_ok_robot_down(self):
        r = PrecheckReport(
            robot=RobotStatus(reachable=False, error="timeout"),
            cameras=[CameraInfo(device_id=0, width=640, height=480)],
        )
        self.assertFalse(r.all_ok)


class TestCheckRobotConnection(unittest.TestCase):
    @patch("openot2.client.OT2Client.health")
    def test_reachable(self, mock_health):
        mock_health.return_value = {
            "name": "OT-2 Alpha",
            "api_version": "5.0.0",
        }
        status = check_robot_connection("169.254.8.56")
        self.assertTrue(status.reachable)
        self.assertEqual(status.name, "OT-2 Alpha")
        self.assertEqual(status.api_version, "5.0.0")
        self.assertEqual(status.error, "")

    @patch("openot2.client.OT2Client.health")
    def test_unreachable(self, mock_health):
        mock_health.side_effect = ConnectionError("refused")
        status = check_robot_connection("192.168.1.99")
        self.assertFalse(status.reachable)
        self.assertIn("refused", status.error)


class TestProbeCameras(unittest.TestCase):
    @patch("openot2.vision.camera.USBCamera.list_cameras")
    def test_found_cameras(self, mock_list):
        mock_list.return_value = [
            {"id": 0, "width": 1920, "height": 1080, "backend": 1200},
            {"id": 2, "width": 640, "height": 480, "backend": 1200},
        ]
        cams = probe_cameras(max_id=5)
        self.assertEqual(len(cams), 2)
        self.assertEqual(cams[0].device_id, 0)
        self.assertEqual(cams[0].width, 1920)
        self.assertEqual(cams[1].device_id, 2)
        mock_list.assert_called_once_with(max_id=5)

    @patch("openot2.vision.camera.USBCamera.list_cameras")
    def test_no_cameras(self, mock_list):
        mock_list.return_value = []
        cams = probe_cameras()
        self.assertEqual(len(cams), 0)


class TestRunDevicePrecheck(unittest.TestCase):
    @patch("openot2.precheck.probe_cameras")
    @patch("openot2.precheck.check_robot_connection")
    def test_all_ok(self, mock_robot, mock_cams):
        mock_robot.return_value = RobotStatus(reachable=True, name="bot")
        mock_cams.return_value = [CameraInfo(device_id=0, width=1920, height=1080)]

        report = run_device_precheck("169.254.8.56")
        self.assertTrue(report.all_ok)
        self.assertEqual(report.robot.name, "bot")
        self.assertEqual(len(report.cameras), 1)

    @patch("openot2.precheck.probe_cameras")
    @patch("openot2.precheck.check_robot_connection")
    def test_robot_down(self, mock_robot, mock_cams):
        mock_robot.return_value = RobotStatus(reachable=False, error="timeout")
        mock_cams.return_value = [CameraInfo(device_id=0, width=640, height=480)]

        report = run_device_precheck("10.0.0.1")
        self.assertFalse(report.all_ok)

    @patch("openot2.precheck.probe_cameras")
    @patch("openot2.precheck.check_robot_connection")
    def test_preview_false_no_matplotlib(self, mock_robot, mock_cams):
        """preview=False should never import matplotlib."""
        mock_robot.return_value = RobotStatus(reachable=True)
        mock_cams.return_value = [CameraInfo(device_id=0, width=1920, height=1080)]

        report = run_device_precheck("169.254.8.56", preview=False)
        self.assertTrue(report.all_ok)


if __name__ == "__main__":
    unittest.main()
