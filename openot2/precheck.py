"""Generic device precheck for OT-2 robot and USB cameras.

Returns structured results — no project-specific logic lives here.

Usage::

    from openot2.precheck import run_device_precheck

    report = run_device_precheck(robot_ip="169.254.8.56")
    if report.all_ok:
        print("All devices ready")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("openot2.precheck")


# ------------------------------------------------------------------
# Result dataclasses
# ------------------------------------------------------------------

@dataclass
class RobotStatus:
    """Result of an OT-2 connectivity check."""
    reachable: bool
    name: str = ""
    api_version: str = ""
    error: str = ""


@dataclass
class CameraInfo:
    """One detected USB camera."""
    device_id: int
    width: int
    height: int
    backend: int = 0


@dataclass
class PrecheckReport:
    """Combined precheck results for robot + cameras."""
    robot: RobotStatus
    cameras: List[CameraInfo] = field(default_factory=list)

    @property
    def all_ok(self) -> bool:
        return self.robot.reachable and len(self.cameras) > 0


# ------------------------------------------------------------------
# Individual checks
# ------------------------------------------------------------------

def check_robot_connection(
    robot_ip: str,
    port: int = 31950,
    timeout: float = 5.0,
) -> RobotStatus:
    """Check OT-2 reachability via ``GET /health``.

    Uses :class:`~openot2.client.OT2Client` internally so the retry /
    header logic stays in one place.
    """
    from openot2.client import OT2Client

    try:
        client = OT2Client(robot_ip=robot_ip, port=port, timeout=timeout, max_retries=0)
        data = client.health(timeout=timeout)
        return RobotStatus(
            reachable=True,
            name=data.get("name", ""),
            api_version=data.get("api_version", ""),
        )
    except Exception as exc:
        logger.warning("Robot unreachable at %s:%s — %s", robot_ip, port, exc)
        return RobotStatus(reachable=False, error=str(exc))


def probe_cameras(max_id: int = 10) -> List[CameraInfo]:
    """Enumerate USB cameras via :func:`openot2.vision.USBCamera.list_cameras`.

    Returns a list of :class:`CameraInfo` without opening preview windows.
    """
    from openot2.vision.camera import USBCamera

    raw = USBCamera.list_cameras(max_id=max_id)
    return [
        CameraInfo(
            device_id=c["id"],
            width=c["width"],
            height=c["height"],
            backend=c.get("backend", 0),
        )
        for c in raw
    ]


# ------------------------------------------------------------------
# Combined precheck
# ------------------------------------------------------------------

def run_device_precheck(
    robot_ip: str,
    port: int = 31950,
    timeout: float = 5.0,
    max_camera_id: int = 10,
    expected_camera_id: Optional[int] = None,
    preview: bool = False,
) -> PrecheckReport:
    """Run robot + camera checks and return a :class:`PrecheckReport`.

    Args:
        robot_ip: OT-2 IPv4 address.
        port: OT-2 HTTP API port.
        timeout: Seconds to wait for the robot /health response.
        max_camera_id: Highest device index to probe for cameras.
        expected_camera_id: If set, warn when this ID is not found.
        preview: If *True*, show a matplotlib preview of each camera
            (delegates to :func:`openot2.vision.precheck_cameras`).

    Returns:
        A :class:`PrecheckReport` with structured results.
    """
    robot_status = check_robot_connection(robot_ip, port, timeout)
    cameras = probe_cameras(max_id=max_camera_id)

    if expected_camera_id is not None:
        found_ids = [c.device_id for c in cameras]
        if expected_camera_id not in found_ids:
            logger.warning(
                "Expected camera %d not found. Available: %s",
                expected_camera_id, found_ids,
            )

    if preview and cameras:
        try:
            from openot2.vision.camera import USBCamera
            USBCamera.precheck_cameras(
                expected_id=expected_camera_id or 0,
                max_id=max_camera_id,
            )
        except Exception as exc:
            logger.warning("Camera preview failed: %s", exc)

    report = PrecheckReport(robot=robot_status, cameras=cameras)

    if robot_status.reachable:
        logger.info("Robot OK: %s (API %s)", robot_status.name, robot_status.api_version)
    else:
        logger.warning("Robot UNREACHABLE: %s", robot_status.error)
    logger.info("Cameras found: %d", len(cameras))

    return report
