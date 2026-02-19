"""Hardware tests requiring a real OT-2 robot.

These tests are skipped unless the ``OT2_ROBOT_IP`` environment variable is set.

Run: ``OT2_ROBOT_IP=169.254.8.56 pytest tests/test_hardware.py``
"""

from __future__ import annotations

import os

import pytest

from openot2.client import OT2Client

ROBOT_IP = os.environ.get("OT2_ROBOT_IP")

pytestmark = pytest.mark.hardware

skip_no_robot = pytest.mark.skipif(
    ROBOT_IP is None,
    reason="OT2_ROBOT_IP env var not set — skipping hardware tests",
)


@skip_no_robot
class TestHardwareConnection:
    """Tests that require a real OT-2 on the network."""

    def test_create_run(self):
        client = OT2Client(ROBOT_IP)
        run_id = client.create_run()
        assert run_id is not None
        assert len(run_id) > 0

    def test_load_pipette_and_labware(self):
        client = OT2Client(ROBOT_IP)
        client.create_run()

        pid = client.load_pipette("p300_multi_gen2", mount="right")
        assert pid is not None

        lid = client.load_labware(
            "opentrons_96_filtertiprack_200ul", slot="11",
        )
        assert lid is not None
        assert client.get_labware_id("11") == lid

    def test_home(self):
        client = OT2Client(ROBOT_IP)
        client.create_run()
        client.home()  # Should not raise

    def test_reconnect_last_run(self):
        client = OT2Client(ROBOT_IP)
        # First create a run so there's something to reconnect to
        original_id = client.create_run()
        client.load_pipette("p300_multi_gen2")
        client.load_labware("opentrons_96_filtertiprack_200ul", slot="11")

        # Now reconnect
        client2 = OT2Client(ROBOT_IP)
        reconnected_id = client2.reconnect_last_run()
        assert reconnected_id is not None
        assert client2.pipette_id is not None

    def test_pick_up_and_drop_tip(self):
        """Full cycle: create run → load → pick up → drop → home."""
        client = OT2Client(ROBOT_IP)
        client.create_run()
        client.load_pipette("p300_multi_gen2", mount="right")
        tiprack_id = client.load_labware(
            "opentrons_96_filtertiprack_200ul", slot="11",
        )

        client.pick_up_tip(tiprack_id, well="A1")
        client.drop_tip(tiprack_id, well="A1")
        client.home()
