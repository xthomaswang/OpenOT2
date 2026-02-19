"""Shared fixtures for OpenOT2 tests."""

from __future__ import annotations

import json
import uuid
from typing import Any, Dict
from unittest.mock import MagicMock

import numpy as np
import pytest
import responses

from openot2.client import OT2Client
from openot2.vision.base_types import PredictionResult

ROBOT_IP = "127.0.0.1"
BASE_URL = f"http://{ROBOT_IP}:31950"


# ---------------------------------------------------------------------------
# Mock OT-2 HTTP server
# ---------------------------------------------------------------------------

def _make_run_id() -> str:
    return str(uuid.uuid4())


def _make_labware_id() -> str:
    return f"labware-{uuid.uuid4().hex[:8]}"


def _make_pipette_id() -> str:
    return f"pipette-{uuid.uuid4().hex[:8]}"


@pytest.fixture
def mock_ot2_server():
    """Activate ``responses`` mock and register common OT-2 endpoints."""
    with responses.RequestsMock(assert_all_requests_are_fired=False) as rsps:
        run_id = _make_run_id()
        pipette_id = _make_pipette_id()
        labware_id = _make_labware_id()

        # POST /runs -> create run
        rsps.add(
            responses.POST,
            f"{BASE_URL}/runs",
            json={"data": {"id": run_id}},
            status=200,
        )

        # GET /runs -> list runs (for reconnect)
        rsps.add(
            responses.GET,
            f"{BASE_URL}/runs",
            json={
                "data": [
                    {
                        "id": run_id,
                        "pipettes": [{"id": pipette_id, "mount": "right"}],
                        "labware": [
                            {"id": labware_id, "location": {"slotName": "11"}},
                        ],
                    }
                ]
            },
            status=200,
        )

        # POST /runs/{id}/commands -> generic command response
        def _command_callback(request):
            body = json.loads(request.body)
            cmd_type = body.get("data", {}).get("commandType", "unknown")
            result: Dict[str, Any] = {}

            if cmd_type == "loadPipette":
                result = {"pipetteId": pipette_id}
            elif cmd_type == "loadLabware":
                result = {"labwareId": _make_labware_id()}
            # other commands don't need special results

            return (200, {}, json.dumps({"data": {"result": result}}))

        rsps.add_callback(
            responses.POST,
            f"{BASE_URL}/runs/{run_id}/commands",
            callback=_command_callback,
            content_type="application/json",
        )

        yield {
            "rsps": rsps,
            "run_id": run_id,
            "pipette_id": pipette_id,
            "labware_id": labware_id,
        }


@pytest.fixture
def mock_client(mock_ot2_server) -> OT2Client:
    """OT2Client connected to the mocked HTTP server."""
    client = OT2Client(ROBOT_IP)
    client.create_run()
    return client


# ---------------------------------------------------------------------------
# Sample predictions
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_tip_prediction() -> PredictionResult:
    """PredictionResult with 8 tips detected, evenly spaced."""
    n = 8
    spacing = 50.0
    bboxes = np.array([
        [i * spacing, 100.0, i * spacing + 30.0, 300.0] for i in range(n)
    ])
    return PredictionResult(
        labels=np.zeros(n, dtype=int),  # class 0 = Tip
        bboxes_xyxy=bboxes,
        class_names=["Tip", "Liquid"],
        confidences=np.ones(n) * 0.9,
        source_image_path="test.jpg",
    )


@pytest.fixture
def sample_tip_prediction_missing() -> PredictionResult:
    """PredictionResult with 6 of 8 tips (positions 3 and 6 missing)."""
    present = [0, 1, 3, 4, 6, 7]  # 0-indexed, missing 2 and 5
    spacing = 50.0
    bboxes = np.array([
        [i * spacing, 100.0, i * spacing + 30.0, 300.0] for i in present
    ])
    return PredictionResult(
        labels=np.zeros(len(present), dtype=int),
        bboxes_xyxy=bboxes,
        class_names=["Tip", "Liquid"],
        confidences=np.ones(len(present)) * 0.9,
    )


@pytest.fixture
def sample_liquid_prediction() -> PredictionResult:
    """PredictionResult with 8 tips and 8 matching liquids."""
    n = 8
    spacing = 50.0
    tip_bboxes = [
        [i * spacing, 50.0, i * spacing + 30.0, 250.0] for i in range(n)
    ]
    # Liquid fills ~50% of tip height (100 of 200 pixels)
    liq_bboxes = [
        [i * spacing + 2, 150.0, i * spacing + 28.0, 250.0] for i in range(n)
    ]
    bboxes = np.array(tip_bboxes + liq_bboxes)
    labels = np.array([0] * n + [1] * n, dtype=int)  # 0=Tip, 1=Liquid
    confs = np.ones(2 * n) * 0.9

    return PredictionResult(
        labels=labels,
        bboxes_xyxy=bboxes,
        class_names=["Tip", "Liquid"],
        confidences=confs,
    )


@pytest.fixture
def sample_config() -> Dict[str, Any]:
    """Valid minimal protocol config."""
    return {
        "labware": {
            "pipette": {"name": "p300_multi_gen2", "mount": "right"},
            "tiprack": {"name": "opentrons_96_filtertiprack_200ul", "slot": "11"},
            "sources": {"name": "opentrons_tough_1_reservoir_300ml", "slots": ["8"]},
            "dispense": {"name": "corning_96_wellplate_360ul_flat", "slot": "10"},
        },
        "settings": {
            "imaging_well": "A1",
            "imaging_offset": (0, 0, 50),
            "base_dir": "test_output",
        },
        "tasks": [
            {"type": "pickup", "well": "A1"},
            {
                "type": "transfer",
                "source_slot": "8",
                "source_well": "A1",
                "dest_slot": "10",
                "dest_well": "A1",
                "volume": 100,
            },
            {"type": "drop", "well": "A1"},
        ],
    }
