"""
OpenOT2 — Basic Usage Example
===============================

This example demonstrates:
1. Connecting to an OT-2 robot
2. Setting up a YOLO vision model
3. Defining a protocol config
4. Executing the protocol with vision checks

Prerequisites:
    pip install openot2[yolo]
"""

from openot2 import OT2Client, setup_logging
from openot2.vision import YOLOAdapter, USBCamera, TipAnalyzer, LiquidAnalyzer
from openot2.vision.analyzers import build_calibration_from_csv
from openot2.protocol import ProtocolExecutor

# --- 1. Setup Logging ---
setup_logging()  # Use setup_logging(logging.DEBUG) for verbose output

# --- 2. Connect to Robot ---
client = OT2Client("169.254.8.56")  # Replace with your robot's IP

# --- 3. Load Vision Model ---
model = YOLOAdapter(
    model_path="path/to/your/model.pt",
    num_classes=2,  # Tip, Liquid
)

# --- 4. Setup Camera ---
camera = USBCamera(camera_id=0)

# --- 5. Load Calibration (optional, for liquid level checks) ---
# calibration = build_calibration_from_csv("path/to/LLD_Calibration_Data.csv")
calibration = None  # Set to None to skip liquid level verification

# --- 6. Define Protocol ---
config = {
    "labware": {
        "pipette": {"name": "p300_multi_gen2", "mount": "right"},
        "tiprack": {"name": "opentrons_96_filtertiprack_200ul", "slot": "11"},
        "sources": {"name": "opentrons_tough_1_reservoir_300ml", "slots": ["8"]},
        "imaging": {"name": "opentrons_96_wellplate_200ul_pcr_full_skirt", "slot": "6"},
        "dispense": {"name": "corning_96_wellplate_360ul_flat", "slot": "10"},
    },
    "settings": {
        "imaging_well": "A1",
        "imaging_offset": (0, 0, 50),
        "base_dir": "output",
    },
    "tasks": [
        # Task 1: Pick up tips with vision verification
        {
            "type": "pickup",
            "well": "A1",
            "check": {
                "type": "tip",
                "expected_tips": 8,
                "conf": 0.6,
            },
        },
        # Task 2: Transfer liquid with vision verification
        {
            "type": "transfer",
            "source_slot": "8",
            "source_well": "A1",
            "dest_slot": "10",
            "dest_well": "A1",
            "volume": 100,
            "origin": "top",
            "offset": (0, 0, -35),
            "check": {
                "type": "liquid",
                "expected_tips": 8,
                "conf": 0.6,
            },
        },
        # Task 3: Drop tips
        {
            "type": "drop",
            "well": "A1",
        },
    ],
}

# --- 7. Execute ---
executor = ProtocolExecutor(
    client=client,
    model=model,
    camera=camera,
    calibration_fn=calibration,
)

if __name__ == "__main__":
    executor.execute(config)
