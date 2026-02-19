# OpenOT2

A modular Python framework for controlling [Opentrons OT-2](https://opentrons.com/ot-2/) liquid handling robots with integrated machine learning and computer vision for real-time protocol verification.

## Features

- **HTTP API Client** — Stateful client with session management, automatic retries, and full pipette/labware lifecycle control
- **Computer Vision Integration** — Real-time tip presence and liquid level verification using object detection models (YOLO, SuperGradients)
- **Task-Based Protocol Execution** — Declarative JSON/dict protocol configs with built-in vision checks at each step
- **Automatic Error Recovery** — Configurable recovery plans with context-based placeholder resolution
- **LLM-Powered Protocol Generation** — Generate protocol configs from natural language experiment descriptions using any OpenAI-compatible API
- **Dry-Run Simulation** — Validate protocols offline before sending commands to the robot
- **Liquid Level Calibration** — Polynomial regression from CSV calibration data for quantitative liquid verification

## Architecture

```
openot2/
├── client.py              # OT-2 HTTP API client
├── utils.py               # Logging, labware loading, well mapping
├── vision/
│   ├── base_types.py      # VisionModel ABC & PredictionResult
│   ├── models.py          # YOLOAdapter, SuperGradientsAdapter
│   ├── camera.py          # USBCamera with platform auto-detection
│   └── analyzers.py       # TipAnalyzer, LiquidAnalyzer, calibration
└── protocol/
    ├── executor.py        # ProtocolExecutor with vision checks
    ├── generator.py       # LLM-powered protocol generation & dry-run
    └── recovery.py        # ErrorRecovery with context resolution
```

## Installation

```bash
pip install openot2
```

### Optional Dependencies

```bash
# YOLO object detection (ultralytics)
pip install openot2[yolo]

# SuperGradients object detection
pip install openot2[supergradients]

# Liquid level calibration (pandas + scikit-learn)
pip install openot2[calibration]

# Everything
pip install openot2[all]
```

> **Note:** The base install includes the OpenAI SDK for LLM-powered protocol generation. Vision model libraries (YOLO, SuperGradients) are optional due to their large size (PyTorch dependency).

## Quick Start

```python
from openot2 import OT2Client, setup_logging
from openot2.vision import YOLOAdapter, USBCamera
from openot2.protocol import ProtocolExecutor

setup_logging()

# Connect to robot
client = OT2Client("169.254.8.56")

# Load vision model and camera
model = YOLOAdapter(model_path="path/to/model.pt", num_classes=2)
camera = USBCamera(camera_id=0)

# Define protocol
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
        {
            "type": "pickup",
            "well": "A1",
            "check": {"type": "tip", "expected_tips": 8, "conf": 0.6},
        },
        {
            "type": "transfer",
            "source_slot": "8",
            "source_well": "A1",
            "dest_slot": "10",
            "dest_well": "A1",
            "volume": 100,
            "origin": "top",
            "offset": (0, 0, -35),
            "check": {"type": "liquid", "expected_tips": 8, "conf": 0.6},
        },
        {"type": "drop", "well": "A1"},
    ],
}

# Execute
executor = ProtocolExecutor(client=client, model=model, camera=camera)
executor.execute(config)
```

## LLM Protocol Generation

Generate protocols from natural language using any OpenAI-compatible API:

```python
from openot2.protocol import ProtocolGenerator

generator = ProtocolGenerator(api_key="sk-...", model="gpt-4o")

config, dry_run = generator.generate_and_validate(
    "Transfer 100uL from reservoir in slot 8 to a 96-well plate in slot 10, "
    "using a p300 multi-channel pipette with tip verification."
)

if dry_run.success:
    executor.execute(config)
```

You can also use `get_protocol_prompt()` to get the prompt template for manual use with any LLM web interface:

```python
from openot2.protocol import get_protocol_prompt
print(get_protocol_prompt())
```

## Vision System

The vision subsystem verifies operations in real time by capturing images and running object detection:

- **TipAnalyzer** — Checks that all expected tips are present after pickup by analyzing spatial positions
- **LiquidAnalyzer** — Verifies liquid levels against calibration curves after aspiration
- **Calibration** — Build volume-to-height mapping from CSV data using polynomial regression

```python
from openot2.vision import build_calibration_from_csv

calibration_fn = build_calibration_from_csv("LLD_Calibration_Data.csv", degree=3)
# calibration_fn(100.0) -> expected liquid height in percent
```

### Custom Vision Models

Implement the `VisionModel` abstract class to integrate any detection framework:

```python
from openot2.vision import VisionModel, PredictionResult

class MyCustomModel(VisionModel):
    def predict(self, image_path: str, conf: float = 0.4) -> PredictionResult:
        # Your inference logic here
        ...

    @property
    def class_names(self) -> list[str]:
        return ["Tip", "Liquid"]
```

## Error Recovery

Define custom recovery plans or use built-in defaults:

```python
# Custom recovery in task config
{
    "type": "transfer",
    "source_slot": "8",
    "source_well": "A1",
    "dest_slot": "10",
    "dest_well": "A1",
    "volume": 100,
    "check": {"type": "liquid", "expected_tips": 8, "conf": 0.6},
    "on_fail": [
        {"type": "dispense", "labware_id": "ctx.source_labware_id",
         "well": "ctx.source_well", "volume": "ctx.volume"},
        {"type": "blow_out", "labware_id": "ctx.source_labware_id",
         "well": "ctx.source_well"},
        {"type": "drop", "labware_id": "ctx.tiprack_id", "well": "ctx.pick_well"},
        {"type": "home"},
    ],
}
```

## Requirements

- Python >= 3.10
- An Opentrons OT-2 robot accessible via network
- USB camera (for vision features)
- Trained object detection model weights (for vision features)

## Running Tests

```bash
pip install openot2[dev]
pytest                    # unit tests only
pytest -m hardware        # hardware tests (requires connected OT-2)
```

## License

MIT
