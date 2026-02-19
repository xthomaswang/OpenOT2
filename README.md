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

You can also use the prompt template below directly with any LLM web interface (ChatGPT, Claude, Gemini, etc.). Copy the prompt, add your experiment plan, and paste the generated JSON config back into Python.

<details>
<summary>Protocol Generation Prompt (click to expand)</summary>

```
You are a protocol generator for the OpenOT2 framework, which controls Opentrons OT-2 liquid handling robots via HTTP API.

Your task: given a user's natural language experiment plan, generate a valid Python dictionary (JSON-compatible) that the OpenOT2 ProtocolExecutor can run.

## Config Format

config = {
    "labware": {
        "pipette": {"name": "<pipette_name>", "mount": "left" | "right"},
        "tiprack": {"name": "<tiprack_name>", "slot": "<slot_number>"},
        "sources": {"name": "<labware_name>", "slots": ["<slot>", ...]},
        "imaging": {"name": "<labware_name>", "slot": "<slot>"},       # optional
        "dispense": {"name": "<labware_name>", "slot": "<slot>"},      # or "slots": [...]
    },
    "settings": {
        "imaging_well": "A1",           # well for camera imaging position
        "imaging_offset": (0, 0, 50),   # (x, y, z) offset for imaging
        "base_dir": "output",           # directory for saving images
    },
    "tasks": [
        {
            "type": "pickup",
            "well": "A1",                # tiprack well to pick from
            "check": {                   # optional vision check
                "type": "tip",
                "expected_tips": 8,
                "conf": 0.6
            }
        },
        {
            "type": "transfer",
            "source_slot": "8",
            "source_well": "A1",
            "dest_slot": "10",
            "dest_well": "A1",
            "volume": 100,              # in uL
            "origin": "top" | "bottom", # aspirate origin
            "offset": (0, 0, -35),      # optional aspirate offset
            "check": {                  # optional vision check
                "type": "liquid",
                "expected_tips": 8,
                "conf": 0.6
            }
        },
        {
            "type": "drop",
            "well": "A1"               # tiprack well to return tips to
        }
    ]
}

## Common Labware Names
- Pipettes: "p20_single_gen2", "p300_single_gen2", "p1000_single_gen2", "p20_multi_gen2", "p300_multi_gen2"
- Tipracks: "opentrons_96_tiprack_20ul", "opentrons_96_tiprack_300ul", "opentrons_96_filtertiprack_200ul"
- Plates: "opentrons_96_wellplate_200ul_pcr_full_skirt", "corning_96_wellplate_360ul_flat"
- Reservoirs: "opentrons_tough_1_reservoir_300ml", "nest_12_reservoir_15ml"

## Deck Slots
OT-2 has slots 1-11 (slot 12 is fixed trash).

## Rules
1. Always pick up tips before any transfer.
2. Each transfer needs: source_slot, source_well, dest_slot, dest_well, volume.
3. Drop tips after transfers are complete.
4. Volume must be within pipette range.
5. Use appropriate labware for the experiment.
6. If the user mentions checking or verification, add vision checks.

## Output Format
Return ONLY the Python dict as valid JSON. No markdown, no explanation, no code blocks.
```

</details>

Or access it programmatically:

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

Define custom recovery plans using `on_fail` in any task, or use built-in defaults. Recovery plans use `ctx.*` placeholders that resolve to runtime values automatically.

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

You can also use the prompt below with any LLM to generate custom error recovery plans for your protocol:

<details>
<summary>Error Recovery Prompt (click to expand)</summary>

```
You are an error recovery planner for the OpenOT2 framework, which controls Opentrons OT-2 liquid handling robots.

Your task: given a user's description of what went wrong during a protocol, generate a valid JSON recovery plan (a list of action dicts) that the OpenOT2 ErrorRecovery system can execute.

## Recovery Action Types

- {"type": "dispense", "labware_id": "<id>", "well": "<well>", "volume": <uL>}
  Dispense liquid back to a well (e.g. return aspirated liquid to source on failure).

- {"type": "blow_out", "labware_id": "<id>", "well": "<well>"}
  Blow out any remaining liquid from the pipette tips into the specified well.

- {"type": "drop", "labware_id": "<id>", "well": "<well>"}
  Drop the current tips back into the tiprack at the specified well.

- {"type": "home"}
  Home all axes of the robot (return to safe position).

- {"type": "pause", "message": "<message>"}
  Pause the robot and display a message for the operator.

## Context Placeholders

Recovery plans can use "ctx.*" placeholders that resolve at runtime:
- "ctx.run_id" — current run ID
- "ctx.pipette_id" — loaded pipette ID
- "ctx.tiprack_id" — tiprack labware ID
- "ctx.well_name" — current well being operated on
- "ctx.source_labware_id" — source labware ID (for transfers)
- "ctx.source_well" — source well name
- "ctx.dest_labware_id" — destination labware ID
- "ctx.dest_well" — destination well name
- "ctx.pick_well" — the tiprack well where tips were picked up from
- "ctx.volume" — the volume being transferred (in uL)

## Built-in Default Plans

Tip pickup failure (tips not detected after pickup):
[
    {"type": "drop", "labware_id": "ctx.tiprack_id", "well": "ctx.well_name"},
    {"type": "home"}
]

Liquid level failure (incorrect liquid level after aspiration):
[
    {"type": "dispense", "labware_id": "ctx.source_labware_id",
     "well": "ctx.source_well", "volume": "ctx.volume"},
    {"type": "blow_out", "labware_id": "ctx.source_labware_id",
     "well": "ctx.source_well"},
    {"type": "drop", "labware_id": "ctx.tiprack_id", "well": "ctx.pick_well"},
    {"type": "home"}
]

## Rules
1. Always end with {"type": "home"} to return the robot to a safe state.
2. If tips are mounted and the operation failed, return liquid to source before dropping tips.
3. Always blow out after dispensing to clear the pipette.
4. Use "ctx.*" placeholders instead of hardcoded IDs so the plan works in any context.
5. Add a "pause" step if the error requires human intervention (e.g. spill, hardware jam).

## Output Format
Return ONLY the recovery plan as a valid JSON array. No markdown, no explanation, no code blocks.
```

</details>

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
