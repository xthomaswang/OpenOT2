# OpenOT2

A modular Python framework for controlling [Opentrons OT-2](https://opentrons.com/ot-2/) liquid handling robots with integrated machine learning, computer vision, and a web-based control interface.

## Features

- **HTTP API Client** — Stateful client with session management, automatic retries, and full pipette/labware lifecycle control
- **Computer Vision** — Real-time tip presence and liquid level verification using object detection models (YOLO, SuperGradients)
- **Protocol Execution** — Declarative JSON/dict protocol configs with built-in vision checks at each step
- **Automatic Error Recovery** — Configurable recovery plans with context-based placeholder resolution
- **LLM Protocol Generation** — Generate protocol configs from natural language using any OpenAI-compatible API
- **Web Controller** — Dark-themed dashboard with real-time run monitoring, setup wizard, and calibration UI
- **Liquid Level Calibration** — Polynomial regression from CSV calibration data for quantitative liquid verification

## Architecture

The project is split into four independent packages:

```
openot2/                  # Robot control core
├── client.py             # OT-2 HTTP API client
├── operations.py         # High-level pipetting workflows (transfer, mix, rinse)
├── utils.py              # Logging, labware loading, well mapping
├── precheck.py           # Hardware pre-validation
└── control/              # Task execution infrastructure
    ├── models.py         # RunStatus, RunStep, TaskRun, RunEvent
    ├── runner.py          # TaskRunner with pause/resume and ETA
    └── store.py          # JSON-file persistence

vision/                   # Computer vision (independent)
├── base_types.py         # VisionModel ABC & PredictionResult
├── models.py             # YOLOAdapter, SuperGradientsAdapter
├── camera.py             # USBCamera with platform auto-detection
└── analyzers.py          # TipAnalyzer, LiquidAnalyzer, calibration

protocol/                 # Protocol execution (depends on openot2 + vision)
├── executor.py           # ProtocolExecutor with vision checks
├── generator.py          # LLM-powered protocol generation & dry-run
└── recovery.py           # ErrorRecovery with context resolution

webapp/                   # Web application (depends on openot2)
├── app.py                # WebApp facade — wires everything together
├── web.py                # FastAPI app factory, API endpoints
├── calibration.py        # Calibration service
├── templates/            # Jinja2 templates (dashboard, setup wizard, calibration)
├── deck.py               # DeckConfig (load deck layout from YAML)
├── handlers.py           # OT2StepHandlers (pluggable step handlers)
└── __main__.py           # CLI: python -m webapp
```

## Installation

```bash
pip install openot2
```

### Optional Dependencies

```bash
pip install openot2[yolo]            # YOLO object detection
pip install openot2[supergradients]  # SuperGradients object detection
pip install openot2[calibration]     # Liquid level calibration (pandas + scikit-learn)
pip install openot2[web]             # Web controller (FastAPI + uvicorn + Jinja2)
pip install openot2[all]             # Everything
```

## Quick Start

### Robot Control

```python
from openot2 import OT2Client, OT2Operations

client = OT2Client("169.254.8.56")
client.create_run()
client.load_pipette("p300_single_gen2", "left")
tiprack = client.load_labware("opentrons_96_tiprack_300ul", "10")
source = client.load_labware("nest_12_reservoir_15ml", "7")
plate = client.load_labware("corning_96_wellplate_360ul_flat", "1")

ops = OT2Operations(client)
ops.transfer(tiprack, source, plate, "A1", "A1", "A1", volume=100)
```

### Protocol Execution with Vision

```python
from openot2 import OT2Client, setup_logging
from vision import YOLOAdapter, USBCamera
from protocol import ProtocolExecutor

setup_logging()

client = OT2Client("169.254.8.56")
model = YOLOAdapter(model_path="path/to/model.pt", num_classes=2)
camera = USBCamera(camera_id=0)

config = {
    "labware": {
        "pipette": {"name": "p300_multi_gen2", "mount": "right"},
        "tiprack": {"name": "opentrons_96_filtertiprack_200ul", "slot": "11"},
        "sources": {"name": "opentrons_tough_1_reservoir_300ml", "slots": ["8"]},
        "dispense": {"name": "corning_96_wellplate_360ul_flat", "slot": "10"},
    },
    "settings": {"base_dir": "output"},
    "tasks": [
        {"type": "pickup", "well": "A1",
         "check": {"type": "tip", "expected_tips": 8, "conf": 0.6}},
        {"type": "transfer", "source_slot": "8", "source_well": "A1",
         "dest_slot": "10", "dest_well": "A1", "volume": 100,
         "check": {"type": "liquid", "expected_tips": 8, "conf": 0.6}},
        {"type": "drop", "well": "A1"},
    ],
}

executor = ProtocolExecutor(client=client, model=model, camera=camera)
executor.execute(config)
```

### Web Controller

```bash
# UI-only mode (no robot)
python -m webapp

# With deck config
python -m webapp --config .config/deck.yaml

# With robot connection
python -m webapp --config .config/deck.yaml --robot 169.254.8.56
```

Or programmatically:

```python
from webapp import WebApp

app = WebApp.from_yaml(".config/deck.yaml", robot_ip="169.254.8.56")
app.register_handler("photograph", my_custom_handler)  # add custom step types
app.run(port=8000)
```

Deck config YAML format:

```yaml
pipettes:
  left: p300_single_gen2
  right: p300_multi_gen2
labware:
  "1": corning_96_wellplate_360ul_flat
  "7": nest_12_reservoir_15ml
  "10": opentrons_96_filtertiprack_200ul
active_pipette: left
```

#### Web UI Pages

**Dashboard** (`/`) — Run management
- Statistics cards (total, running, completed, failed)
- Card-based run list with status badges and progress bars
- "New Run" modal with step-by-step configuration
- Auto-refreshes every 5 seconds

**Create Task** (`/generate`) — 5-step wizard to build custom tasks using any LLM
- Step 1: **Connection** — verify OT-2 is online
- Step 2: **Create Task** — describe experiment, generate a comprehensive prompt (full API reference, labware catalog, workflow patterns), copy to LLM, paste back the output (deck YAML + steps JSON + optional template)
- Step 3: **Verify & Save** — YAML/JSON syntax validation, deck layout preview, step timeline preview, save to `.config/tasks/{timestamp}_{name}/`
- Step 4: **Calibration** — auto-reads saved task's deck config, lists labware in protocol execution order (same-slot actions grouped), inline XYZ nudge controller with log-scale step size (0.1-50mm)
- Step 5: **Create Run** — apply saved task and redirect to live run page

**Saved Tasks** (`/tasks`) — Browse and re-run previously saved tasks
- Card list sorted by timestamp (newest first)
- Expand to see deck grid visualization, pipette info, step timeline
- Simulate button: animated step-through preview
- "Run This Task" to create a new run instantly
- Delete tasks

**Live Run Monitor** (`/runs/{id}`) — Real-time execution dashboard
- Progress bar with ETA countdown and elapsed timer
- Control buttons (Start/Pause/Resume/Abort) with auto-enable/disable
- Step timeline with expand/collapse for params and output
- Chart.js bar chart showing step durations in real-time
- Event log with color-coded entries and filters (All / Errors / Steps)

**Hardware** (`/hardware`) — Device setup and diagnostics
- Quick precheck: one-click robot + camera scan
- Robot connection test (auto-detect or manual IP)
- Camera scan & preview: auto-captures from all USB cameras, user selects the plate camera
- Link to calibration tool

**Calibration** (`/calibration`) — Pipette offset tuning
- Card-based target layout with XYZ nudge controls
- Test aspirate/dispense actions
- Profile save/load/create

## LLM Protocol Generation

Generate protocols from natural language using any OpenAI-compatible API:

```python
from protocol import ProtocolGenerator

generator = ProtocolGenerator(api_key="sk-...", model="gpt-4o")
config, dry_run = generator.generate_and_validate(
    "Transfer 100uL from reservoir in slot 8 to a 96-well plate in slot 10"
)
if dry_run.success:
    executor.execute(config)
```

Or access the prompt template for use with any LLM:

```python
from protocol import get_protocol_prompt
print(get_protocol_prompt())
```

## Vision System

Real-time verification by capturing images and running object detection:

- **TipAnalyzer** — Checks tip presence after pickup by analyzing spatial positions
- **LiquidAnalyzer** — Verifies liquid levels against calibration curves
- **Calibration** — Build volume-to-height mapping from CSV data

```python
from vision import build_calibration_from_csv

calibration_fn = build_calibration_from_csv("LLD_Calibration_Data.csv", degree=3)
```

### Custom Vision Models

```python
from vision import VisionModel, PredictionResult

class MyModel(VisionModel):
    def predict(self, image_path: str, conf: float = 0.4) -> PredictionResult:
        ...

    @property
    def class_names(self) -> list[str]:
        return ["Tip", "Liquid"]
```

## Error Recovery

Define custom recovery plans using `on_fail` in any task, or use built-in defaults. Recovery plans use `ctx.*` placeholders that resolve to runtime values automatically.

```python
{
    "type": "transfer",
    "source_slot": "8", "source_well": "A1",
    "dest_slot": "10", "dest_well": "A1", "volume": 100,
    "check": {"type": "liquid", "expected_tips": 8, "conf": 0.6},
    "on_fail": [
        {"type": "dispense", "labware_id": "ctx.source_labware_id",
         "well": "ctx.source_well", "volume": "ctx.volume"},
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
