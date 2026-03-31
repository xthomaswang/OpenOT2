"""LLM-powered protocol auto-generator with validation and dry-run."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, ValidationError

logger = logging.getLogger("openot2.protocol.generator")


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_PROTOCOL_FORMAT_PROMPT = """\
You are a protocol generator for the OpenOT2 framework, which controls Opentrons OT-2 \
liquid handling robots via HTTP API.

Your task: given a user's natural language experiment plan, generate a valid Python \
dictionary (JSON-compatible) that the OpenOT2 ProtocolExecutor can run.

## Config Format

```python
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
```

## Common Labware Names
- Pipettes: "p20_single_gen2", "p300_single_gen2", "p1000_single_gen2", \
"p20_multi_gen2", "p300_multi_gen2"
- Tipracks: "opentrons_96_tiprack_20ul", "opentrons_96_tiprack_300ul", \
"opentrons_96_filtertiprack_200ul"
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
"""


def get_protocol_prompt() -> str:
    """Return the prompt template describing the protocol config format.

    Use this to manually paste into any LLM web interface (ChatGPT, Gemini,
    Claude, etc.) along with your experiment plan. Then paste the generated
    config back into Python to run.

    Example::

        prompt = get_protocol_prompt()
        print(prompt)
        # Copy the prompt + your plan into ChatGPT
        # Paste the result as:
        # config = { ... }
        # executor.execute(config)
    """
    return _PROTOCOL_FORMAT_PROMPT


# ---------------------------------------------------------------------------
# Pydantic validation schemas (internal)
# ---------------------------------------------------------------------------

class _VisionCheck(BaseModel):
    type: str
    expected_tips: int = 8
    conf: float = 0.6

class _Task(BaseModel):
    type: str
    well: Optional[str] = None
    source_slot: Optional[str] = None
    source_well: Optional[str] = None
    dest_slot: Optional[str] = None
    dest_well: Optional[str] = None
    volume: Optional[float] = None
    origin: str = "bottom"
    offset: Optional[Any] = None
    check: Optional[_VisionCheck] = None
    on_fail: Optional[List[dict]] = None

class _PipetteConfig(BaseModel):
    name: str
    mount: str = "right"

class _TiprackConfig(BaseModel):
    name: str
    slot: str

class _LabwareConfig(BaseModel):
    pipette: _PipetteConfig
    tiprack: _TiprackConfig
    sources: Optional[dict] = None
    imaging: Optional[dict] = None
    dispense: Optional[dict] = None

class _ProtocolSchema(BaseModel):
    labware: _LabwareConfig
    settings: dict
    tasks: List[_Task]


# ---------------------------------------------------------------------------
# Dry-run simulation
# ---------------------------------------------------------------------------

@dataclass
class DryRunResult:
    """Result of a protocol dry-run simulation."""

    success: bool
    steps_executed: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class _MockClient:
    """Minimal OT-2 client simulator for dry-run validation."""

    def __init__(self) -> None:
        self.has_tip = False
        self.current_volume = 0.0
        self.steps: List[str] = []
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.loaded_labware: Dict[str, str] = {}

    def load_pipette(self, name: str, mount: str = "right") -> None:
        self.steps.append(f"Load pipette: {name} ({mount})")

    def load_labware(self, name: str, slot: str) -> None:
        if slot in self.loaded_labware:
            self.errors.append(f"Slot {slot} already occupied by {self.loaded_labware[slot]}")
        self.loaded_labware[slot] = name
        self.steps.append(f"Load labware: {name} in slot {slot}")

    def pick_up_tip(self, well: str) -> None:
        if self.has_tip:
            self.errors.append(f"Pickup at {well}: already holding a tip")
        self.has_tip = True
        self.steps.append(f"Pick up tip: {well}")

    def aspirate(self, volume: float, source_slot: str, source_well: str) -> None:
        if not self.has_tip:
            self.errors.append(f"Aspirate {volume}uL: no tip mounted")
        self.current_volume += volume
        self.steps.append(f"Aspirate {volume}uL from slot {source_slot} well {source_well}")

    def dispense(self, volume: float, dest_slot: str, dest_well: str) -> None:
        if not self.has_tip:
            self.errors.append(f"Dispense {volume}uL: no tip mounted")
        if self.current_volume < volume:
            self.warnings.append(
                f"Dispense {volume}uL but only {self.current_volume}uL aspirated"
            )
        self.current_volume = max(0, self.current_volume - volume)
        self.steps.append(f"Dispense {volume}uL to slot {dest_slot} well {dest_well}")

    def drop_tip(self, well: str) -> None:
        if not self.has_tip:
            self.errors.append(f"Drop tip at {well}: no tip to drop")
        self.has_tip = False
        self.current_volume = 0.0
        self.steps.append(f"Drop tip: {well}")


def _dry_run_protocol(config: Dict[str, Any]) -> DryRunResult:
    """Simulate protocol execution and check logical correctness."""
    mock = _MockClient()

    try:
        labware = config.get("labware", {})

        # Load equipment
        pip = labware.get("pipette", {})
        mock.load_pipette(pip.get("name", "unknown"), pip.get("mount", "right"))

        tiprack = labware.get("tiprack", {})
        mock.load_labware(tiprack.get("name", "unknown"), tiprack.get("slot", "?"))

        sources = labware.get("sources", {})
        for slot in sources.get("slots", []):
            mock.load_labware(sources.get("name", "unknown"), slot)

        dispense = labware.get("dispense", {})
        for slot in dispense.get("slots", [dispense.get("slot", "")]):
            if slot:
                mock.load_labware(dispense.get("name", "unknown"), slot)

        imaging = labware.get("imaging", {})
        if imaging.get("slot"):
            mock.load_labware(imaging.get("name", "unknown"), imaging["slot"])

        # Execute tasks
        for task in config.get("tasks", []):
            t = task.get("type")
            if t == "pickup":
                mock.pick_up_tip(task.get("well", "A1"))
            elif t == "transfer":
                mock.aspirate(
                    task.get("volume", 0),
                    task.get("source_slot", "?"),
                    task.get("source_well", "?"),
                )
                mock.dispense(
                    task.get("volume", 0),
                    task.get("dest_slot", "?"),
                    task.get("dest_well", "?"),
                )
            elif t == "drop":
                mock.drop_tip(task.get("well", "A1"))

        # Final check
        if mock.has_tip:
            mock.warnings.append("Protocol ended with tip still mounted.")

    except Exception as exc:
        mock.errors.append(f"Simulation crashed: {exc}")

    return DryRunResult(
        success=len(mock.errors) == 0,
        steps_executed=mock.steps,
        errors=mock.errors,
        warnings=mock.warnings,
    )


# ---------------------------------------------------------------------------
# ProtocolGenerator
# ---------------------------------------------------------------------------

class ProtocolGenerator:
    """LLM-powered protocol generator using OpenAI-compatible API.

    Supports any provider via the ``base_url`` parameter:

    - OpenAI: ``ProtocolGenerator(api_key="sk-...", model="gpt-4o")``
    - Anthropic (via proxy): ``ProtocolGenerator(api_key="...", base_url="https://...", model="claude-...")``
    - Local (Ollama etc.): ``ProtocolGenerator(base_url="http://localhost:11434/v1", model="llama3")``

    Args:
        api_key: API key (can also be set via ``OPENAI_API_KEY`` env var).
        base_url: Base URL override for non-OpenAI providers.
        model: Model name (default ``"gpt-4o"``).

    Raises:
        ImportError: If ``openai`` is not installed.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "gpt-4o",
    ) -> None:
        from openai import OpenAI

        kwargs: Dict[str, Any] = {}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url

        self._client = OpenAI(**kwargs)
        self._model = model

    def generate(
        self,
        plan: str,
        labware_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Generate a protocol config from a natural language plan.

        Args:
            plan: Natural language experiment description.
            labware_context: Optional dict describing available labware on deck.

        Returns:
            Protocol config dict ready for :meth:`ProtocolExecutor.execute`.

        Raises:
            ValueError: If the LLM response cannot be parsed as JSON.
        """
        messages = [
            {"role": "system", "content": _PROTOCOL_FORMAT_PROMPT},
        ]

        user_content = plan
        if labware_context:
            user_content += f"\n\nAvailable labware context:\n{json.dumps(labware_context, indent=2)}"

        messages.append({"role": "user", "content": user_content})

        logger.info("Generating protocol from plan...")
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=0.1,
        )

        raw = response.choices[0].message.content.strip()

        # Strip markdown code fences if present
        if raw.startswith("```"):
            lines = raw.split("\n")
            lines = [l for l in lines if not l.startswith("```")]
            raw = "\n".join(lines)

        try:
            config = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Failed to parse LLM response as JSON: {exc}\nRaw: {raw}") from exc

        logger.info("Protocol generated successfully.")
        return config

    @staticmethod
    def validate(config: Dict[str, Any]) -> List[str]:
        """Validate a protocol config against the schema.

        Returns:
            List of error messages (empty if valid).
        """
        errors: List[str] = []
        try:
            _ProtocolSchema.model_validate(config)
        except ValidationError as exc:
            for err in exc.errors():
                loc = " -> ".join(str(l) for l in err["loc"])
                errors.append(f"{loc}: {err['msg']}")
        return errors

    @staticmethod
    def dry_run(config: Dict[str, Any]) -> DryRunResult:
        """Simulate protocol execution to check logical correctness.

        Checks for issues like aspirating without a tip, dispensing more
        than aspirated, double-picking tips, etc.

        Returns:
            :class:`DryRunResult` with steps, errors, and warnings.
        """
        return _dry_run_protocol(config)

    def generate_and_validate(
        self,
        plan: str,
        labware_context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], DryRunResult]:
        """Generate, validate, and dry-run a protocol in one call.

        Returns:
            Tuple of ``(config, dry_run_result)``.

        Raises:
            ValueError: If generation or schema validation fails.
        """
        config = self.generate(plan, labware_context)

        # Schema validation
        schema_errors = self.validate(config)
        if schema_errors:
            raise ValueError(f"Generated config has schema errors: {schema_errors}")

        # Dry run
        result = self.dry_run(config)

        if result.errors:
            logger.warning("Dry-run found errors: %s", result.errors)
        if result.warnings:
            logger.warning("Dry-run warnings: %s", result.warnings)

        return config, result
