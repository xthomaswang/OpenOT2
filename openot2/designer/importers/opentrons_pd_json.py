"""Import Opentrons Protocol Designer JSON files into OpenOT2 IR.

Supports Protocol Designer JSON schema versions 6, 7, and 8.

* **v7/v8** — The modern command-based format where protocol steps are
  represented as a ``commands`` array with ``commandType`` entries.
* **v6** — The legacy step-form format using ``orderedStepIds`` +
  ``savedStepForms``.

Usage::

    from openot2.designer.importers import import_pd_json

    result = import_pd_json("protocol.json")
    protocol_ir = result.protocol  # ProtocolIR
    print(result.warnings)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Union

from openot2.designer.ir import (
    DeckSetup,
    Edge,
    LabwareEntry,
    LiquidEntry,
    ModuleEntry,
    Node,
    NodeKind,
    PipetteEntry,
    ProtocolIR,
)


# ---------------------------------------------------------------------------
# Import result
# ---------------------------------------------------------------------------

@dataclass
class ImportResult:
    """Container for the import outcome."""

    protocol: ProtocolIR
    source_format: str = "opentrons_pd_json"
    source_version: str | None = None
    unsupported_features: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def import_pd_json(source: Union[str, Path, dict]) -> ImportResult:
    """Import an Opentrons Protocol Designer JSON file.

    Parameters
    ----------
    source:
        A file path (``str`` or ``Path``) to a PD JSON file, or a
        pre-parsed ``dict``.

    Returns
    -------
    ImportResult
        Contains the :class:`ProtocolIR` plus import metadata.

    Raises
    ------
    ValueError
        If the JSON cannot be parsed or the schema version is unrecognised.
    """
    if isinstance(source, dict):
        data = source
    else:
        path = Path(source)
        with open(path) as f:
            data = json.load(f)

    schema_version = data.get("schemaVersion")
    if schema_version is None:
        raise ValueError(
            "Missing 'schemaVersion' — is this an Opentrons Protocol "
            "Designer JSON file?"
        )

    result = ImportResult(
        protocol=ProtocolIR(name="Untitled"),
        source_version=str(schema_version),
    )

    if schema_version >= 7:
        _import_v7(data, result)
    elif schema_version >= 6:
        _import_v6(data, result)
    else:
        raise ValueError(
            f"Unsupported PD JSON schema version: {schema_version}. "
            f"Versions 6-8 are supported."
        )

    return result


# ---------------------------------------------------------------------------
# Metadata extraction (shared across versions)
# ---------------------------------------------------------------------------

def _extract_metadata(data: dict, result: ImportResult) -> None:
    """Extract protocol metadata from the top-level object."""
    md = data.get("metadata", {})
    protocol = result.protocol

    protocol.name = md.get("protocolName") or md.get("name") or "Untitled"
    protocol.description = md.get("description", "")
    protocol.metadata = {
        "author": md.get("author", ""),
        "created": md.get("created"),
        "last_modified": md.get("lastModified"),
        "source_format": result.source_format,
        "source_version": result.source_version,
    }

    robot = data.get("robot", {})
    if robot:
        protocol.metadata["robot_model"] = robot.get("model", "")
        protocol.metadata["deck_id"] = robot.get("deckId", "")


# ---------------------------------------------------------------------------
# Deck extraction (shared across versions)
# ---------------------------------------------------------------------------

def _extract_deck(data: dict, result: ImportResult) -> None:
    """Extract labware, pipettes, modules, and liquids into the DeckSetup."""
    deck = result.protocol.deck

    # Pipettes
    pipettes_data = data.get("pipettes", {})
    for pid, pinfo in pipettes_data.items():
        mount = pinfo.get("mount", "")
        pipette_type = pinfo.get("name", "")
        deck.pipettes.append(PipetteEntry(
            id=pid,
            mount=mount,
            pipette_type=pipette_type,
        ))

    # Labware
    labware_data = data.get("labware", {})
    for lid, linfo in labware_data.items():
        slot = linfo.get("slot", "")
        # definition_id may be a string key or embedded in labware entry
        definition_id = linfo.get("definitionId", "")
        display_name = linfo.get("displayName", "")
        # Resolve labware type from definition if available
        labware_type = definition_id
        definitions = data.get("labwareDefinitions", {})
        if definition_id in definitions:
            defn = definitions[definition_id]
            metadata = defn.get("metadata", {})
            labware_type = metadata.get("displayName", definition_id)

        deck.labware.append(LabwareEntry(
            id=lid,
            slot=str(slot),
            labware_type=labware_type,
            display_name=display_name or labware_type,
        ))

    # Modules
    modules_data = data.get("modules", {})
    for mid, minfo in modules_data.items():
        deck.modules.append(ModuleEntry(
            id=mid,
            slot=str(minfo.get("slot", "")),
            module_type=minfo.get("model", ""),
            display_name=minfo.get("model", ""),
        ))

    # Liquids
    liquids_data = data.get("liquids", {})
    for liq_id, liq_info in liquids_data.items():
        deck.liquids.append(LiquidEntry(
            id=liq_id,
            name=liq_info.get("displayName", ""),
            color=liq_info.get("displayColor"),
            description=liq_info.get("description", ""),
        ))


# ---------------------------------------------------------------------------
# v7/v8 command-based import
# ---------------------------------------------------------------------------

# Maps PD commandType to OpenOT2 NodeKind
_COMMAND_KIND_MAP: dict[str, NodeKind] = {
    "aspirate": NodeKind.transfer,
    "dispense": NodeKind.transfer,
    "blowout": NodeKind.transfer,
    "touchTip": NodeKind.transfer,
    "pickUpTip": NodeKind.transfer,
    "dropTip": NodeKind.transfer,
    "moveToWell": NodeKind.transfer,
    "waitForResume": NodeKind.pause,
    "waitForDuration": NodeKind.delay,
    "moveLabware": NodeKind.move_labware,
    # Module commands
    "temperatureModule/setTargetTemperature": NodeKind.module_action,
    "temperatureModule/deactivate": NodeKind.module_action,
    "thermocycler/setTargetBlockTemperature": NodeKind.module_action,
    "thermocycler/setTargetLidTemperature": NodeKind.module_action,
    "thermocycler/openLid": NodeKind.module_action,
    "thermocycler/closeLid": NodeKind.module_action,
    "thermocycler/runProfile": NodeKind.module_action,
    "thermocycler/deactivateBlock": NodeKind.module_action,
    "thermocycler/deactivateLid": NodeKind.module_action,
    "heaterShaker/setTargetTemperature": NodeKind.module_action,
    "heaterShaker/setAndWaitForShakeSpeed": NodeKind.module_action,
    "heaterShaker/deactivateHeater": NodeKind.module_action,
    "heaterShaker/deactivateShaker": NodeKind.module_action,
    "heaterShaker/openLabwareLatch": NodeKind.module_action,
    "heaterShaker/closeLabwareLatch": NodeKind.module_action,
    "magneticModule/engage": NodeKind.module_action,
    "magneticModule/disengage": NodeKind.module_action,
}

# Transfer sub-operations are grouped into a single transfer node at a higher
# level.  These are the raw atomic commands that form a transfer.
_TRANSFER_ATOMIC_COMMANDS = frozenset({
    "aspirate", "dispense", "blowout", "touchTip",
    "pickUpTip", "dropTip", "moveToWell",
})


def _import_v7(data: dict, result: ImportResult) -> None:
    """Import a v7/v8 command-based PD JSON."""
    _extract_metadata(data, result)
    _extract_deck(data, result)

    # We process in two modes:
    # 1. If designerApplication.data has stepId groupings, use those to
    #    create higher-level transfer/mix nodes (preferred).
    # 2. Otherwise, fall back to one node per raw command.
    designer_data = data.get("designerApplication", {}).get("data", {})
    saved_step_forms = designer_data.get("savedStepForms", {})
    ordered_step_ids = designer_data.get("orderedStepIds", [])

    if saved_step_forms and ordered_step_ids:
        _import_v7_with_step_forms(
            data, saved_step_forms, ordered_step_ids, result,
        )
    else:
        commands = data.get("commands", [])
        if not commands:
            result.warnings.append("Protocol contains no commands")
            return
        _import_v7_raw_commands(commands, result)


def _import_v7_with_step_forms(
    data: dict,
    saved_step_forms: dict[str, Any],
    ordered_step_ids: list[str],
    result: ImportResult,
) -> None:
    """Import v7/v8 using designerApplication step forms for grouping."""
    for step_id in ordered_step_ids:
        if step_id == "__INITIAL_DECK_SETUP_STEP__":
            continue

        form = saved_step_forms.get(step_id)
        if form is None:
            result.warnings.append(f"Step '{step_id}' referenced but not found in savedStepForms")
            continue

        step_type = form.get("stepType", "")
        node = _step_form_to_node(step_id, step_type, form, result)
        if node is not None:
            result.protocol.nodes.append(node)


def _step_form_to_node(
    step_id: str,
    step_type: str,
    form: dict[str, Any],
    result: ImportResult,
) -> Node | None:
    """Convert a saved step form into an IR Node."""
    if step_type == "moveLiquid":
        return _step_form_transfer(step_id, form, result)
    elif step_type == "mix":
        return _step_form_mix(step_id, form, result)
    elif step_type == "pause":
        return _step_form_pause(step_id, form, result)
    elif step_type == "temperature":
        return _step_form_module_action(step_id, form, "temperature", result)
    elif step_type == "magnet":
        return _step_form_module_action(step_id, form, "magnet", result)
    elif step_type == "thermocycler":
        return _step_form_module_action(step_id, form, "thermocycler", result)
    elif step_type == "heaterShaker":
        return _step_form_module_action(step_id, form, "heaterShaker", result)
    elif step_type == "moveLabware":
        return _step_form_move_labware(step_id, form, result)
    elif step_type == "comment":
        # Comments don't produce nodes but we can store them as metadata
        msg = form.get("message", "")
        result.protocol.metadata.setdefault("comments", []).append(msg)
        return None
    else:
        result.warnings.append(f"Unknown step type '{step_type}' (step {step_id})")
        result.unsupported_features.append(step_type)
        return None


def _step_form_transfer(
    step_id: str, form: dict, result: ImportResult,
) -> Node:
    """Convert a moveLiquid step form into a transfer Node."""
    params: dict[str, Any] = {
        "pipette_id": form.get("pipette", ""),
        "source_labware_id": form.get("aspirate_labware", ""),
        "source_wells": _parse_wells(form.get("aspirate_wells", "")),
        "dest_labware_id": form.get("dispense_labware", ""),
        "dest_wells": _parse_wells(form.get("dispense_wells", "")),
        "volume": _safe_float(form.get("volume", 0)),
    }

    # Optional parameters
    if form.get("aspirate_flowRate"):
        params["aspirate_flow_rate"] = _safe_float(form["aspirate_flowRate"])
    if form.get("dispense_flowRate"):
        params["dispense_flow_rate"] = _safe_float(form["dispense_flowRate"])
    if form.get("aspirate_mix_before_checkbox"):
        params["mix_before"] = {
            "volume": _safe_float(form.get("aspirate_mix_before_volume", 0)),
            "cycles": _safe_int(form.get("aspirate_mix_before_times", 0)),
        }
    if form.get("dispense_mix_after_checkbox"):
        params["mix_after"] = {
            "volume": _safe_float(form.get("dispense_mix_after_volume", 0)),
            "cycles": _safe_int(form.get("dispense_mix_after_times", 0)),
        }
    if form.get("disposalVolume_checkbox"):
        params["disposal_volume"] = _safe_float(form.get("disposalVolume_volume", 0))
    if form.get("blowout_checkbox"):
        params["blowout_location"] = form.get("blowout_location", "")

    return Node(
        id=step_id,
        kind=NodeKind.transfer,
        label=form.get("stepName", "") or "Transfer",
        params=params,
    )


def _step_form_mix(
    step_id: str, form: dict, result: ImportResult,
) -> Node:
    """Convert a mix step form into a mix Node."""
    return Node(
        id=step_id,
        kind=NodeKind.mix,
        label=form.get("stepName", "") or "Mix",
        params={
            "pipette_id": form.get("pipette", ""),
            "labware_id": form.get("labware", ""),
            "wells": _parse_wells(form.get("wells", "")),
            "volume": _safe_float(form.get("volume", 0)),
            "cycles": _safe_int(form.get("times", 1)),
        },
    )


def _step_form_pause(
    step_id: str, form: dict, result: ImportResult,
) -> Node:
    """Convert a pause step form into a pause or delay Node."""
    pause_type = form.get("pauseAction", "")

    if pause_type == "untilTime":
        # Timed delay
        hours = _safe_int(form.get("pauseHour", 0))
        minutes = _safe_int(form.get("pauseMinute", 0))
        seconds = _safe_int(form.get("pauseSecond", 0))
        total_seconds = hours * 3600 + minutes * 60 + seconds
        return Node(
            id=step_id,
            kind=NodeKind.delay,
            label=form.get("stepName", "") or "Delay",
            params={"seconds": total_seconds},
        )
    else:
        # Manual pause (untilResume or default)
        return Node(
            id=step_id,
            kind=NodeKind.pause,
            label=form.get("stepName", "") or "Pause",
            params={"message": form.get("pauseMessage", "")},
        )


def _step_form_module_action(
    step_id: str, form: dict, module_type: str, result: ImportResult,
) -> Node:
    """Convert a module step form into a module_action Node."""
    params: dict[str, Any] = {
        "module_id": form.get("moduleId", ""),
        "action": form.get("moduleAction", module_type),
    }

    # Temperature-related params
    if form.get("setTemperature"):
        params["target_temperature"] = _safe_float(
            form.get("targetTemperature", 0)
        )

    # Magnet params
    if form.get("engageHeight"):
        params["engage_height"] = _safe_float(form["engageHeight"])

    # Thermocycler params
    if form.get("blockTargetTemp"):
        params["block_temperature"] = _safe_float(form["blockTargetTemp"])
    if form.get("lidTargetTemp"):
        params["lid_temperature"] = _safe_float(form["lidTargetTemp"])

    # Heater-shaker params
    if form.get("targetSpeed"):
        params["shake_speed"] = _safe_int(form["targetSpeed"])

    return Node(
        id=step_id,
        kind=NodeKind.module_action,
        label=form.get("stepName", "") or f"{module_type} action",
        params=params,
    )


def _step_form_move_labware(
    step_id: str, form: dict, result: ImportResult,
) -> Node:
    """Convert a moveLabware step form into a move_labware Node."""
    return Node(
        id=step_id,
        kind=NodeKind.move_labware,
        label=form.get("stepName", "") or "Move Labware",
        params={
            "labware_id": form.get("labware", ""),
            "new_slot": str(form.get("newLocation", "")),
        },
    )


def _import_v7_raw_commands(
    commands: list[dict], result: ImportResult,
) -> None:
    """Import v7/v8 as one IR node per command (no step grouping)."""
    for i, cmd in enumerate(commands):
        cmd_type = cmd.get("commandType", "")
        params = cmd.get("params", {})
        key = cmd.get("key", f"cmd_{i}")

        node = _command_to_node(key, cmd_type, params, i, result)
        if node is not None:
            result.protocol.nodes.append(node)


def _command_to_node(
    key: str,
    cmd_type: str,
    params: dict[str, Any],
    index: int,
    result: ImportResult,
) -> Node | None:
    """Convert a single PD command into an IR Node."""

    if cmd_type == "aspirate":
        return Node(
            id=key,
            kind=NodeKind.transfer,
            label=f"Aspirate {params.get('volume', '?')} uL",
            params={
                "pipette_id": params.get("pipetteId", ""),
                "source_labware_id": params.get("labwareId", ""),
                "source_wells": [params.get("wellName", "A1")],
                "dest_labware_id": "",
                "dest_wells": [],
                "volume": _safe_float(params.get("volume", 0)),
                "flow_rate": _safe_float(params.get("flowRate", 0)),
                "pd_command_type": "aspirate",
            },
        )

    elif cmd_type == "dispense":
        return Node(
            id=key,
            kind=NodeKind.transfer,
            label=f"Dispense {params.get('volume', '?')} uL",
            params={
                "pipette_id": params.get("pipetteId", ""),
                "source_labware_id": "",
                "source_wells": [],
                "dest_labware_id": params.get("labwareId", ""),
                "dest_wells": [params.get("wellName", "A1")],
                "volume": _safe_float(params.get("volume", 0)),
                "flow_rate": _safe_float(params.get("flowRate", 0)),
                "pd_command_type": "dispense",
            },
        )

    elif cmd_type in ("pickUpTip", "dropTip"):
        return Node(
            id=key,
            kind=NodeKind.transfer,
            label=cmd_type,
            params={
                "pipette_id": params.get("pipetteId", ""),
                "source_labware_id": params.get("labwareId", ""),
                "source_wells": [params.get("wellName", "A1")] if params.get("wellName") else [],
                "dest_labware_id": "",
                "dest_wells": [],
                "volume": 0,
                "pd_command_type": cmd_type,
            },
        )

    elif cmd_type in ("blowout", "touchTip", "moveToWell"):
        return Node(
            id=key,
            kind=NodeKind.transfer,
            label=cmd_type,
            params={
                "pipette_id": params.get("pipetteId", ""),
                "source_labware_id": params.get("labwareId", ""),
                "source_wells": [params.get("wellName", "A1")] if params.get("wellName") else [],
                "dest_labware_id": "",
                "dest_wells": [],
                "volume": 0,
                "pd_command_type": cmd_type,
            },
        )

    elif cmd_type == "waitForResume":
        return Node(
            id=key,
            kind=NodeKind.pause,
            label="Pause",
            params={"message": params.get("message", "")},
        )

    elif cmd_type == "waitForDuration":
        return Node(
            id=key,
            kind=NodeKind.delay,
            label=f"Delay {params.get('seconds', 0)}s",
            params={"seconds": _safe_float(params.get("seconds", 0))},
        )

    elif cmd_type == "moveLabware":
        return Node(
            id=key,
            kind=NodeKind.move_labware,
            label="Move Labware",
            params={
                "labware_id": params.get("labwareId", ""),
                "new_slot": str(params.get("newLocation", "")),
            },
        )

    elif cmd_type == "comment":
        # Store as metadata comment, not a node
        msg = params.get("message", "")
        result.protocol.metadata.setdefault("comments", []).append(msg)
        return None

    elif "/" in cmd_type:
        # Module command like "temperatureModule/setTargetTemperature"
        parts = cmd_type.split("/", 1)
        return Node(
            id=key,
            kind=NodeKind.module_action,
            label=cmd_type,
            params={
                "module_id": params.get("moduleId", ""),
                "action": parts[1] if len(parts) > 1 else cmd_type,
                **{k: v for k, v in params.items() if k != "moduleId"},
            },
        )

    else:
        result.warnings.append(
            f"Unknown command type '{cmd_type}' at index {index}"
        )
        result.unsupported_features.append(cmd_type)
        return None


# ---------------------------------------------------------------------------
# v6 step-form-only import
# ---------------------------------------------------------------------------

def _import_v6(data: dict, result: ImportResult) -> None:
    """Import a v6 step-form-based PD JSON."""
    _extract_metadata(data, result)
    _extract_deck(data, result)

    designer_data = data.get("designerApplication", {}).get("data", {})
    saved_step_forms = designer_data.get("savedStepForms", {})
    ordered_step_ids = designer_data.get("orderedStepIds", [])

    if not ordered_step_ids:
        result.warnings.append("Protocol contains no steps (orderedStepIds empty)")
        return

    for step_id in ordered_step_ids:
        if step_id == "__INITIAL_DECK_SETUP_STEP__":
            continue

        form = saved_step_forms.get(step_id)
        if form is None:
            result.warnings.append(
                f"Step '{step_id}' referenced but not found in savedStepForms"
            )
            continue

        step_type = form.get("stepType", "")
        node = _step_form_to_node(step_id, step_type, form, result)
        if node is not None:
            result.protocol.nodes.append(node)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_wells(value: Any) -> list[str]:
    """Parse well specification into a list of well names."""
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value:
        # Comma-separated or single well
        return [w.strip() for w in value.split(",") if w.strip()]
    return []


def _safe_float(value: Any) -> float:
    """Safely convert to float, defaulting to 0."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _safe_int(value: Any) -> int:
    """Safely convert to int, defaulting to 0."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
