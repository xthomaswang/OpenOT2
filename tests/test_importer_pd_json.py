"""Tests for the Opentrons Protocol Designer JSON importer."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import pytest

from openot2.designer.ir import NodeKind, ProtocolIR
from openot2.designer.importers.opentrons_pd_json import (
    ImportResult,
    import_pd_json,
)
from openot2.designer.importers.opentrons_pd_python import import_pd_python


# ---------------------------------------------------------------------------
# Fixtures: representative PD JSON structures
# ---------------------------------------------------------------------------

def _make_v8_protocol(**overrides: Any) -> dict:
    """Build a minimal v8 PD JSON with sensible defaults."""
    base: dict[str, Any] = {
        "schemaVersion": 8,
        "metadata": {
            "protocolName": "Test Protocol",
            "author": "Test Author",
            "description": "A test protocol",
            "created": 1700000000000,
            "lastModified": 1700000001000,
        },
        "robot": {
            "model": "OT-2 Standard",
            "deckId": "ot2_standard",
        },
        "pipettes": {
            "pip_left": {
                "name": "p300_single_gen2",
                "mount": "left",
            },
        },
        "labware": {
            "tiprack_1": {
                "slot": "10",
                "definitionId": "opentrons_96_tiprack_300ul",
                "displayName": "Tip Rack",
            },
            "plate_1": {
                "slot": "1",
                "definitionId": "corning_96_wellplate_360ul_flat",
                "displayName": "Source Plate",
            },
            "plate_2": {
                "slot": "2",
                "definitionId": "corning_96_wellplate_360ul_flat",
                "displayName": "Dest Plate",
            },
        },
        "labwareDefinitions": {
            "opentrons_96_tiprack_300ul": {
                "metadata": {"displayName": "Opentrons 96 Tip Rack 300 uL"},
            },
            "corning_96_wellplate_360ul_flat": {
                "metadata": {"displayName": "Corning 96 Well Plate 360 uL"},
            },
        },
        "modules": {},
        "liquids": {
            "liq_1": {
                "displayName": "Water",
                "displayColor": "#0000ff",
                "description": "DI Water",
            },
        },
        "commands": [],
    }
    base.update(overrides)
    return base


def _make_v8_with_commands() -> dict:
    """v8 protocol with raw commands (no step forms)."""
    proto = _make_v8_protocol()
    proto["commands"] = [
        {
            "commandType": "pickUpTip",
            "key": "cmd_0",
            "params": {
                "pipetteId": "pip_left",
                "labwareId": "tiprack_1",
                "wellName": "A1",
            },
        },
        {
            "commandType": "aspirate",
            "key": "cmd_1",
            "params": {
                "pipetteId": "pip_left",
                "labwareId": "plate_1",
                "wellName": "A1",
                "volume": 100,
                "flowRate": 7.56,
            },
        },
        {
            "commandType": "dispense",
            "key": "cmd_2",
            "params": {
                "pipetteId": "pip_left",
                "labwareId": "plate_2",
                "wellName": "A1",
                "volume": 100,
                "flowRate": 7.56,
            },
        },
        {
            "commandType": "dropTip",
            "key": "cmd_3",
            "params": {
                "pipetteId": "pip_left",
                "labwareId": "tiprack_1",
                "wellName": "A1",
            },
        },
    ]
    return proto


def _make_v8_with_step_forms() -> dict:
    """v8 protocol with designerApplication step forms."""
    proto = _make_v8_protocol()
    proto["designerApplication"] = {
        "name": "opentrons/protocol-designer",
        "version": "8.0.0",
        "data": {
            "orderedStepIds": [
                "__INITIAL_DECK_SETUP_STEP__",
                "step_transfer_1",
                "step_mix_1",
                "step_pause_1",
                "step_delay_1",
            ],
            "savedStepForms": {
                "__INITIAL_DECK_SETUP_STEP__": {"stepType": "manualIntervention"},
                "step_transfer_1": {
                    "stepType": "moveLiquid",
                    "stepName": "Transfer Water",
                    "pipette": "pip_left",
                    "aspirate_labware": "plate_1",
                    "aspirate_wells": "A1,A2,A3",
                    "dispense_labware": "plate_2",
                    "dispense_wells": "B1,B2,B3",
                    "volume": "100",
                    "aspirate_flowRate": "7.56",
                    "dispense_flowRate": "7.56",
                },
                "step_mix_1": {
                    "stepType": "mix",
                    "stepName": "Mix Sample",
                    "pipette": "pip_left",
                    "labware": "plate_2",
                    "wells": "B1,B2,B3",
                    "volume": "50",
                    "times": "5",
                },
                "step_pause_1": {
                    "stepType": "pause",
                    "stepName": "Wait for user",
                    "pauseAction": "untilResume",
                    "pauseMessage": "Check the plate",
                },
                "step_delay_1": {
                    "stepType": "pause",
                    "stepName": "Incubate",
                    "pauseAction": "untilTime",
                    "pauseHour": "0",
                    "pauseMinute": "5",
                    "pauseSecond": "30",
                },
            },
        },
    }
    return proto


def _make_v6_protocol() -> dict:
    """Build a minimal v6 PD JSON."""
    return {
        "schemaVersion": 6,
        "metadata": {
            "protocolName": "V6 Protocol",
            "author": "Test",
            "description": "Legacy format test",
        },
        "robot": {"model": "OT-2 Standard"},
        "pipettes": {
            "pip_right": {
                "name": "p20_single_gen2",
                "mount": "right",
            },
        },
        "labware": {
            "well_plate": {
                "slot": "3",
                "definitionId": "corning_96_wellplate_360ul_flat",
                "displayName": "Well Plate",
            },
        },
        "labwareDefinitions": {
            "corning_96_wellplate_360ul_flat": {
                "metadata": {"displayName": "Corning 96 Well Plate 360 uL"},
            },
        },
        "modules": {},
        "liquids": {},
        "designerApplication": {
            "name": "opentrons/protocol-designer",
            "version": "6.0.0",
            "data": {
                "orderedStepIds": [
                    "__INITIAL_DECK_SETUP_STEP__",
                    "step_1",
                ],
                "savedStepForms": {
                    "__INITIAL_DECK_SETUP_STEP__": {"stepType": "manualIntervention"},
                    "step_1": {
                        "stepType": "moveLiquid",
                        "stepName": "Simple Transfer",
                        "pipette": "pip_right",
                        "aspirate_labware": "well_plate",
                        "aspirate_wells": "A1",
                        "dispense_labware": "well_plate",
                        "dispense_wells": "B1",
                        "volume": "10",
                    },
                },
            },
        },
    }


# ---------------------------------------------------------------------------
# Tests: basic import
# ---------------------------------------------------------------------------

class TestImportBasic:
    """Basic import functionality."""

    def test_import_from_dict(self):
        proto = _make_v8_with_commands()
        result = import_pd_json(proto)
        assert isinstance(result, ImportResult)
        assert isinstance(result.protocol, ProtocolIR)
        assert result.source_format == "opentrons_pd_json"
        assert result.source_version == "8"

    def test_import_from_file(self, tmp_path: Path):
        proto = _make_v8_with_commands()
        fpath = tmp_path / "protocol.json"
        fpath.write_text(json.dumps(proto))
        result = import_pd_json(fpath)
        assert result.protocol.name == "Test Protocol"

    def test_import_from_str_path(self, tmp_path: Path):
        proto = _make_v8_with_commands()
        fpath = tmp_path / "protocol.json"
        fpath.write_text(json.dumps(proto))
        result = import_pd_json(str(fpath))
        assert result.protocol.name == "Test Protocol"

    def test_missing_schema_version(self):
        with pytest.raises(ValueError, match="schemaVersion"):
            import_pd_json({"metadata": {}})

    def test_unsupported_schema_version(self):
        with pytest.raises(ValueError, match="Unsupported PD JSON schema version: 3"):
            import_pd_json({"schemaVersion": 3})


# ---------------------------------------------------------------------------
# Tests: metadata extraction
# ---------------------------------------------------------------------------

class TestMetadata:
    """Protocol metadata extraction."""

    def test_metadata_fields(self):
        result = import_pd_json(_make_v8_with_commands())
        p = result.protocol
        assert p.name == "Test Protocol"
        assert p.description == "A test protocol"
        assert p.metadata["author"] == "Test Author"
        assert p.metadata["robot_model"] == "OT-2 Standard"
        assert p.metadata["source_format"] == "opentrons_pd_json"

    def test_missing_metadata_uses_defaults(self):
        proto = _make_v8_protocol()
        proto["metadata"] = {}
        result = import_pd_json(proto)
        assert result.protocol.name == "Untitled"


# ---------------------------------------------------------------------------
# Tests: deck extraction
# ---------------------------------------------------------------------------

class TestDeck:
    """Deck setup: labware, pipettes, modules, liquids."""

    def test_pipettes(self):
        result = import_pd_json(_make_v8_with_commands())
        pipettes = result.protocol.deck.pipettes
        assert len(pipettes) == 1
        assert pipettes[0].id == "pip_left"
        assert pipettes[0].mount == "left"
        assert pipettes[0].pipette_type == "p300_single_gen2"

    def test_labware(self):
        result = import_pd_json(_make_v8_with_commands())
        labware = result.protocol.deck.labware
        assert len(labware) == 3
        ids = {lw.id for lw in labware}
        assert "tiprack_1" in ids
        assert "plate_1" in ids
        assert "plate_2" in ids

    def test_labware_resolves_display_name_from_definition(self):
        result = import_pd_json(_make_v8_with_commands())
        tiprack = next(lw for lw in result.protocol.deck.labware if lw.id == "tiprack_1")
        assert "Tip Rack" in tiprack.display_name or "tiprack" in tiprack.display_name.lower()

    def test_liquids(self):
        result = import_pd_json(_make_v8_with_commands())
        liquids = result.protocol.deck.liquids
        assert len(liquids) == 1
        assert liquids[0].id == "liq_1"
        assert liquids[0].name == "Water"
        assert liquids[0].color == "#0000ff"

    def test_modules(self):
        proto = _make_v8_protocol()
        proto["modules"] = {
            "temp_mod_1": {
                "slot": "3",
                "model": "temperatureModuleV2",
            },
        }
        result = import_pd_json(proto)
        modules = result.protocol.deck.modules
        assert len(modules) == 1
        assert modules[0].id == "temp_mod_1"
        assert modules[0].module_type == "temperatureModuleV2"
        assert modules[0].slot == "3"


# ---------------------------------------------------------------------------
# Tests: v7/v8 raw command import
# ---------------------------------------------------------------------------

class TestV8RawCommands:
    """Import v7/v8 protocols with raw commands (no step forms)."""

    def test_command_count(self):
        result = import_pd_json(_make_v8_with_commands())
        assert len(result.protocol.nodes) == 4

    def test_aspirate_command(self):
        result = import_pd_json(_make_v8_with_commands())
        node = result.protocol.nodes[1]  # aspirate
        assert node.kind == NodeKind.transfer
        assert node.params["volume"] == 100.0
        assert node.params["pd_command_type"] == "aspirate"

    def test_dispense_command(self):
        result = import_pd_json(_make_v8_with_commands())
        node = result.protocol.nodes[2]  # dispense
        assert node.kind == NodeKind.transfer
        assert node.params["volume"] == 100.0
        assert node.params["pd_command_type"] == "dispense"

    def test_pick_up_tip_command(self):
        result = import_pd_json(_make_v8_with_commands())
        node = result.protocol.nodes[0]
        assert node.kind == NodeKind.transfer
        assert node.params["pd_command_type"] == "pickUpTip"

    def test_drop_tip_command(self):
        result = import_pd_json(_make_v8_with_commands())
        node = result.protocol.nodes[3]
        assert node.kind == NodeKind.transfer
        assert node.params["pd_command_type"] == "dropTip"

    def test_pause_command(self):
        proto = _make_v8_protocol()
        proto["commands"] = [{
            "commandType": "waitForResume",
            "key": "cmd_pause",
            "params": {"message": "Check the plate"},
        }]
        result = import_pd_json(proto)
        assert len(result.protocol.nodes) == 1
        assert result.protocol.nodes[0].kind == NodeKind.pause
        assert result.protocol.nodes[0].params["message"] == "Check the plate"

    def test_delay_command(self):
        proto = _make_v8_protocol()
        proto["commands"] = [{
            "commandType": "waitForDuration",
            "key": "cmd_delay",
            "params": {"seconds": 120},
        }]
        result = import_pd_json(proto)
        assert len(result.protocol.nodes) == 1
        assert result.protocol.nodes[0].kind == NodeKind.delay
        assert result.protocol.nodes[0].params["seconds"] == 120.0

    def test_move_labware_command(self):
        proto = _make_v8_protocol()
        proto["commands"] = [{
            "commandType": "moveLabware",
            "key": "cmd_move",
            "params": {"labwareId": "plate_1", "newLocation": "5"},
        }]
        result = import_pd_json(proto)
        assert len(result.protocol.nodes) == 1
        assert result.protocol.nodes[0].kind == NodeKind.move_labware

    def test_module_command(self):
        proto = _make_v8_protocol()
        proto["commands"] = [{
            "commandType": "temperatureModule/setTargetTemperature",
            "key": "cmd_temp",
            "params": {"moduleId": "temp_mod_1", "celsius": 37},
        }]
        result = import_pd_json(proto)
        assert len(result.protocol.nodes) == 1
        node = result.protocol.nodes[0]
        assert node.kind == NodeKind.module_action
        assert node.params["action"] == "setTargetTemperature"

    def test_comment_command_stored_as_metadata(self):
        proto = _make_v8_protocol()
        proto["commands"] = [{
            "commandType": "comment",
            "key": "cmd_comment",
            "params": {"message": "Step marker"},
        }]
        result = import_pd_json(proto)
        assert len(result.protocol.nodes) == 0
        assert "Step marker" in result.protocol.metadata.get("comments", [])

    def test_empty_commands_warning(self):
        proto = _make_v8_protocol()
        proto["commands"] = []
        result = import_pd_json(proto)
        assert any("no commands" in w.lower() for w in result.warnings)


# ---------------------------------------------------------------------------
# Tests: v7/v8 step-form-based import
# ---------------------------------------------------------------------------

class TestV8StepForms:
    """Import v7/v8 protocols using designerApplication step forms."""

    def test_step_count(self):
        result = import_pd_json(_make_v8_with_step_forms())
        # 4 steps (initial deck setup is skipped)
        assert len(result.protocol.nodes) == 4

    def test_transfer_step(self):
        result = import_pd_json(_make_v8_with_step_forms())
        node = result.protocol.nodes[0]
        assert node.kind == NodeKind.transfer
        assert node.label == "Transfer Water"
        assert node.params["volume"] == 100.0
        assert node.params["source_wells"] == ["A1", "A2", "A3"]
        assert node.params["dest_wells"] == ["B1", "B2", "B3"]

    def test_mix_step(self):
        result = import_pd_json(_make_v8_with_step_forms())
        node = result.protocol.nodes[1]
        assert node.kind == NodeKind.mix
        assert node.label == "Mix Sample"
        assert node.params["volume"] == 50.0
        assert node.params["cycles"] == 5

    def test_pause_step(self):
        result = import_pd_json(_make_v8_with_step_forms())
        node = result.protocol.nodes[2]
        assert node.kind == NodeKind.pause
        assert node.params["message"] == "Check the plate"

    def test_delay_step(self):
        result = import_pd_json(_make_v8_with_step_forms())
        node = result.protocol.nodes[3]
        assert node.kind == NodeKind.delay
        assert node.params["seconds"] == 330  # 5*60 + 30

    def test_transfer_optional_params(self):
        proto = _make_v8_with_step_forms()
        form = proto["designerApplication"]["data"]["savedStepForms"]["step_transfer_1"]
        form["aspirate_mix_before_checkbox"] = True
        form["aspirate_mix_before_volume"] = "20"
        form["aspirate_mix_before_times"] = "3"
        form["disposalVolume_checkbox"] = True
        form["disposalVolume_volume"] = "5"
        result = import_pd_json(proto)
        node = result.protocol.nodes[0]
        assert node.params["mix_before"]["volume"] == 20.0
        assert node.params["mix_before"]["cycles"] == 3
        assert node.params["disposal_volume"] == 5.0


# ---------------------------------------------------------------------------
# Tests: v6 import
# ---------------------------------------------------------------------------

class TestV6Import:
    """Import v6 PD JSON (legacy step-form format)."""

    def test_basic_v6_import(self):
        result = import_pd_json(_make_v6_protocol())
        assert result.source_version == "6"
        assert result.protocol.name == "V6 Protocol"
        assert len(result.protocol.nodes) == 1

    def test_v6_transfer(self):
        result = import_pd_json(_make_v6_protocol())
        node = result.protocol.nodes[0]
        assert node.kind == NodeKind.transfer
        assert node.params["volume"] == 10.0

    def test_v6_pipettes(self):
        result = import_pd_json(_make_v6_protocol())
        pipettes = result.protocol.deck.pipettes
        assert len(pipettes) == 1
        assert pipettes[0].pipette_type == "p20_single_gen2"


# ---------------------------------------------------------------------------
# Tests: unsupported features and warnings
# ---------------------------------------------------------------------------

class TestUnsupportedFeatures:
    """Graceful handling of unknown/unsupported PD features."""

    def test_unknown_command_type_warning(self):
        proto = _make_v8_protocol()
        proto["commands"] = [{
            "commandType": "futureRobotDance",
            "key": "cmd_dance",
            "params": {},
        }]
        result = import_pd_json(proto)
        assert len(result.protocol.nodes) == 0
        assert any("futureRobotDance" in w for w in result.warnings)
        assert "futureRobotDance" in result.unsupported_features

    def test_unknown_step_type_warning(self):
        proto = _make_v8_with_step_forms()
        forms = proto["designerApplication"]["data"]["savedStepForms"]
        forms["step_unknown"] = {"stepType": "laserBeam", "stepName": "Pew"}
        proto["designerApplication"]["data"]["orderedStepIds"].append("step_unknown")
        result = import_pd_json(proto)
        assert any("laserBeam" in w for w in result.warnings)
        assert "laserBeam" in result.unsupported_features

    def test_missing_step_form_warning(self):
        proto = _make_v8_with_step_forms()
        proto["designerApplication"]["data"]["orderedStepIds"].append("step_ghost")
        result = import_pd_json(proto)
        assert any("step_ghost" in w for w in result.warnings)

    def test_no_crash_on_missing_params(self):
        """Import should not crash even with minimal/empty commands."""
        proto = _make_v8_protocol()
        proto["commands"] = [
            {"commandType": "aspirate", "key": "x", "params": {}},
            {"commandType": "dispense", "key": "y", "params": {}},
        ]
        result = import_pd_json(proto)
        assert len(result.protocol.nodes) == 2


# ---------------------------------------------------------------------------
# Tests: IR output validity
# ---------------------------------------------------------------------------

class TestIRValidity:
    """Verify the imported IR has consistent structure."""

    def test_all_nodes_have_ids(self):
        result = import_pd_json(_make_v8_with_commands())
        for node in result.protocol.nodes:
            assert node.id, f"Node missing id: {node}"

    def test_node_ids_are_unique(self):
        result = import_pd_json(_make_v8_with_commands())
        ids = [n.id for n in result.protocol.nodes]
        assert len(ids) == len(set(ids))

    def test_all_nodes_have_valid_kind(self):
        result = import_pd_json(_make_v8_with_step_forms())
        for node in result.protocol.nodes:
            assert isinstance(node.kind, NodeKind)

    def test_linear_protocol_no_graph_edges(self):
        result = import_pd_json(_make_v8_with_commands())
        assert not result.protocol.is_graph()
        assert result.protocol.edges == []

    def test_protocol_has_deck_setup(self):
        result = import_pd_json(_make_v8_with_commands())
        deck = result.protocol.deck
        assert len(deck.pipettes) > 0
        assert len(deck.labware) > 0


# ---------------------------------------------------------------------------
# Tests: module step forms
# ---------------------------------------------------------------------------

class TestModuleStepForms:
    """Module-related step form imports."""

    def test_temperature_step(self):
        proto = _make_v8_with_step_forms()
        forms = proto["designerApplication"]["data"]["savedStepForms"]
        forms["step_temp"] = {
            "stepType": "temperature",
            "stepName": "Heat to 37C",
            "moduleId": "temp_mod_1",
            "setTemperature": "true",
            "targetTemperature": "37",
        }
        proto["designerApplication"]["data"]["orderedStepIds"].append("step_temp")
        result = import_pd_json(proto)
        temp_nodes = [n for n in result.protocol.nodes if n.id == "step_temp"]
        assert len(temp_nodes) == 1
        assert temp_nodes[0].kind == NodeKind.module_action
        assert temp_nodes[0].params["target_temperature"] == 37.0

    def test_move_labware_step(self):
        proto = _make_v8_with_step_forms()
        forms = proto["designerApplication"]["data"]["savedStepForms"]
        forms["step_move"] = {
            "stepType": "moveLabware",
            "stepName": "Move plate",
            "labware": "plate_1",
            "newLocation": "5",
        }
        proto["designerApplication"]["data"]["orderedStepIds"].append("step_move")
        result = import_pd_json(proto)
        move_nodes = [n for n in result.protocol.nodes if n.id == "step_move"]
        assert len(move_nodes) == 1
        assert move_nodes[0].kind == NodeKind.move_labware
        assert move_nodes[0].params["new_slot"] == "5"


# ---------------------------------------------------------------------------
# Tests: Python importer scaffold
# ---------------------------------------------------------------------------

class TestPythonImporter:
    """Python protocol importer raises NotImplementedError."""

    def test_not_implemented(self):
        with pytest.raises(NotImplementedError, match="not yet supported"):
            import_pd_python("some_protocol.py")


# ---------------------------------------------------------------------------
# Tests: edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge cases and robustness checks."""

    def test_protocol_with_no_labware(self):
        proto = _make_v8_protocol()
        proto["labware"] = {}
        proto["commands"] = [{
            "commandType": "waitForResume",
            "key": "p",
            "params": {"message": "hi"},
        }]
        result = import_pd_json(proto)
        assert len(result.protocol.deck.labware) == 0
        assert len(result.protocol.nodes) == 1

    def test_protocol_with_no_pipettes(self):
        proto = _make_v8_protocol()
        proto["pipettes"] = {}
        proto["commands"] = [{
            "commandType": "waitForDuration",
            "key": "d",
            "params": {"seconds": 10},
        }]
        result = import_pd_json(proto)
        assert len(result.protocol.deck.pipettes) == 0

    def test_wells_as_list(self):
        """Wells can be provided as a list directly."""
        proto = _make_v8_with_step_forms()
        forms = proto["designerApplication"]["data"]["savedStepForms"]
        forms["step_transfer_1"]["aspirate_wells"] = ["A1", "A2"]
        result = import_pd_json(proto)
        node = result.protocol.nodes[0]
        assert node.params["source_wells"] == ["A1", "A2"]

    def test_volume_as_string_is_parsed(self):
        """Volumes provided as strings should be parsed to float."""
        result = import_pd_json(_make_v8_with_step_forms())
        node = result.protocol.nodes[0]  # transfer
        assert isinstance(node.params["volume"], float)

    def test_comment_step_form_stored_as_metadata(self):
        proto = _make_v8_with_step_forms()
        forms = proto["designerApplication"]["data"]["savedStepForms"]
        forms["step_comment"] = {
            "stepType": "comment",
            "message": "Important note",
        }
        proto["designerApplication"]["data"]["orderedStepIds"].append("step_comment")
        result = import_pd_json(proto)
        assert "Important note" in result.protocol.metadata.get("comments", [])
