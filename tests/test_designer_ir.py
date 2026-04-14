"""Tests for the OpenOT2 designer IR, validator, and compiler."""

from __future__ import annotations

import json

import pytest

from openot2.control.graph_models import TaskGraph
from openot2.control.models import TaskRun
from openot2.designer import (
    DeckSetup,
    Edge,
    LabwareEntry,
    LiquidEntry,
    ModuleEntry,
    Node,
    NodeKind,
    PipetteEntry,
    ProtocolIR,
    compile_graph,
    compile_linear,
    validate,
)
from openot2.designer.compiler import CompileError


# ---------------------------------------------------------------------------
# Fixtures — reusable protocol pieces
# ---------------------------------------------------------------------------

@pytest.fixture()
def deck() -> DeckSetup:
    return DeckSetup(
        labware=[
            LabwareEntry(id="plate1", slot="1",
                         labware_type="corning_96_wellplate_360ul_flat",
                         display_name="Plate"),
            LabwareEntry(id="reservoir1", slot="7",
                         labware_type="nest_12_reservoir_15ml",
                         display_name="Reservoir"),
        ],
        pipettes=[
            PipetteEntry(id="p300", mount="right",
                         pipette_type="p300_single_gen2"),
        ],
        modules=[
            ModuleEntry(id="tempdeck1", slot="3",
                        module_type="temperatureModuleV2",
                        display_name="Temp"),
        ],
        liquids=[
            LiquidEntry(id="dye_red", name="Red Dye", color="#FF0000"),
        ],
    )


def _linear_protocol(deck: DeckSetup) -> ProtocolIR:
    """A simple linear 3-step protocol: transfer → mix → delay."""
    return ProtocolIR(
        name="Linear Test",
        description="Transfer + mix + delay",
        deck=deck,
        nodes=[
            Node(
                id="step_transfer",
                kind=NodeKind.transfer,
                label="Transfer dye",
                params={
                    "pipette_id": "p300",
                    "source_labware_id": "reservoir1",
                    "source_wells": ["A1"],
                    "dest_labware_id": "plate1",
                    "dest_wells": ["A1"],
                    "volume": 100.0,
                },
            ),
            Node(
                id="step_mix",
                kind=NodeKind.mix,
                label="Mix well",
                params={
                    "pipette_id": "p300",
                    "labware_id": "plate1",
                    "wells": ["A1"],
                    "volume": 80.0,
                    "cycles": 3,
                },
            ),
            Node(
                id="step_delay",
                kind=NodeKind.delay,
                label="Settle",
                params={"seconds": 30.0},
            ),
        ],
    )


def _branching_protocol(deck: DeckSetup) -> ProtocolIR:
    """A protocol with capture → predict → branch."""
    return ProtocolIR(
        name="Branch Test",
        description="Capture + predict + branch",
        deck=deck,
        nodes=[
            Node(
                id="do_transfer",
                kind=NodeKind.transfer,
                label="Initial transfer",
                params={
                    "pipette_id": "p300",
                    "source_labware_id": "reservoir1",
                    "source_wells": ["A1"],
                    "dest_labware_id": "plate1",
                    "dest_wells": ["A1"],
                    "volume": 50.0,
                },
                next="do_capture",
            ),
            Node(
                id="do_capture",
                kind=NodeKind.capture,
                label="Take image",
                params={"camera_id": 0, "label": "qc"},
                next="do_predict",
            ),
            Node(
                id="do_predict",
                kind=NodeKind.predict,
                label="Run model",
                params={"model_path": "model.pt"},
                next="do_branch",
            ),
            Node(
                id="do_branch",
                kind=NodeKind.branch,
                label="Check quality",
                params={"condition": "predict.output.result"},
                on_true="done_pause",
                on_false="retry_transfer",
            ),
            Node(
                id="retry_transfer",
                kind=NodeKind.transfer,
                label="Retry transfer",
                params={
                    "pipette_id": "p300",
                    "source_labware_id": "reservoir1",
                    "source_wells": ["A1"],
                    "dest_labware_id": "plate1",
                    "dest_wells": ["A1"],
                    "volume": 50.0,
                },
                next="do_capture",
            ),
            Node(
                id="done_pause",
                kind=NodeKind.pause,
                label="Done",
                params={"message": "QC passed"},
            ),
        ],
    )


# ===================================================================
# IR creation
# ===================================================================

class TestIRCreation:
    def test_empty_protocol(self):
        p = ProtocolIR(name="empty")
        assert p.name == "empty"
        assert p.nodes == []
        assert p.edges == []
        assert not p.is_graph()
        assert not p.has_branch_nodes()

    def test_linear_protocol(self, deck: DeckSetup):
        p = _linear_protocol(deck)
        assert len(p.nodes) == 3
        assert p.node_ids() == ["step_transfer", "step_mix", "step_delay"]
        assert not p.is_graph()
        assert not p.has_branch_nodes()

    def test_branching_protocol(self, deck: DeckSetup):
        p = _branching_protocol(deck)
        assert len(p.nodes) == 6
        assert p.is_graph()
        assert p.has_branch_nodes()

    def test_node_by_id(self, deck: DeckSetup):
        p = _linear_protocol(deck)
        node = p.node_by_id("step_mix")
        assert node is not None
        assert node.kind == NodeKind.mix
        assert p.node_by_id("nonexistent") is None

    def test_deck_setup(self, deck: DeckSetup):
        assert len(deck.labware) == 2
        assert len(deck.pipettes) == 1
        assert len(deck.modules) == 1
        assert len(deck.liquids) == 1
        assert deck.pipettes[0].mount == "right"

    def test_node_kinds_are_strings(self):
        assert NodeKind.transfer.value == "transfer"
        assert NodeKind.branch.value == "branch"
        assert NodeKind.capture.value == "capture"

    def test_explicit_edges(self, deck: DeckSetup):
        p = ProtocolIR(
            name="With edges",
            deck=deck,
            nodes=[
                Node(id="a", kind=NodeKind.pause),
                Node(id="b", kind=NodeKind.pause),
            ],
            edges=[Edge(source="a", target="b", label="default")],
        )
        assert p.is_graph()
        assert len(p.edges) == 1

    def test_variables(self):
        p = ProtocolIR(
            name="vars",
            variables={"target_color": [255, 0, 0], "threshold": 0.05},
        )
        assert p.variables["threshold"] == 0.05


# ===================================================================
# Serialization
# ===================================================================

class TestSerialization:
    def test_round_trip_json(self, deck: DeckSetup):
        p = _linear_protocol(deck)
        data = p.model_dump(mode="json")
        restored = ProtocolIR.model_validate(data)
        assert restored.name == p.name
        assert len(restored.nodes) == len(p.nodes)
        assert restored.nodes[0].kind == NodeKind.transfer

    def test_json_string_round_trip(self, deck: DeckSetup):
        p = _branching_protocol(deck)
        json_str = p.model_dump_json()
        restored = ProtocolIR.model_validate_json(json_str)
        assert restored.name == p.name
        assert restored.has_branch_nodes()

    def test_node_serialization(self):
        n = Node(id="x", kind=NodeKind.capture, params={"camera_id": 0})
        data = n.model_dump(mode="json")
        assert data["kind"] == "capture"
        assert data["params"]["camera_id"] == 0
        restored = Node.model_validate(data)
        assert restored.kind == NodeKind.capture

    def test_json_is_parseable(self, deck: DeckSetup):
        p = _linear_protocol(deck)
        raw = json.loads(p.model_dump_json())
        assert isinstance(raw, dict)
        assert raw["name"] == "Linear Test"
        assert len(raw["nodes"]) == 3


# ===================================================================
# Validation
# ===================================================================

class TestValidation:
    def test_valid_linear(self, deck: DeckSetup):
        p = _linear_protocol(deck)
        assert validate(p) == []

    def test_valid_branching(self, deck: DeckSetup):
        p = _branching_protocol(deck)
        assert validate(p) == []

    def test_duplicate_node_ids(self, deck: DeckSetup):
        p = ProtocolIR(
            name="dup",
            deck=deck,
            nodes=[
                Node(id="same", kind=NodeKind.pause),
                Node(id="same", kind=NodeKind.delay, params={"seconds": 1}),
            ],
        )
        errors = validate(p)
        assert any("Duplicate node id" in e for e in errors)

    def test_unknown_edge_target(self, deck: DeckSetup):
        p = ProtocolIR(
            name="bad edge",
            deck=deck,
            nodes=[
                Node(id="a", kind=NodeKind.pause, next="nonexistent"),
            ],
        )
        errors = validate(p)
        assert any("unknown node 'nonexistent'" in e for e in errors)

    def test_branch_missing_targets(self, deck: DeckSetup):
        p = ProtocolIR(
            name="bad branch",
            deck=deck,
            nodes=[
                Node(
                    id="br",
                    kind=NodeKind.branch,
                    params={"condition": "x > 0"},
                ),
            ],
        )
        errors = validate(p)
        assert any("missing on_true" in e for e in errors)
        assert any("missing on_false" in e for e in errors)

    def test_missing_required_params(self, deck: DeckSetup):
        p = ProtocolIR(
            name="missing params",
            deck=deck,
            nodes=[
                Node(id="t", kind=NodeKind.transfer, params={}),
            ],
        )
        errors = validate(p)
        assert any("missing required params" in e for e in errors)

    def test_deck_reference_missing_pipette(self, deck: DeckSetup):
        p = ProtocolIR(
            name="bad pip",
            deck=deck,
            nodes=[
                Node(
                    id="t",
                    kind=NodeKind.transfer,
                    params={
                        "pipette_id": "nonexistent_pipette",
                        "source_labware_id": "reservoir1",
                        "source_wells": ["A1"],
                        "dest_labware_id": "plate1",
                        "dest_wells": ["A1"],
                        "volume": 100.0,
                    },
                ),
            ],
        )
        errors = validate(p)
        assert any("pipette_id" in e and "not found on deck" in e for e in errors)

    def test_deck_reference_missing_labware(self, deck: DeckSetup):
        p = ProtocolIR(
            name="bad lw",
            deck=deck,
            nodes=[
                Node(
                    id="t",
                    kind=NodeKind.transfer,
                    params={
                        "pipette_id": "p300",
                        "source_labware_id": "ghost",
                        "source_wells": ["A1"],
                        "dest_labware_id": "plate1",
                        "dest_wells": ["A1"],
                        "volume": 100.0,
                    },
                ),
            ],
        )
        errors = validate(p)
        assert any("source_labware_id" in e and "not found on deck" in e
                    for e in errors)

    def test_deck_reference_missing_module(self, deck: DeckSetup):
        p = ProtocolIR(
            name="bad mod",
            deck=deck,
            nodes=[
                Node(
                    id="m",
                    kind=NodeKind.module_action,
                    params={
                        "module_id": "ghost_module",
                        "action": "set_temperature",
                    },
                ),
            ],
        )
        errors = validate(p)
        assert any("module_id" in e and "not found on deck" in e for e in errors)

    def test_unreachable_node(self, deck: DeckSetup):
        p = ProtocolIR(
            name="island",
            deck=deck,
            nodes=[
                Node(id="a", kind=NodeKind.pause, next="b"),
                Node(id="b", kind=NodeKind.pause),
                Node(id="island", kind=NodeKind.pause),  # no edge to this
            ],
        )
        errors = validate(p)
        assert any("unreachable" in e for e in errors)

    def test_explicit_edge_bad_source(self, deck: DeckSetup):
        p = ProtocolIR(
            name="bad edge",
            deck=deck,
            nodes=[Node(id="a", kind=NodeKind.pause)],
            edges=[Edge(source="ghost", target="a")],
        )
        errors = validate(p)
        assert any("Edge source 'ghost'" in e for e in errors)

    def test_valid_pause_no_params(self, deck: DeckSetup):
        """Pause/capture/predict have no required params."""
        p = ProtocolIR(
            name="ok",
            deck=deck,
            nodes=[Node(id="p", kind=NodeKind.pause)],
        )
        assert validate(p) == []

    def test_empty_protocol_valid(self):
        p = ProtocolIR(name="empty")
        assert validate(p) == []


# ===================================================================
# Linear compile
# ===================================================================

class TestLinearCompile:
    def test_basic_compile(self, deck: DeckSetup):
        p = _linear_protocol(deck)
        run = compile_linear(p)
        assert isinstance(run, TaskRun)
        assert run.name == "Linear Test"
        assert len(run.steps) == 3
        assert run.steps[0].kind == "transfer"
        assert run.steps[0].key == "step_transfer"
        assert run.steps[1].kind == "mix"
        assert run.steps[2].kind == "delay"

    def test_params_carried_through(self, deck: DeckSetup):
        p = _linear_protocol(deck)
        run = compile_linear(p)
        assert run.steps[0].params["volume"] == 100.0
        assert run.steps[1].params["cycles"] == 3
        assert run.steps[2].params["seconds"] == 30.0

    def test_metadata_in_output(self, deck: DeckSetup):
        p = _linear_protocol(deck)
        p.metadata["author"] = "test"
        run = compile_linear(p)
        assert run.metadata["designer_protocol_id"] == p.id
        assert run.metadata["author"] == "test"

    def test_step_names(self, deck: DeckSetup):
        p = _linear_protocol(deck)
        run = compile_linear(p)
        assert run.steps[0].name == "Transfer dye"
        assert run.steps[2].name == "Settle"

    def test_step_name_fallback(self, deck: DeckSetup):
        """Nodes without labels get auto-generated names."""
        p = ProtocolIR(
            name="no labels",
            deck=deck,
            nodes=[Node(id="x", kind=NodeKind.pause)],
        )
        run = compile_linear(p)
        assert run.steps[0].name == "pause_0"

    def test_rejects_branch_nodes(self, deck: DeckSetup):
        p = _branching_protocol(deck)
        with pytest.raises(CompileError) as exc_info:
            compile_linear(p)
        assert "branch" in str(exc_info.value).lower()

    def test_rejects_invalid_protocol(self, deck: DeckSetup):
        p = ProtocolIR(
            name="bad",
            deck=deck,
            nodes=[Node(id="t", kind=NodeKind.transfer, params={})],
        )
        with pytest.raises(CompileError):
            compile_linear(p)

    def test_single_node(self, deck: DeckSetup):
        p = ProtocolIR(
            name="one",
            deck=deck,
            nodes=[
                Node(id="x", kind=NodeKind.capture, label="snap"),
            ],
        )
        run = compile_linear(p)
        assert len(run.steps) == 1
        assert run.steps[0].kind == "capture"


# ===================================================================
# Graph compile
# ===================================================================

class TestGraphCompile:
    def test_branching_compile(self, deck: DeckSetup):
        p = _branching_protocol(deck)
        graph = compile_graph(p)
        assert isinstance(graph, TaskGraph)
        assert graph.name == "Branch Test"
        assert len(graph.nodes) == 6
        assert graph.start == "do_transfer"

    def test_branch_edges(self, deck: DeckSetup):
        p = _branching_protocol(deck)
        graph = compile_graph(p)
        branch = graph.nodes["do_branch"]
        assert branch.kind == "branch"
        assert branch.on_true == "done_pause"
        assert branch.on_false == "retry_transfer"

    def test_linear_as_graph(self, deck: DeckSetup):
        """A linear protocol compiled as graph gets auto-chained edges."""
        p = _linear_protocol(deck)
        graph = compile_graph(p)
        assert graph.nodes["step_transfer"].next == "step_mix"
        assert graph.nodes["step_mix"].next == "step_delay"
        assert graph.nodes["step_delay"].next is None  # last node

    def test_graph_validation_errors(self, deck: DeckSetup):
        p = ProtocolIR(
            name="bad",
            deck=deck,
            nodes=[Node(id="t", kind=NodeKind.transfer, params={})],
        )
        with pytest.raises(CompileError):
            compile_graph(p)

    def test_empty_protocol_error(self, deck: DeckSetup):
        p = ProtocolIR(name="empty", deck=deck)
        with pytest.raises(CompileError) as exc_info:
            compile_graph(p)
        assert "no nodes" in str(exc_info.value).lower()

    def test_metadata_in_graph(self, deck: DeckSetup):
        p = _branching_protocol(deck)
        p.metadata["version"] = 2
        graph = compile_graph(p)
        assert graph.metadata["designer_protocol_id"] == p.id
        assert graph.metadata["version"] == 2

    def test_graph_node_names(self, deck: DeckSetup):
        p = _branching_protocol(deck)
        graph = compile_graph(p)
        assert graph.nodes["do_capture"].name == "Take image"

    def test_graph_validates_graph(self, deck: DeckSetup):
        """The produced TaskGraph should pass its own validation."""
        p = _branching_protocol(deck)
        graph = compile_graph(p)
        assert graph.validate_graph() == []


# ===================================================================
# Node kind coverage
# ===================================================================

class TestNodeKindCoverage:
    """Ensure all required node kinds are representable and compilable."""

    def _single_node_protocol(
        self, deck: DeckSetup, kind: NodeKind, params: dict
    ) -> ProtocolIR:
        return ProtocolIR(
            name=f"{kind.value} test",
            deck=deck,
            nodes=[Node(id="n", kind=kind, params=params)],
        )

    def test_transfer(self, deck: DeckSetup):
        p = self._single_node_protocol(deck, NodeKind.transfer, {
            "pipette_id": "p300",
            "source_labware_id": "reservoir1",
            "source_wells": ["A1"],
            "dest_labware_id": "plate1",
            "dest_wells": ["A1"],
            "volume": 50.0,
        })
        assert validate(p) == []
        run = compile_linear(p)
        assert run.steps[0].kind == "transfer"

    def test_mix(self, deck: DeckSetup):
        p = self._single_node_protocol(deck, NodeKind.mix, {
            "pipette_id": "p300",
            "labware_id": "plate1",
            "wells": ["A1"],
            "volume": 80.0,
            "cycles": 3,
        })
        assert validate(p) == []

    def test_pause(self, deck: DeckSetup):
        p = self._single_node_protocol(deck, NodeKind.pause, {
            "message": "Insert plate",
        })
        assert validate(p) == []

    def test_delay(self, deck: DeckSetup):
        p = self._single_node_protocol(deck, NodeKind.delay, {
            "seconds": 60.0, "message": "Wait for settling",
        })
        assert validate(p) == []

    def test_module_action(self, deck: DeckSetup):
        p = self._single_node_protocol(deck, NodeKind.module_action, {
            "module_id": "tempdeck1",
            "action": "set_temperature",
            "temperature": 37.0,
        })
        assert validate(p) == []

    def test_move_labware(self, deck: DeckSetup):
        p = self._single_node_protocol(deck, NodeKind.move_labware, {
            "labware_id": "plate1",
            "new_slot": "5",
        })
        assert validate(p) == []

    def test_capture(self, deck: DeckSetup):
        p = self._single_node_protocol(deck, NodeKind.capture, {
            "camera_id": 0, "label": "pre-mix",
        })
        assert validate(p) == []

    def test_predict(self, deck: DeckSetup):
        p = self._single_node_protocol(deck, NodeKind.predict, {
            "model_path": "models/color_qc.pt",
            "min_detections": 1,
        })
        assert validate(p) == []

    def test_branch(self, deck: DeckSetup):
        p = ProtocolIR(
            name="branch test",
            deck=deck,
            nodes=[
                Node(
                    id="br",
                    kind=NodeKind.branch,
                    params={"condition": "predict.output.result"},
                    on_true="yes",
                    on_false="no",
                ),
                Node(id="yes", kind=NodeKind.pause),
                Node(id="no", kind=NodeKind.pause),
            ],
        )
        assert validate(p) == []
