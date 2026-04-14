"""Tests for openot2.designer.extensions — capture/predict/branch builders and dataflow."""

from __future__ import annotations

import pytest

from openot2.designer.ir import (
    DeckSetup,
    Edge,
    LabwareEntry,
    Node,
    NodeKind,
    PipetteEntry,
    ProtocolIR,
)
from openot2.designer.extensions import (
    build_capture_predict_branch,
    capture_artifact_key,
    get_dataflow,
    link_capture_to_predict,
    link_predict_to_branch,
    make_branch,
    make_capture,
    make_predict,
    predict_result_key,
)
from openot2.designer.validator import validate


# ---------------------------------------------------------------------------
# Artifact key conventions
# ---------------------------------------------------------------------------

class TestArtifactKeys:
    def test_capture_artifact_key(self):
        assert capture_artifact_key("cap1") == "cap1.image"

    def test_predict_result_key(self):
        assert predict_result_key("pred1") == "pred1.result"


# ---------------------------------------------------------------------------
# Node builders
# ---------------------------------------------------------------------------

class TestMakeCapture:
    def test_defaults(self):
        node = make_capture()
        assert node.kind == NodeKind.capture
        assert node.params["camera_id"] == 0
        assert "artifact_key" in node.params
        assert node.label == "Capture image"

    def test_explicit_id(self):
        node = make_capture(node_id="my_cap")
        assert node.id == "my_cap"
        assert node.params["artifact_key"] == "my_cap.image"

    def test_custom_artifact_key(self):
        node = make_capture(artifact_key="custom.img")
        assert node.params["artifact_key"] == "custom.img"

    def test_extra_params(self):
        node = make_capture(extra_params={"exposure": 100})
        assert node.params["exposure"] == 100


class TestMakePredict:
    def test_defaults(self):
        node = make_predict(model="yolo_v8")
        assert node.kind == NodeKind.predict
        assert node.params["model"] == "yolo_v8"
        assert "result_key" in node.params

    def test_explicit_id(self):
        node = make_predict(node_id="my_pred", model="m")
        assert node.id == "my_pred"
        assert node.params["result_key"] == "my_pred.result"

    def test_source_artifact(self):
        node = make_predict(model="m", source_artifact="cap1.image")
        assert node.params["source_artifact"] == "cap1.image"

    def test_threshold(self):
        node = make_predict(model="m", threshold=0.8)
        assert node.params["threshold"] == 0.8

    def test_extra_params(self):
        node = make_predict(model="m", extra_params={"batch_size": 4})
        assert node.params["batch_size"] == 4


class TestMakeBranch:
    def test_basic(self):
        node = make_branch(condition="pred.result", on_true="a", on_false="b")
        assert node.kind == NodeKind.branch
        assert node.params["condition"] == "pred.result"
        assert node.on_true == "a"
        assert node.on_false == "b"

    def test_explicit_id(self):
        node = make_branch(
            node_id="br1", condition="x", on_true="a", on_false="b"
        )
        assert node.id == "br1"

    def test_label(self):
        node = make_branch(
            condition="x", on_true="a", on_false="b", label="QC check"
        )
        assert node.label == "QC check"

    def test_auto_label(self):
        node = make_branch(condition="pred.result", on_true="a", on_false="b")
        assert "pred.result" in node.label


# ---------------------------------------------------------------------------
# Link helpers
# ---------------------------------------------------------------------------

class TestLinkHelpers:
    def test_link_capture_to_predict(self):
        cap = make_capture(node_id="cap1")
        pred = make_predict(node_id="pred1", model="m")
        link_capture_to_predict(cap, pred)
        assert pred.params["source_artifact"] == "cap1.image"

    def test_link_predict_to_branch(self):
        pred = make_predict(node_id="pred1", model="m")
        br = make_branch(condition="placeholder", on_true="a", on_false="b")
        link_predict_to_branch(pred, br)
        assert br.params["condition"] == "pred1.result"


# ---------------------------------------------------------------------------
# Dataflow extraction
# ---------------------------------------------------------------------------

class TestGetDataflow:
    def test_capture_predict_flow(self):
        cap = make_capture(node_id="cap1")
        pred = make_predict(node_id="pred1", model="m")
        link_capture_to_predict(cap, pred)

        proto = ProtocolIR(name="test", nodes=[cap, pred])
        flows = get_dataflow(proto)
        assert len(flows) == 1
        assert flows[0]["producer"] == "cap1"
        assert flows[0]["consumer"] == "pred1"
        assert flows[0]["artifact"] == "cap1.image"

    def test_full_chain_flow(self):
        cap = make_capture(node_id="cap1")
        pred = make_predict(node_id="pred1", model="m")
        br = make_branch(
            node_id="br1", condition="placeholder", on_true="done", on_false="retry"
        )
        link_capture_to_predict(cap, pred)
        link_predict_to_branch(pred, br)

        proto = ProtocolIR(
            name="test",
            nodes=[cap, pred, br],
        )
        flows = get_dataflow(proto)
        assert len(flows) == 2
        producers = {f["producer"] for f in flows}
        assert "cap1" in producers
        assert "pred1" in producers

    def test_no_flow_in_linear_protocol(self):
        proto = ProtocolIR(
            name="test",
            nodes=[
                Node(id="n1", kind=NodeKind.transfer, params={}),
                Node(id="n2", kind=NodeKind.delay, params={"seconds": 5}),
            ],
        )
        assert get_dataflow(proto) == []


# ---------------------------------------------------------------------------
# Workflow builder
# ---------------------------------------------------------------------------

class TestBuildCapturePredicBranch:
    def test_basic_build(self):
        cap, pred, br = build_capture_predict_branch(
            model="tip_check", on_true="ok", on_false="fail"
        )
        assert cap.kind == NodeKind.capture
        assert pred.kind == NodeKind.predict
        assert br.kind == NodeKind.branch

    def test_wiring(self):
        cap, pred, br = build_capture_predict_branch(
            model="tip_check", on_true="ok", on_false="fail"
        )
        # Capture output flows to predict input
        assert pred.params["source_artifact"] == cap.params["artifact_key"]
        # Predict result flows to branch condition
        assert br.params["condition"] == pred.params["result_key"]

    def test_graph_edges(self):
        cap, pred, br = build_capture_predict_branch(
            capture_id="c",
            predict_id="p",
            branch_id="b",
            model="m",
            on_true="ok",
            on_false="fail",
        )
        assert cap.next == "p"
        assert pred.next == "b"
        assert br.on_true == "ok"
        assert br.on_false == "fail"

    def test_custom_ids(self):
        cap, pred, br = build_capture_predict_branch(
            capture_id="my_cap",
            predict_id="my_pred",
            branch_id="my_br",
            model="m",
            on_true="a",
            on_false="b",
        )
        assert cap.id == "my_cap"
        assert pred.id == "my_pred"
        assert br.id == "my_br"

    def test_threshold(self):
        _, pred, _ = build_capture_predict_branch(
            model="m", on_true="a", on_false="b", threshold=0.9
        )
        assert pred.params["threshold"] == 0.9


# ---------------------------------------------------------------------------
# Full workflow: transfer -> mix -> capture -> predict -> branch
# ---------------------------------------------------------------------------

class TestFullWorkflowIR:
    def _build_workflow(self) -> ProtocolIR:
        deck = DeckSetup(
            labware=[
                LabwareEntry(id="plate1", slot="1", labware_type="corning_96"),
                LabwareEntry(id="plate2", slot="2", labware_type="corning_96"),
            ],
            pipettes=[
                PipetteEntry(id="pip1", mount="right", pipette_type="p300_single"),
            ],
        )

        transfer = Node(
            id="transfer1",
            kind=NodeKind.transfer,
            params={
                "pipette_id": "pip1",
                "source_labware_id": "plate1",
                "source_wells": ["A1"],
                "dest_labware_id": "plate2",
                "dest_wells": ["A1"],
                "volume": 100,
            },
            next="mix1",
        )
        mix = Node(
            id="mix1",
            kind=NodeKind.mix,
            params={
                "pipette_id": "pip1",
                "labware_id": "plate2",
                "wells": ["A1"],
                "volume": 50,
                "cycles": 3,
            },
        )

        cap, pred, br = build_capture_predict_branch(
            capture_id="cap1",
            predict_id="pred1",
            branch_id="br1",
            model="tip_detector_v2",
            on_true="done",
            on_false="retry",
            threshold=0.85,
        )

        mix.next = "cap1"

        done = Node(id="done", kind=NodeKind.pause, params={"message": "Success"})
        retry = Node(
            id="retry",
            kind=NodeKind.mix,
            params={
                "pipette_id": "pip1",
                "labware_id": "plate2",
                "wells": ["A1"],
                "volume": 50,
                "cycles": 5,
            },
            next="cap1",
        )

        return ProtocolIR(
            name="Transfer with QC",
            deck=deck,
            nodes=[transfer, mix, cap, pred, br, done, retry],
        )

    def test_workflow_is_graph(self):
        proto = self._build_workflow()
        assert proto.is_graph()

    def test_workflow_has_branch(self):
        proto = self._build_workflow()
        assert proto.has_branch_nodes()

    def test_workflow_validates(self):
        proto = self._build_workflow()
        errors = validate(proto)
        assert errors == [], f"Validation errors: {errors}"

    def test_workflow_dataflow(self):
        proto = self._build_workflow()
        flows = get_dataflow(proto)
        assert len(flows) == 2

        cap_to_pred = [f for f in flows if f["producer"] == "cap1"]
        assert len(cap_to_pred) == 1
        assert cap_to_pred[0]["consumer"] == "pred1"

        pred_to_br = [f for f in flows if f["producer"] == "pred1"]
        assert len(pred_to_br) == 1
        assert pred_to_br[0]["consumer"] == "br1"

    def test_workflow_node_count(self):
        proto = self._build_workflow()
        assert len(proto.nodes) == 7

    def test_workflow_node_kinds(self):
        proto = self._build_workflow()
        kinds = [n.kind for n in proto.nodes]
        assert NodeKind.capture in kinds
        assert NodeKind.predict in kinds
        assert NodeKind.branch in kinds
        assert NodeKind.transfer in kinds
        assert NodeKind.mix in kinds
        assert NodeKind.pause in kinds
