"""Tests for openot2.designer.visualize — timeline, graph, and Mermaid projections."""

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
from openot2.designer.visualize import graph, summarize_node, timeline, to_mermaid


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _deck() -> DeckSetup:
    return DeckSetup(
        labware=[
            LabwareEntry(id="plate1", slot="1", labware_type="corning_96"),
            LabwareEntry(id="plate2", slot="2", labware_type="corning_96"),
        ],
        pipettes=[PipetteEntry(id="pip1", mount="right", pipette_type="p300_single")],
    )


def _linear_protocol() -> ProtocolIR:
    """Three-step linear protocol: transfer -> mix -> delay."""
    return ProtocolIR(
        id="proto-linear",
        name="Linear test",
        deck=_deck(),
        nodes=[
            Node(
                id="n1",
                kind=NodeKind.transfer,
                label="",
                params={
                    "pipette_id": "pip1",
                    "source_labware_id": "plate1",
                    "source_wells": ["A1"],
                    "dest_labware_id": "plate2",
                    "dest_wells": ["A1"],
                    "volume": 50,
                },
            ),
            Node(
                id="n2",
                kind=NodeKind.mix,
                label="Mix well",
                params={
                    "pipette_id": "pip1",
                    "labware_id": "plate2",
                    "wells": ["A1"],
                    "volume": 30,
                    "cycles": 3,
                },
            ),
            Node(id="n3", kind=NodeKind.delay, params={"seconds": 10}),
        ],
    )


def _branched_protocol() -> ProtocolIR:
    """Protocol with capture -> predict -> branch -> two paths."""
    return ProtocolIR(
        id="proto-branch",
        name="Branched test",
        deck=_deck(),
        nodes=[
            Node(
                id="t1",
                kind=NodeKind.transfer,
                params={
                    "pipette_id": "pip1",
                    "source_labware_id": "plate1",
                    "source_wells": ["A1"],
                    "dest_labware_id": "plate2",
                    "dest_wells": ["A1"],
                    "volume": 100,
                },
            ),
            Node(
                id="cap1",
                kind=NodeKind.capture,
                label="QC capture",
                params={"camera_id": 0, "artifact_key": "cap1.image"},
            ),
            Node(
                id="pred1",
                kind=NodeKind.predict,
                params={
                    "model": "tip_check_v2",
                    "source_artifact": "cap1.image",
                    "result_key": "pred1.result",
                },
                next="br1",
            ),
            Node(
                id="br1",
                kind=NodeKind.branch,
                params={"condition": "pred1.result"},
                on_true="done",
                on_false="retry",
            ),
            Node(id="done", kind=NodeKind.pause, params={"message": "All good"}),
            Node(
                id="retry",
                kind=NodeKind.mix,
                params={
                    "pipette_id": "pip1",
                    "labware_id": "plate2",
                    "wells": ["A1"],
                    "volume": 30,
                    "cycles": 5,
                },
                next="cap1",
            ),
        ],
        edges=[
            Edge(source="t1", target="cap1"),
        ],
    )


# ---------------------------------------------------------------------------
# summarize_node tests
# ---------------------------------------------------------------------------

class TestSummarizeNode:
    def test_transfer_summary(self):
        node = Node(
            id="x",
            kind=NodeKind.transfer,
            params={
                "pipette_id": "p",
                "source_labware_id": "s",
                "source_wells": ["A1"],
                "dest_labware_id": "d",
                "dest_wells": ["B2"],
                "volume": 75,
            },
        )
        s = summarize_node(node)
        assert "Transfer" in s
        assert "75" in s
        assert "A1" in s or "['A1']" in s

    def test_label_overrides(self):
        node = Node(id="x", kind=NodeKind.transfer, label="Custom label", params={})
        assert summarize_node(node) == "Transfer: Custom label"

    def test_mix_summary(self):
        node = Node(
            id="x",
            kind=NodeKind.mix,
            params={"volume": 30, "cycles": 3, "wells": ["A1"]},
        )
        s = summarize_node(node)
        assert "Mix" in s
        assert "x3" in s

    def test_delay_summary(self):
        node = Node(id="x", kind=NodeKind.delay, params={"seconds": 60})
        assert "60s" in summarize_node(node)

    def test_capture_summary(self):
        node = Node(
            id="x",
            kind=NodeKind.capture,
            params={"camera_id": 1, "artifact_key": "cap.image"},
        )
        s = summarize_node(node)
        assert "Capture" in s
        assert "cap.image" in s

    def test_predict_summary(self):
        node = Node(
            id="x",
            kind=NodeKind.predict,
            params={"model": "yolo_v8", "source_artifact": "cap.image"},
        )
        s = summarize_node(node)
        assert "yolo_v8" in s
        assert "cap.image" in s

    def test_branch_summary(self):
        node = Node(
            id="x",
            kind=NodeKind.branch,
            params={"condition": "pred.result"},
            on_true="a",
            on_false="b",
        )
        s = summarize_node(node)
        assert "Branch" in s
        assert "pred.result" in s

    def test_pause_with_message(self):
        node = Node(id="x", kind=NodeKind.pause, params={"message": "Wait for user"})
        s = summarize_node(node)
        assert "Pause" in s
        assert "Wait for user" in s

    def test_pause_no_message(self):
        node = Node(id="x", kind=NodeKind.pause, params={})
        assert summarize_node(node) == "Pause"

    def test_module_action_summary(self):
        node = Node(
            id="x",
            kind=NodeKind.module_action,
            params={"module_id": "temp1", "action": "set_temp"},
        )
        s = summarize_node(node)
        assert "set_temp" in s
        assert "temp1" in s

    def test_move_labware_summary(self):
        node = Node(
            id="x",
            kind=NodeKind.move_labware,
            params={"labware_id": "plate1", "new_slot": "5"},
        )
        s = summarize_node(node)
        assert "plate1" in s
        assert "slot 5" in s


# ---------------------------------------------------------------------------
# Timeline tests
# ---------------------------------------------------------------------------

class TestTimeline:
    def test_timeline_count(self):
        proto = _linear_protocol()
        tl = timeline(proto)
        assert len(tl) == 3

    def test_timeline_order(self):
        proto = _linear_protocol()
        tl = timeline(proto)
        assert [e["index"] for e in tl] == [0, 1, 2]
        assert [e["id"] for e in tl] == ["n1", "n2", "n3"]

    def test_timeline_kinds(self):
        proto = _linear_protocol()
        tl = timeline(proto)
        assert [e["kind"] for e in tl] == ["transfer", "mix", "delay"]

    def test_timeline_summaries_present(self):
        proto = _linear_protocol()
        tl = timeline(proto)
        for entry in tl:
            assert isinstance(entry["summary"], str)
            assert len(entry["summary"]) > 0

    def test_empty_protocol(self):
        proto = ProtocolIR(name="Empty")
        assert timeline(proto) == []


# ---------------------------------------------------------------------------
# Graph projection tests
# ---------------------------------------------------------------------------

class TestGraph:
    def test_linear_graph_nodes(self):
        proto = _linear_protocol()
        g = graph(proto)
        assert len(g["nodes"]) == 3
        assert g["protocol_name"] == "Linear test"

    def test_linear_graph_edges_chained(self):
        proto = _linear_protocol()
        g = graph(proto)
        edges = g["edges"]
        # Linear fallback should chain n1->n2->n3
        assert len(edges) == 2
        assert edges[0]["source"] == "n1"
        assert edges[0]["target"] == "n2"
        assert edges[1]["source"] == "n2"
        assert edges[1]["target"] == "n3"

    def test_branched_graph_edges(self):
        proto = _branched_protocol()
        g = graph(proto)
        edge_tuples = {(e["source"], e["target"]) for e in g["edges"]}
        # Must have branch edges
        assert ("br1", "done") in edge_tuples
        assert ("br1", "retry") in edge_tuples
        # Explicit edge
        assert ("t1", "cap1") in edge_tuples
        # Predict -> branch
        assert ("pred1", "br1") in edge_tuples

    def test_branch_node_is_decision(self):
        proto = _branched_protocol()
        g = graph(proto)
        branch_nodes = [n for n in g["nodes"] if n.get("is_decision")]
        assert len(branch_nodes) == 1
        assert branch_nodes[0]["id"] == "br1"

    def test_branch_edge_labels(self):
        proto = _branched_protocol()
        g = graph(proto)
        true_edges = [e for e in g["edges"] if e["condition"] == "true"]
        false_edges = [e for e in g["edges"] if e["condition"] == "false"]
        assert len(true_edges) == 1
        assert true_edges[0]["label"] == "yes"
        assert len(false_edges) == 1
        assert false_edges[0]["label"] == "no"

    def test_no_duplicate_edges(self):
        proto = _branched_protocol()
        g = graph(proto)
        edge_keys = [(e["source"], e["target"], e["condition"]) for e in g["edges"]]
        assert len(edge_keys) == len(set(edge_keys))

    def test_empty_protocol(self):
        proto = ProtocolIR(name="Empty")
        g = graph(proto)
        assert g["nodes"] == []
        assert g["edges"] == []


# ---------------------------------------------------------------------------
# Mermaid export tests
# ---------------------------------------------------------------------------

class TestMermaid:
    def test_mermaid_starts_with_graph_td(self):
        proto = _linear_protocol()
        m = to_mermaid(proto)
        assert m.startswith("graph TD")

    def test_mermaid_has_all_nodes(self):
        proto = _linear_protocol()
        m = to_mermaid(proto)
        for node in proto.nodes:
            assert node.id in m

    def test_mermaid_branch_diamond(self):
        proto = _branched_protocol()
        m = to_mermaid(proto)
        # Branch nodes should use diamond syntax {text}
        assert 'br1{"' in m or "br1{" in m

    def test_mermaid_edges_present(self):
        proto = _branched_protocol()
        m = to_mermaid(proto)
        assert "-->" in m
        # Branch edge labels
        assert "|yes|" in m
        assert "|no|" in m
