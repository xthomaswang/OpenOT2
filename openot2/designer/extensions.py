"""OpenOT2-native extension nodes: capture, predict, branch.

These nodes go beyond what Opentrons Protocol Designer can express and are
the primary reason the OpenOT2 designer IR exists as its own representation.

This module provides:

* **Builder functions** — convenient constructors that produce correctly
  parameterised :class:`Node` objects.
* **Artifact / dataflow helpers** — functions that describe how nodes
  reference each other's outputs (e.g. ``predict`` consumes an image
  artifact produced by ``capture``).
* **Workflow builder** — a small helper to construct a
  capture -> predict -> branch sub-graph in one call.
"""

from __future__ import annotations

from typing import Any, Optional

from openot2.designer.ir import Edge, Node, NodeKind, ProtocolIR


# ---------------------------------------------------------------------------
# Artifact key conventions
# ---------------------------------------------------------------------------

def capture_artifact_key(node_id: str) -> str:
    """Return the canonical artifact key for a capture node's output.

    Convention: ``"<node_id>.image"``
    """
    return f"{node_id}.image"


def predict_result_key(node_id: str) -> str:
    """Return the canonical result key for a predict node's output.

    Convention: ``"<node_id>.result"``
    """
    return f"{node_id}.result"


# ---------------------------------------------------------------------------
# Node builders
# ---------------------------------------------------------------------------

def make_capture(
    *,
    node_id: Optional[str] = None,
    label: str = "",
    camera_id: Any = 0,
    artifact_key: Optional[str] = None,
    extra_params: Optional[dict[str, Any]] = None,
) -> Node:
    """Create a capture node.

    Parameters
    ----------
    node_id:
        Explicit node id.  Auto-generated if omitted.
    label:
        Human-readable label.
    camera_id:
        Camera identifier passed to the capture handler.
    artifact_key:
        Key under which the captured image will be stored in the protocol's
        artifact namespace.  Defaults to ``"<node_id>.image"``.
    extra_params:
        Additional handler parameters merged into ``params``.
    """
    params: dict[str, Any] = {"camera_id": camera_id}
    if extra_params:
        params.update(extra_params)

    node = Node(
        kind=NodeKind.capture,
        label=label or "Capture image",
        params=params,
    )
    if node_id is not None:
        node.id = node_id

    # Set artifact key (needs node.id to be resolved first)
    node.params["artifact_key"] = artifact_key or capture_artifact_key(node.id)
    return node


def make_predict(
    *,
    node_id: Optional[str] = None,
    label: str = "",
    model: str = "",
    source_artifact: Optional[str] = None,
    result_key: Optional[str] = None,
    threshold: Optional[float] = None,
    extra_params: Optional[dict[str, Any]] = None,
) -> Node:
    """Create a predict node.

    Parameters
    ----------
    node_id:
        Explicit node id.  Auto-generated if omitted.
    label:
        Human-readable label.
    model:
        Model identifier or path (e.g. ``"tip_detector_v2"``).
    source_artifact:
        Artifact key of the image to run prediction on.  Typically the
        output of a preceding ``capture`` node.
    result_key:
        Key under which the prediction result is stored.  Defaults to
        ``"<node_id>.result"``.
    threshold:
        Optional confidence threshold for the prediction.
    extra_params:
        Additional handler parameters.
    """
    params: dict[str, Any] = {"model": model}
    if source_artifact:
        params["source_artifact"] = source_artifact
    if threshold is not None:
        params["threshold"] = threshold
    if extra_params:
        params.update(extra_params)

    node = Node(
        kind=NodeKind.predict,
        label=label or f"Predict ({model})" if model else "Predict",
        params=params,
    )
    if node_id is not None:
        node.id = node_id

    node.params["result_key"] = result_key or predict_result_key(node.id)
    return node


def make_branch(
    *,
    node_id: Optional[str] = None,
    label: str = "",
    condition: str,
    on_true: str,
    on_false: str,
    extra_params: Optional[dict[str, Any]] = None,
) -> Node:
    """Create a branch (conditional) node.

    Parameters
    ----------
    node_id:
        Explicit node id.  Auto-generated if omitted.
    label:
        Human-readable label.
    condition:
        The condition expression evaluated at runtime.  Typically a
        reference to a predict result key, e.g.
        ``"predict_qc.result.pass"``.
    on_true:
        Node id to transition to when the condition is truthy.
    on_false:
        Node id to transition to when the condition is falsy.
    extra_params:
        Additional handler parameters.
    """
    params: dict[str, Any] = {"condition": condition}
    if extra_params:
        params.update(extra_params)

    node = Node(
        kind=NodeKind.branch,
        label=label or f"Branch on {condition}",
        params=params,
        on_true=on_true,
        on_false=on_false,
    )
    if node_id is not None:
        node.id = node_id

    return node


# ---------------------------------------------------------------------------
# Dataflow helpers
# ---------------------------------------------------------------------------

def link_capture_to_predict(capture_node: Node, predict_node: Node) -> None:
    """Wire a capture node's artifact into a predict node's source.

    Sets ``predict_node.params["source_artifact"]`` to the capture node's
    artifact key, establishing a clear dataflow dependency.
    """
    artifact = capture_node.params.get(
        "artifact_key", capture_artifact_key(capture_node.id)
    )
    predict_node.params["source_artifact"] = artifact


def link_predict_to_branch(predict_node: Node, branch_node: Node) -> None:
    """Wire a predict node's result key as the branch condition.

    Sets ``branch_node.params["condition"]`` to the predict node's
    result key.
    """
    result = predict_node.params.get(
        "result_key", predict_result_key(predict_node.id)
    )
    branch_node.params["condition"] = result


def get_dataflow(protocol: ProtocolIR) -> list[dict[str, Any]]:
    """Extract artifact dataflow edges from the protocol.

    Returns a list of dicts ``{"producer", "consumer", "artifact"}``
    describing which nodes produce artifacts consumed by other nodes.
    """
    # Build an index of artifact_key -> producer node id
    producers: dict[str, str] = {}
    for node in protocol.nodes:
        if "artifact_key" in node.params:
            producers[node.params["artifact_key"]] = node.id
        if "result_key" in node.params:
            producers[node.params["result_key"]] = node.id

    # Find consumers
    flows: list[dict[str, Any]] = []
    for node in protocol.nodes:
        for key in ("source_artifact", "condition"):
            ref = node.params.get(key)
            if ref and ref in producers:
                flows.append({
                    "producer": producers[ref],
                    "consumer": node.id,
                    "artifact": ref,
                })
    return flows


# ---------------------------------------------------------------------------
# Workflow builder
# ---------------------------------------------------------------------------

def build_capture_predict_branch(
    *,
    capture_id: str = "capture",
    predict_id: str = "predict",
    branch_id: str = "branch",
    model: str,
    on_true: str,
    on_false: str,
    camera_id: Any = 0,
    threshold: Optional[float] = None,
) -> tuple[Node, Node, Node]:
    """Build a linked capture -> predict -> branch sub-graph.

    Returns a tuple of ``(capture_node, predict_node, branch_node)`` with
    artifact references already wired together.  The caller should append
    these nodes to a :class:`ProtocolIR` and set ``capture_node.next`` /
    ``predict_node.next`` if needed for graph connectivity.

    Parameters
    ----------
    capture_id, predict_id, branch_id:
        Explicit node ids.
    model:
        Model identifier for the predict node.
    on_true, on_false:
        Branch targets.
    camera_id:
        Camera identifier for capture.
    threshold:
        Optional prediction confidence threshold.
    """
    cap = make_capture(node_id=capture_id, camera_id=camera_id)
    pred = make_predict(
        node_id=predict_id,
        model=model,
        threshold=threshold,
    )
    br = make_branch(
        node_id=branch_id,
        condition="",  # will be set by link helper
        on_true=on_true,
        on_false=on_false,
    )

    # Wire dataflow
    link_capture_to_predict(cap, pred)
    link_predict_to_branch(pred, br)

    # Wire graph edges
    cap.next = predict_id
    pred.next = branch_id

    return cap, pred, br
