"""Compile a :class:`ProtocolIR` into executable formats.

Two compilation targets are supported:

* **Linear** — produces a :class:`TaskRun` with ordered :class:`RunStep`
  objects.  Suitable for protocols without branching.

* **Graph** — produces a :class:`TaskGraph` with :class:`GraphNode` objects.
  Supports conditional branching via ``on_true`` / ``on_false`` edges.

Both compilers validate the IR before emitting; a :class:`CompileError` is
raised if validation fails.
"""

from __future__ import annotations

from typing import Any

from openot2.control.graph_models import GraphNode, TaskGraph
from openot2.control.models import RunStep, TaskRun
from openot2.designer.ir import Node, NodeKind, ProtocolIR
from openot2.designer.validator import validate


class CompileError(Exception):
    """Raised when the IR cannot be compiled due to validation errors."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__(
            f"Protocol IR has {len(errors)} validation error(s):\n"
            + "\n".join(f"  - {e}" for e in errors)
        )


# ---------------------------------------------------------------------------
# Linear compiler → TaskRun
# ---------------------------------------------------------------------------

def compile_linear(protocol: ProtocolIR) -> TaskRun:
    """Compile a linear protocol IR into a :class:`TaskRun`.

    Branch nodes are not allowed in a linear compile — they will produce a
    validation error.  Use :func:`compile_graph` for branching protocols.

    Parameters
    ----------
    protocol:
        A validated :class:`ProtocolIR` with no branch nodes.

    Returns
    -------
    TaskRun
        Ready for execution by :class:`TaskRunner`.

    Raises
    ------
    CompileError
        If the IR contains validation errors or branch nodes.
    """
    errors = validate(protocol)

    # Branch nodes are incompatible with linear execution
    if protocol.has_branch_nodes():
        errors.append(
            "Linear compile does not support branch nodes — "
            "use compile_graph() instead"
        )

    if errors:
        raise CompileError(errors)

    steps: list[RunStep] = []
    for i, node in enumerate(protocol.nodes):
        step = _node_to_run_step(node, index=i)
        steps.append(step)

    return TaskRun(
        name=protocol.name,
        steps=steps,
        metadata={
            "designer_protocol_id": protocol.id,
            "description": protocol.description,
            **protocol.metadata,
        },
    )


# ---------------------------------------------------------------------------
# Graph compiler → TaskGraph
# ---------------------------------------------------------------------------

def compile_graph(protocol: ProtocolIR) -> TaskGraph:
    """Compile a protocol IR into a :class:`TaskGraph`.

    All node kinds are supported, including branch nodes.  If the protocol
    has no explicit graph edges (purely linear), the compiler chains nodes
    sequentially via ``next`` pointers.

    Parameters
    ----------
    protocol:
        A validated :class:`ProtocolIR`.

    Returns
    -------
    TaskGraph
        Ready for execution by :class:`GraphRunner`.

    Raises
    ------
    CompileError
        If the IR contains validation errors.
    """
    errors = validate(protocol)
    if errors:
        raise CompileError(errors)

    if not protocol.nodes:
        raise CompileError(["Protocol has no nodes"])

    nodes: dict[str, GraphNode] = {}
    node_list = protocol.nodes

    for i, node in enumerate(node_list):
        graph_node = _node_to_graph_node(node)

        # If the node has no explicit edges and isn't the last node,
        # chain to the next node in list order (linear fallback).
        if (
            graph_node.next is None
            and graph_node.on_true is None
            and graph_node.on_false is None
            and i < len(node_list) - 1
        ):
            graph_node.next = node_list[i + 1].id

        nodes[node.id] = graph_node

    return TaskGraph(
        name=protocol.name,
        start=node_list[0].id,
        nodes=nodes,
        metadata={
            "designer_protocol_id": protocol.id,
            "description": protocol.description,
            **protocol.metadata,
        },
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _node_to_run_step(node: Node, index: int) -> RunStep:
    """Convert a designer IR node to a :class:`RunStep`."""
    return RunStep(
        key=node.id,
        name=node.label or f"{node.kind.value}_{index}",
        kind=node.kind.value,
        params=dict(node.params),
    )


def _node_to_graph_node(node: Node) -> GraphNode:
    """Convert a designer IR node to a :class:`GraphNode`."""
    return GraphNode(
        kind=node.kind.value,
        params=dict(node.params),
        name=node.label or None,
        next=node.next,
        on_true=node.on_true,
        on_false=node.on_false,
    )
