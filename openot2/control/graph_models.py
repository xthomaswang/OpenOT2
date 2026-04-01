"""DAG-based task graph models for conditional protocol execution.

A :class:`TaskGraph` represents a protocol as a directed graph of nodes
(steps) connected by edges.  Each node can branch based on the boolean
``result`` field in its handler output — enabling conditional logic like
retry loops, quality checks, and ML-driven decisions.

Example graph (JSON)::

    {
        "name": "Transfer with QC",
        "start": "pickup",
        "nodes": {
            "pickup":  {"kind": "pick_up_tip", "params": {"slot": "10", "well": "A1"},
                        "next": "aspirate"},
            "aspirate":{"kind": "aspirate",    "params": {"slot": "7", "well": "A1", "volume": 100},
                        "next": "dispense"},
            "dispense":{"kind": "dispense",    "params": {"slot": "1", "well": "A1", "volume": 100},
                        "next": "capture"},
            "capture": {"kind": "capture",     "params": {"camera_id": 0, "label": "qc"},
                        "next": "check"},
            "check":   {"kind": "predict",     "params": {"model_path": "model.pt", "min_detections": 1},
                        "on_true": "drop", "on_false": "retry"},
            "retry":   {"kind": "aspirate",    "params": {"slot": "7", "well": "A1", "volume": 100},
                        "next": "dispense"},
            "drop":    {"kind": "drop_tip",    "params": {},
                        "next": null}
        }
    }
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


class NodeStatus(str, enum.Enum):
    """Execution status of a graph node."""

    pending = "pending"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"
    skipped = "skipped"


class GraphNode(BaseModel):
    """A single node in a :class:`TaskGraph`.

    Attributes
    ----------
    kind:
        Handler kind (e.g. ``"aspirate"``, ``"predict"``).
    params:
        Parameters passed to the handler.
    next:
        Default next node ID.  Used when the handler output has no
        ``result`` field, or when ``on_true``/``on_false`` are not set.
    on_true:
        Next node if handler output ``result`` is truthy.
    on_false:
        Next node if handler output ``result`` is falsy.
    name:
        Human-readable label (optional, defaults to node ID).
    max_visits:
        Maximum times this node can be executed (prevents infinite loops).
        Default 10.
    """

    kind: str
    params: dict[str, Any] = Field(default_factory=dict)
    next: Optional[str] = None
    on_true: Optional[str] = None
    on_false: Optional[str] = None
    name: Optional[str] = None
    max_visits: int = 10

    # Runtime state (populated during execution)
    status: NodeStatus = NodeStatus.pending
    visit_count: int = 0
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    output: Optional[dict[str, Any]] = None
    error: Optional[str] = None


class GraphStatus(str, enum.Enum):
    """Lifecycle status of a task graph execution."""

    draft = "draft"
    running = "running"
    completed = "completed"
    failed = "failed"
    aborted = "aborted"


class TaskGraph(BaseModel):
    """A protocol represented as a directed acyclic graph (with optional cycles).

    Nodes are keyed by string IDs.  Execution starts at ``start`` and
    follows edges (``next``, ``on_true``, ``on_false``) until a node
    with no outgoing edge is reached or an error occurs.
    """

    id: str = Field(default_factory=_uuid)
    name: str
    start: str
    nodes: dict[str, GraphNode]
    status: GraphStatus = GraphStatus.draft
    current_node: Optional[str] = None
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)
    metadata: dict[str, Any] = Field(default_factory=dict)
    execution_log: list[dict[str, Any]] = Field(default_factory=list)

    def validate_graph(self) -> list[str]:
        """Return a list of validation errors (empty = valid)."""
        errors = []
        if self.start not in self.nodes:
            errors.append(f"Start node '{self.start}' not found in nodes")
        for nid, node in self.nodes.items():
            for edge in [node.next, node.on_true, node.on_false]:
                if edge is not None and edge not in self.nodes:
                    errors.append(f"Node '{nid}' references unknown node '{edge}'")
        return errors
