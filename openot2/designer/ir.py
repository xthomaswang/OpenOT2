"""Canonical intermediate representation for OpenOT2 protocol designs.

The IR is the single source of truth for all protocol representations inside
the designer.  External formats (Opentrons PD JSON, PD Python, manual
construction) are *imported into* the IR; execution formats (``TaskRun``,
``TaskGraph``) are *compiled from* the IR.

Node kinds fall into three tiers:

* **Liquid-handling** — transfer, mix, pause, delay, module_action, move_labware
* **Analysis** — capture, predict
* **Control-flow** — branch (conditional edge routing)
"""

from __future__ import annotations

import enum
import uuid
from typing import Any, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _uuid() -> str:
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class NodeKind(str, enum.Enum):
    """Supported node types in the designer IR."""

    # Liquid handling
    transfer = "transfer"
    mix = "mix"
    pause = "pause"
    delay = "delay"
    module_action = "module_action"
    move_labware = "move_labware"

    # Analysis / capture
    capture = "capture"
    predict = "predict"

    # Control flow
    branch = "branch"


# ---------------------------------------------------------------------------
# Deck setup
# ---------------------------------------------------------------------------

class LabwareEntry(BaseModel):
    """A single piece of labware placed on the deck."""

    id: str
    slot: str
    labware_type: str
    display_name: str = ""


class PipetteEntry(BaseModel):
    """A pipette mounted on the robot."""

    id: str
    mount: str  # "left" | "right"
    pipette_type: str


class ModuleEntry(BaseModel):
    """A hardware module (temperature, magnetic, thermocycler, etc.)."""

    id: str
    slot: str
    module_type: str
    display_name: str = ""


class LiquidEntry(BaseModel):
    """A liquid definition (dye, reagent, buffer, etc.)."""

    id: str
    name: str
    color: Optional[str] = None
    description: str = ""


class DeckSetup(BaseModel):
    """Complete deck configuration: labware, pipettes, modules, liquids."""

    labware: list[LabwareEntry] = Field(default_factory=list)
    pipettes: list[PipetteEntry] = Field(default_factory=list)
    modules: list[ModuleEntry] = Field(default_factory=list)
    liquids: list[LiquidEntry] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

class Node(BaseModel):
    """A single step / operation in the protocol.

    Parameters are stored in a flat ``params`` dict whose schema depends on
    ``kind``.  See ``openot2.designer.validator`` for per-kind validation.

    Graph edges are optional.  A protocol with no edges on any node is
    treated as a linear sequence (node order = list order).  When edges are
    present the protocol becomes a graph.

    Attributes
    ----------
    id:
        Unique node identifier.  Used by edges and branch targets.
    kind:
        The type of operation (see :class:`NodeKind`).
    label:
        Human-readable description shown in the UI.
    params:
        Kind-specific parameters.
    next:
        Default successor node id for graph protocols.
    on_true / on_false:
        Conditional successors for ``branch`` nodes.
    """

    id: str = Field(default_factory=_uuid)
    kind: NodeKind
    label: str = ""
    params: dict[str, Any] = Field(default_factory=dict)

    # Graph edges (only used when protocol has graph semantics)
    next: Optional[str] = None
    on_true: Optional[str] = None
    on_false: Optional[str] = None


# ---------------------------------------------------------------------------
# Edges (explicit, for visualization or complex topologies)
# ---------------------------------------------------------------------------

class Edge(BaseModel):
    """An explicit directed edge between two nodes.

    Edges complement the implicit ``next``/``on_true``/``on_false`` fields on
    :class:`Node`.  They are primarily used for visualization and for
    topologies that cannot be expressed via node-level fields alone.
    """

    source: str
    target: str
    condition: Optional[str] = None  # "true", "false", or None for default
    label: str = ""


# ---------------------------------------------------------------------------
# Protocol IR (top-level)
# ---------------------------------------------------------------------------

class ProtocolIR(BaseModel):
    """Canonical intermediate representation of an OpenOT2 protocol.

    This is the single object that importers produce and compilers consume.
    It is a strict superset of what Opentrons Protocol Designer can represent,
    adding capture / predict / branch node kinds plus optional graph edges.
    """

    id: str = Field(default_factory=_uuid)
    name: str
    description: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    deck: DeckSetup = Field(default_factory=DeckSetup)
    nodes: list[Node] = Field(default_factory=list)
    edges: list[Edge] = Field(default_factory=list)

    # Optional protocol-level variables / artifacts
    variables: dict[str, Any] = Field(default_factory=dict)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def node_by_id(self, node_id: str) -> Optional[Node]:
        """Look up a node by its id, or return ``None``."""
        for n in self.nodes:
            if n.id == node_id:
                return n
        return None

    def node_ids(self) -> list[str]:
        """Return all node ids in order."""
        return [n.id for n in self.nodes]

    def is_graph(self) -> bool:
        """Return True if the protocol uses graph semantics.

        A protocol is considered a graph if any node has explicit edge
        fields set or if the ``edges`` list is non-empty.
        """
        if self.edges:
            return True
        return any(
            n.next is not None or n.on_true is not None or n.on_false is not None
            for n in self.nodes
        )

    def has_branch_nodes(self) -> bool:
        """Return True if any node is a branch node."""
        return any(n.kind == NodeKind.branch for n in self.nodes)
