"""Validation rules for the OpenOT2 designer IR.

``validate()`` returns a list of human-readable error strings.  An empty list
means the protocol is valid.
"""

from __future__ import annotations

from typing import Any

from openot2.designer.ir import Edge, Node, NodeKind, ProtocolIR


# ---------------------------------------------------------------------------
# Per-kind required-param specs
# ---------------------------------------------------------------------------

# Maps NodeKind → set of required param keys.
_REQUIRED_PARAMS: dict[NodeKind, set[str]] = {
    NodeKind.transfer: {"pipette_id", "source_labware_id", "source_wells",
                        "dest_labware_id", "dest_wells", "volume"},
    NodeKind.mix: {"pipette_id", "labware_id", "wells", "volume", "cycles"},
    NodeKind.pause: set(),  # message is optional
    NodeKind.delay: {"seconds"},
    NodeKind.module_action: {"module_id", "action"},
    NodeKind.move_labware: {"labware_id", "new_slot"},
    NodeKind.capture: set(),  # all params optional
    NodeKind.predict: set(),  # all params optional
    NodeKind.branch: {"condition"},
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate(protocol: ProtocolIR) -> list[str]:
    """Validate a :class:`ProtocolIR` and return a list of errors.

    Checks performed:

    1. Node ids are unique.
    2. Node edge targets reference existing nodes.
    3. Explicit edges reference existing nodes.
    4. Branch nodes have ``on_true`` and ``on_false`` set.
    5. Required params are present for each node kind.
    6. Deck references in node params point to declared deck entries.
    7. Graph reachability (all nodes reachable from the start when graph).
    """
    errors: list[str] = []
    _check_unique_node_ids(protocol, errors)
    node_ids = set(protocol.node_ids())
    _check_node_edges(protocol, node_ids, errors)
    _check_explicit_edges(protocol, node_ids, errors)
    _check_branch_nodes(protocol, errors)
    _check_required_params(protocol, errors)
    _check_deck_references(protocol, errors)
    if protocol.is_graph():
        _check_graph_reachability(protocol, node_ids, errors)
    return errors


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def _check_unique_node_ids(protocol: ProtocolIR, errors: list[str]) -> None:
    seen: set[str] = set()
    for node in protocol.nodes:
        if node.id in seen:
            errors.append(f"Duplicate node id: '{node.id}'")
        seen.add(node.id)


def _check_node_edges(
    protocol: ProtocolIR, node_ids: set[str], errors: list[str]
) -> None:
    for node in protocol.nodes:
        for field in ("next", "on_true", "on_false"):
            target = getattr(node, field)
            if target is not None and target not in node_ids:
                errors.append(
                    f"Node '{node.id}' ({node.kind.value}): "
                    f"{field} references unknown node '{target}'"
                )


def _check_explicit_edges(
    protocol: ProtocolIR, node_ids: set[str], errors: list[str]
) -> None:
    for edge in protocol.edges:
        if edge.source not in node_ids:
            errors.append(
                f"Edge source '{edge.source}' not found in nodes"
            )
        if edge.target not in node_ids:
            errors.append(
                f"Edge target '{edge.target}' not found in nodes"
            )


def _check_branch_nodes(protocol: ProtocolIR, errors: list[str]) -> None:
    for node in protocol.nodes:
        if node.kind == NodeKind.branch:
            if node.on_true is None:
                errors.append(
                    f"Branch node '{node.id}': missing on_true target"
                )
            if node.on_false is None:
                errors.append(
                    f"Branch node '{node.id}': missing on_false target"
                )


def _check_required_params(protocol: ProtocolIR, errors: list[str]) -> None:
    for node in protocol.nodes:
        required = _REQUIRED_PARAMS.get(node.kind, set())
        missing = required - set(node.params.keys())
        if missing:
            errors.append(
                f"Node '{node.id}' ({node.kind.value}): "
                f"missing required params: {sorted(missing)}"
            )


def _check_deck_references(protocol: ProtocolIR, errors: list[str]) -> None:
    """Check that labware/pipette/module ids in node params exist on deck."""
    labware_ids = {lw.id for lw in protocol.deck.labware}
    pipette_ids = {p.id for p in protocol.deck.pipettes}
    module_ids = {m.id for m in protocol.deck.modules}

    for node in protocol.nodes:
        params = node.params

        # Pipette references
        if "pipette_id" in params and params["pipette_id"] not in pipette_ids:
            errors.append(
                f"Node '{node.id}' ({node.kind.value}): "
                f"pipette_id '{params['pipette_id']}' not found on deck"
            )

        # Labware references
        for key in ("source_labware_id", "dest_labware_id", "labware_id"):
            if key in params and params[key] not in labware_ids:
                errors.append(
                    f"Node '{node.id}' ({node.kind.value}): "
                    f"{key} '{params[key]}' not found on deck"
                )

        # Module references
        if "module_id" in params and params["module_id"] not in module_ids:
            errors.append(
                f"Node '{node.id}' ({node.kind.value}): "
                f"module_id '{params['module_id']}' not found on deck"
            )


def _check_graph_reachability(
    protocol: ProtocolIR, node_ids: set[str], errors: list[str]
) -> None:
    """Warn about unreachable nodes when graph edges are present."""
    if not protocol.nodes:
        return

    # Build adjacency from node-level edges and explicit edges
    adj: dict[str, set[str]] = {nid: set() for nid in node_ids}
    for node in protocol.nodes:
        for field in ("next", "on_true", "on_false"):
            target = getattr(node, field)
            if target is not None and target in node_ids:
                adj[node.id].add(target)
    for edge in protocol.edges:
        if edge.source in node_ids and edge.target in node_ids:
            adj[edge.source].add(edge.target)

    # BFS from first node
    start = protocol.nodes[0].id
    visited: set[str] = set()
    queue = [start]
    while queue:
        current = queue.pop(0)
        if current in visited:
            continue
        visited.add(current)
        queue.extend(adj.get(current, set()))

    unreachable = node_ids - visited
    for nid in sorted(unreachable):
        node = protocol.node_by_id(nid)
        kind = node.kind.value if node else "unknown"
        errors.append(f"Node '{nid}' ({kind}) is unreachable from start")
