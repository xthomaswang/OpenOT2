"""Visualization projections for OpenOT2 protocol IR.

Two projection modes:

* **Timeline** — a flat ordered list of step summaries for linear protocols.
* **Graph** — nodes + edges dicts suitable for graph renderers (e.g. Mermaid,
  Cytoscape, vis.js).

Both projections produce plain Python dicts / lists so they can be serialised
to JSON for any front-end without coupling to a specific UI framework.
"""

from __future__ import annotations

from typing import Any

from openot2.designer.ir import Edge, Node, NodeKind, ProtocolIR


# ---------------------------------------------------------------------------
# Node summary helpers
# ---------------------------------------------------------------------------

_KIND_VERB: dict[NodeKind, str] = {
    NodeKind.transfer: "Transfer",
    NodeKind.mix: "Mix",
    NodeKind.pause: "Pause",
    NodeKind.delay: "Delay",
    NodeKind.module_action: "Module action",
    NodeKind.move_labware: "Move labware",
    NodeKind.capture: "Capture image",
    NodeKind.predict: "Run prediction",
    NodeKind.branch: "Branch",
}


def summarize_node(node: Node) -> str:
    """Return a one-line human-readable summary of a node.

    The summary combines the verb with the most salient parameters for each
    node kind so that protocol reviews are scannable at a glance.
    """
    verb = _KIND_VERB.get(node.kind, node.kind.value.capitalize())
    p = node.params

    if node.label:
        return f"{verb}: {node.label}"

    if node.kind == NodeKind.transfer:
        vol = p.get("volume", "?")
        src = p.get("source_wells", "?")
        dst = p.get("dest_wells", "?")
        return f"{verb} {vol} uL from {src} -> {dst}"

    if node.kind == NodeKind.mix:
        vol = p.get("volume", "?")
        cycles = p.get("cycles", "?")
        wells = p.get("wells", "?")
        return f"{verb} {vol} uL x{cycles} in {wells}"

    if node.kind == NodeKind.pause:
        msg = p.get("message", "")
        return f"{verb}: {msg}" if msg else verb

    if node.kind == NodeKind.delay:
        secs = p.get("seconds", "?")
        return f"{verb} {secs}s"

    if node.kind == NodeKind.module_action:
        action = p.get("action", "?")
        mod = p.get("module_id", "?")
        return f"{verb} '{action}' on {mod}"

    if node.kind == NodeKind.move_labware:
        lw = p.get("labware_id", "?")
        slot = p.get("new_slot", "?")
        return f"{verb} {lw} -> slot {slot}"

    if node.kind == NodeKind.capture:
        cam = p.get("camera_id", "default")
        artifact = p.get("artifact_key", "")
        extra = f" -> {artifact}" if artifact else ""
        return f"{verb} (camera={cam}){extra}"

    if node.kind == NodeKind.predict:
        model = p.get("model", p.get("model_path", "?"))
        source = p.get("source_artifact", "")
        extra = f" on {source}" if source else ""
        return f"{verb} model={model}{extra}"

    if node.kind == NodeKind.branch:
        cond = p.get("condition", "?")
        return f"{verb} on {cond}"

    return verb


# ---------------------------------------------------------------------------
# Timeline projection (linear protocols)
# ---------------------------------------------------------------------------

def _timeline_entry(node: Node, index: int) -> dict[str, Any]:
    """Build a single timeline entry dict."""
    return {
        "index": index,
        "id": node.id,
        "kind": node.kind.value,
        "label": node.label,
        "summary": summarize_node(node),
    }


def timeline(protocol: ProtocolIR) -> list[dict[str, Any]]:
    """Return an ordered timeline of step summaries.

    Each entry is a dict with keys: ``index``, ``id``, ``kind``, ``label``,
    ``summary``.  Best suited for protocols without branching.
    """
    return [_timeline_entry(n, i) for i, n in enumerate(protocol.nodes)]


# ---------------------------------------------------------------------------
# Graph projection (branched / DAG protocols)
# ---------------------------------------------------------------------------

def _graph_node(node: Node) -> dict[str, Any]:
    """Build a graph-node dict for visualization."""
    entry: dict[str, Any] = {
        "id": node.id,
        "kind": node.kind.value,
        "label": node.label,
        "summary": summarize_node(node),
    }
    if node.kind == NodeKind.branch:
        entry["is_decision"] = True
    return entry


def _collect_edges(protocol: ProtocolIR) -> list[dict[str, Any]]:
    """Collect all edges — implicit (node-level) + explicit."""
    edges: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str | None]] = set()

    # Implicit edges from node fields
    for i, node in enumerate(protocol.nodes):
        if node.next is not None:
            key = (node.id, node.next, None)
            if key not in seen:
                edges.append({
                    "source": node.id,
                    "target": node.next,
                    "condition": None,
                    "label": "",
                })
                seen.add(key)

        if node.on_true is not None:
            key = (node.id, node.on_true, "true")
            if key not in seen:
                edges.append({
                    "source": node.id,
                    "target": node.on_true,
                    "condition": "true",
                    "label": "yes",
                })
                seen.add(key)

        if node.on_false is not None:
            key = (node.id, node.on_false, "false")
            if key not in seen:
                edges.append({
                    "source": node.id,
                    "target": node.on_false,
                    "condition": "false",
                    "label": "no",
                })
                seen.add(key)

        # Linear fallback: chain to next node if no explicit edges
        if (
            node.next is None
            and node.on_true is None
            and node.on_false is None
            and i < len(protocol.nodes) - 1
        ):
            nxt = protocol.nodes[i + 1].id
            key = (node.id, nxt, None)
            if key not in seen:
                edges.append({
                    "source": node.id,
                    "target": nxt,
                    "condition": None,
                    "label": "",
                })
                seen.add(key)

    # Explicit edges
    for edge in protocol.edges:
        key = (edge.source, edge.target, edge.condition)
        if key not in seen:
            edges.append({
                "source": edge.source,
                "target": edge.target,
                "condition": edge.condition,
                "label": edge.label,
            })
            seen.add(key)

    return edges


def graph(protocol: ProtocolIR) -> dict[str, Any]:
    """Return a graph projection with ``nodes`` and ``edges`` lists.

    Suitable for rendering with any graph library.  Includes both implicit
    edges (from node-level ``next``/``on_true``/``on_false``) and explicit
    :class:`Edge` objects, deduplicated.
    """
    return {
        "protocol_id": protocol.id,
        "protocol_name": protocol.name,
        "nodes": [_graph_node(n) for n in protocol.nodes],
        "edges": _collect_edges(protocol),
    }


# ---------------------------------------------------------------------------
# Mermaid export (text-based graph rendering)
# ---------------------------------------------------------------------------

def to_mermaid(protocol: ProtocolIR) -> str:
    """Render the protocol as a Mermaid flowchart string.

    Branch nodes are rendered as diamond ``{text}`` shapes; all other nodes
    use rounded rectangles ``(text)``.
    """
    lines: list[str] = ["graph TD"]

    # Node declarations
    for node in protocol.nodes:
        summary = summarize_node(node).replace('"', "'")
        if node.kind == NodeKind.branch:
            lines.append(f'    {node.id}{{"{summary}"}}')
        else:
            lines.append(f'    {node.id}("{summary}")')

    # Edges
    g = graph(protocol)
    for edge in g["edges"]:
        src = edge["source"]
        tgt = edge["target"]
        lbl = edge["label"]
        if lbl:
            lines.append(f"    {src} -->|{lbl}| {tgt}")
        else:
            lines.append(f"    {src} --> {tgt}")

    return "\n".join(lines)
