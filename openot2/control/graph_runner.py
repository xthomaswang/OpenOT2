"""DAG-based task graph runner with conditional branching.

:class:`GraphRunner` walks a :class:`TaskGraph` from its start node,
executing each node's handler and following edges based on the handler's
boolean ``result`` output.

Usage::

    from openot2.control.graph_runner import GraphRunner
    from openot2.control.graph_models import TaskGraph

    graph = TaskGraph(name="My Protocol", start="step1", nodes={...})
    runner = GraphRunner(handlers={...})
    result = runner.run(graph)
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Callable

from openot2.control.graph_models import (
    GraphNode,
    GraphStatus,
    NodeStatus,
    TaskGraph,
)

logger = logging.getLogger("openot2.control.graph_runner")


class GraphRunner:
    """Execute a :class:`TaskGraph` by walking nodes and following edges.

    Parameters
    ----------
    handlers:
        Mapping of ``kind`` string to callable handler.
        Each handler receives ``(node, context)`` and returns a dict.
        If the dict contains a ``"result"`` key (bool), it determines
        which branch to follow (``on_true`` / ``on_false``).
    on_node_start:
        Optional callback ``(graph, node_id, node)`` called before each node.
    on_node_done:
        Optional callback ``(graph, node_id, node)`` called after each node.
    """

    def __init__(
        self,
        handlers: dict[str, Callable] | None = None,
        on_node_start: Callable | None = None,
        on_node_done: Callable | None = None,
    ) -> None:
        self._handlers: dict[str, Callable] = dict(handlers) if handlers else {}
        self._on_node_start = on_node_start
        self._on_node_done = on_node_done

    def register_handler(self, kind: str, handler: Callable) -> None:
        """Register a handler for a node kind."""
        self._handlers[kind] = handler

    def run(self, graph: TaskGraph, context: Any = None) -> TaskGraph:
        """Execute the graph from start to completion.

        Returns the graph with updated node statuses and execution log.
        """
        errors = graph.validate_graph()
        if errors:
            raise ValueError(f"Invalid graph: {'; '.join(errors)}")

        graph.status = GraphStatus.running
        graph.current_node = graph.start
        graph.updated_at = _now()

        logger.info("Graph '%s' started at node '%s'", graph.name, graph.start)

        while graph.current_node is not None:
            node_id = graph.current_node
            node = graph.nodes[node_id]

            # Check max visits (prevent infinite loops)
            if node.visit_count >= node.max_visits:
                node.status = NodeStatus.failed
                node.error = f"Max visits ({node.max_visits}) exceeded"
                graph.status = GraphStatus.failed
                graph.updated_at = _now()
                self._log(graph, node_id, "max_visits_exceeded", node.error)
                logger.error("Node '%s' exceeded max visits", node_id)
                return graph

            # Execute the node
            try:
                next_id = self._execute_node(graph, node_id, node, context)
            except Exception as exc:
                node.status = NodeStatus.failed
                node.finished_at = _now()
                node.error = str(exc)
                if node.started_at:
                    node.duration_seconds = (
                        node.finished_at - node.started_at
                    ).total_seconds()
                graph.status = GraphStatus.failed
                graph.updated_at = _now()
                self._log(graph, node_id, "node_failed", str(exc))
                logger.error("Node '%s' failed: %s", node_id, exc)
                return graph

            graph.current_node = next_id
            graph.updated_at = _now()

        # All done
        graph.status = GraphStatus.completed
        graph.current_node = None
        graph.updated_at = _now()
        logger.info("Graph '%s' completed", graph.name)
        return graph

    def _execute_node(
        self,
        graph: TaskGraph,
        node_id: str,
        node: GraphNode,
        context: Any,
    ) -> str | None:
        """Execute a single node and return the next node ID (or None)."""
        handler = self._handlers.get(node.kind)
        if handler is None:
            raise RuntimeError(f"No handler for kind '{node.kind}'")

        # Mark running
        node.status = NodeStatus.running
        node.started_at = _now()
        node.visit_count += 1
        display_name = node.name or node_id

        if self._on_node_start:
            self._on_node_start(graph, node_id, node)

        self._log(graph, node_id, "node_started", f"{display_name} ({node.kind})")
        logger.info("Executing node '%s' (%s) [visit %d]", node_id, node.kind, node.visit_count)

        # Run handler — create a minimal step-like object for compatibility
        # with existing handlers that expect RunStep
        step_proxy = _StepProxy(
            name=display_name,
            kind=node.kind,
            params=node.params,
        )
        result = handler(step_proxy, context)

        # Mark succeeded
        node.status = NodeStatus.succeeded
        node.finished_at = _now()
        node.duration_seconds = (node.finished_at - node.started_at).total_seconds()
        node.output = result if isinstance(result, dict) else {}

        if self._on_node_done:
            self._on_node_done(graph, node_id, node)

        self._log(
            graph, node_id, "node_succeeded",
            f"{display_name} done in {node.duration_seconds:.2f}s",
            {"output": node.output},
        )

        # Determine next node
        if node.on_true is not None or node.on_false is not None:
            # Branching node — check result
            bool_result = bool(result.get("result", True)) if isinstance(result, dict) else True
            next_id = node.on_true if bool_result else node.on_false
            branch = "on_true" if bool_result else "on_false"
            self._log(
                graph, node_id, "branch",
                f"result={bool_result} → {branch} → {next_id}",
            )
            logger.info("Branch: result=%s → %s → '%s'", bool_result, branch, next_id)
            return next_id
        else:
            return node.next

    def _log(
        self,
        graph: TaskGraph,
        node_id: str,
        event_type: str,
        message: str,
        payload: dict | None = None,
    ) -> None:
        """Append an entry to the graph's execution log."""
        graph.execution_log.append({
            "id": uuid.uuid4().hex[:12],
            "node_id": node_id,
            "type": event_type,
            "message": message,
            "timestamp": _now().isoformat(),
            "payload": payload,
        })


class _StepProxy:
    """Minimal proxy that quacks like a RunStep for handler compatibility."""

    __slots__ = ("name", "kind", "params", "id")

    def __init__(self, name: str, kind: str, params: dict) -> None:
        self.name = name
        self.kind = kind
        self.params = params
        self.id = uuid.uuid4().hex


def _now() -> datetime:
    return datetime.now(timezone.utc)
