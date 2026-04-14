"""Helpers for the canonical task state shape.

A task's state is always a plain ``dict`` so that plugins are free to choose
their own schema.  This module provides factory and accessor helpers that
make it easy to create states that follow the recommended shape::

    {
        "phase": str,          # e.g. "idle", "running", "calibrating", "done"
        "iteration": int,      # current iteration counter
        "history": [...],      # per-iteration result summaries
        "metrics": {...},      # accumulated scalar metrics
        "artifacts": {...},    # paths / URIs to produced artefacts
        "cache": {...},        # transient data (model state, etc.)
        "terminal": bool,      # True once the task cannot make further progress
    }
"""

from __future__ import annotations

from typing import Any


def make_state(
    *,
    phase: str = "idle",
    iteration: int = 0,
    history: list[dict[str, Any]] | None = None,
    metrics: dict[str, Any] | None = None,
    artifacts: dict[str, Any] | None = None,
    cache: dict[str, Any] | None = None,
    terminal: bool = False,
    **extra: Any,
) -> dict[str, Any]:
    """Create a new task state dict with sensible defaults."""
    state: dict[str, Any] = {
        "phase": phase,
        "iteration": iteration,
        "history": history if history is not None else [],
        "metrics": metrics if metrics is not None else {},
        "artifacts": artifacts if artifacts is not None else {},
        "cache": cache if cache is not None else {},
        "terminal": terminal,
    }
    state.update(extra)
    return state


def is_terminal(state: dict[str, Any]) -> bool:
    """Return ``True`` if the state is marked as terminal."""
    return bool(state.get("terminal", False))


def current_phase(state: dict[str, Any]) -> str:
    """Return the current phase string (defaults to ``"idle"``)."""
    return str(state.get("phase", "idle"))


def current_iteration(state: dict[str, Any]) -> int:
    """Return the current iteration counter (defaults to ``0``)."""
    return int(state.get("iteration", 0))
