"""Canonical task plugin interfaces.

Every task plugin must satisfy :class:`TaskPlugin`.  Plugins that extend the
web shell should additionally provide a :class:`TaskWebExtension` via
:meth:`TaskPlugin.web_extension`.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class TaskWebExtension(Protocol):
    """Optional web-layer extensions provided by a task plugin.

    The generic web shell calls these hooks to inject task-specific routes,
    UI data, and status fields without knowing anything about the task's
    domain (wells, slots, colours, …).
    """

    def extra_routes(self) -> Any:
        """Return framework-specific route descriptors (e.g. Flask Blueprint)."""
        ...

    def ui_payload(self, config: Any, state: dict) -> dict:
        """Return JSON-serialisable data for the task's UI panel."""
        ...

    def extra_status(self, config: Any, state: dict) -> dict:
        """Return extra fields merged into the generic status response."""
        ...


@runtime_checkable
class TaskPlugin(Protocol):
    """The single interface every task must implement.

    The framework (runner, CLI, web shell) programmes against this protocol.
    Task-specific logic is entirely contained inside the implementing class.
    """

    name: str

    # -- configuration -------------------------------------------------------

    def load_config(self, path: str) -> Any:
        """Load and return a task-specific configuration from *path*."""
        ...

    def build_deck_config(self, config: Any) -> Any:
        """Return an OT-2 deck layout derived from *config*."""
        ...

    # -- state ---------------------------------------------------------------

    def initial_state(self, config: Any, mode: str) -> dict:
        """Create the initial task state dict for *mode* (e.g. ``"run"``)."""
        ...

    # -- planning & run construction -----------------------------------------

    def build_plan(self, config: Any, state: dict, mode: str) -> dict:
        """Return a high-level plan dict describing the full experiment."""
        ...

    def build_iteration_run(
        self,
        config: Any,
        state: dict,
        iteration: int,
        mode: str,
    ) -> Any:
        """Build a full :class:`~openot2.control.models.TaskRun` for one iteration.

        The returned run should include every step the task needs (liquid
        handling, wait, capture, analysis, modelling, …) using ``$ref``
        bindings where later steps depend on earlier outputs.
        """
        ...

    def build_calibration_run(self, config: Any, state: dict) -> Any:
        """Build a :class:`TaskRun` for the calibration procedure."""
        ...

    def build_tip_check_run(self, config: Any, state: dict) -> Any:
        """Build a :class:`TaskRun` for a tip-presence check."""
        ...

    # -- result handling -----------------------------------------------------

    def apply_run_result(
        self, config: Any, state: dict, run: Any, mode: str,
    ) -> dict:
        """Fold the completed *run* back into *state* and return the new state."""
        ...

    # -- calibration ---------------------------------------------------------

    def build_calibration_targets(self, config: Any) -> Any:
        """Return calibration target descriptors for the task's labware."""
        ...

    # -- status & web --------------------------------------------------------

    def status_payload(self, config: Any, state: dict) -> dict:
        """Return the canonical status dict consumed by the generic web shell."""
        ...

    def web_extension(self, config: Any) -> TaskWebExtension | None:
        """Return a :class:`TaskWebExtension`, or *None* if not needed."""
        ...
