"""Configurable error recovery with context-based placeholder resolution."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from openot2.client import OT2Client

logger = logging.getLogger("openot2.protocol.recovery")


@dataclass
class RecoveryContext:
    """Typed context for error recovery placeholder resolution.

    Recovery plans can reference context values via ``"ctx.field_name"``
    strings (e.g. ``"ctx.source_well"`` resolves to ``self.source_well``).

    Use :attr:`extra` for custom fields not covered by the typed attributes.
    """

    run_id: str
    pipette_id: Optional[str] = None
    tiprack_id: Optional[str] = None
    well_name: Optional[str] = None
    source_labware_id: Optional[str] = None
    source_well: Optional[str] = None
    dest_labware_id: Optional[str] = None
    dest_well: Optional[str] = None
    pick_well: Optional[str] = None
    volume: Optional[float] = None
    vision_result: Optional[Any] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def resolve(self, value: Any) -> Any:
        """Resolve a ``ctx.xxx`` placeholder to its actual value.

        If *value* is a string starting with ``"ctx."``, looks up the
        corresponding attribute (or ``self.extra`` key). Otherwise returns
        *value* as-is.

        Raises:
            ValueError: If the context key does not exist.
        """
        if not isinstance(value, str) or not value.startswith("ctx."):
            return value

        key = value[4:]  # strip "ctx."
        if hasattr(self, key) and key != "extra":
            result = getattr(self, key)
            if result is None:
                raise ValueError(f"Context key '{key}' is None.")
            return result
        if key in self.extra:
            return self.extra[key]
        raise ValueError(f"Context key '{key}' not found in RecoveryContext.")


class ErrorRecovery:
    """Execute recovery plans defined in protocol configs.

    A recovery plan is a list of action dicts, e.g.::

        [
            {"type": "dispense", "labware_id": "ctx.source_labware_id",
             "well": "ctx.source_well", "volume": "ctx.volume"},
            {"type": "blow_out", "labware_id": "ctx.source_labware_id",
             "well": "ctx.source_well"},
            {"type": "drop", "labware_id": "ctx.tiprack_id",
             "well": "ctx.pick_well"},
            {"type": "home"},
        ]

    Args:
        client: Connected :class:`OT2Client`.
    """

    def __init__(self, client: OT2Client) -> None:
        self._client = client

    def execute(
        self,
        recovery_tasks: List[Dict[str, Any]],
        context: RecoveryContext,
    ) -> bool:
        """Execute a recovery plan.

        Returns:
            *True* if all steps succeeded.
        """
        logger.info("Starting recovery (%d steps)...", len(recovery_tasks))

        try:
            for i, task in enumerate(recovery_tasks):
                task_type = task.get("type")
                logger.info("Step %d: %s", i + 1, task_type)

                r = context.resolve  # shorthand

                if task_type == "dispense":
                    self._client.dispense(
                        volume=r(task["volume"]),
                        labware_id=r(task["labware_id"]),
                        well=r(task["well"]),
                        origin="bottom",
                    )
                elif task_type == "blow_out":
                    self._client.blow_out(
                        labware_id=r(task["labware_id"]),
                        well=r(task["well"]),
                    )
                elif task_type == "drop":
                    self._client.drop_tip(
                        labware_id=r(task.get("labware_id")),
                        well=r(task.get("well", "A1")),
                    )
                elif task_type == "home":
                    self._client.home()
                elif task_type == "pause":
                    self._client.pause(message=task.get("message", "Paused by recovery."))
                else:
                    logger.warning("Unknown recovery task type: %s", task_type)

            logger.info("Recovery plan completed.")
            return True

        except Exception as exc:
            logger.error("Recovery plan failed: %s", exc)
            try:
                self._client.home()
            except Exception:
                pass
            return False

    # ------------------------------------------------------------------
    # Default recovery plans
    # ------------------------------------------------------------------

    @staticmethod
    def default_pickup_recovery() -> List[Dict[str, Any]]:
        """Default recovery plan for a failed tip pickup."""
        return [
            {"type": "drop", "labware_id": "ctx.tiprack_id", "well": "ctx.well_name"},
            {"type": "home"},
        ]

    @staticmethod
    def default_liquid_recovery() -> List[Dict[str, Any]]:
        """Default recovery plan for a failed liquid level check."""
        return [
            {"type": "dispense", "labware_id": "ctx.source_labware_id",
             "well": "ctx.source_well", "volume": "ctx.volume"},
            {"type": "blow_out", "labware_id": "ctx.source_labware_id",
             "well": "ctx.source_well"},
            {"type": "drop", "labware_id": "ctx.tiprack_id", "well": "ctx.pick_well"},
            {"type": "home"},
        ]
