"""Reusable OT-2 step handlers for the web controller.

:class:`OT2StepHandlers` encapsulates all handler logic as instance
methods — no global state.  Call :meth:`register_all` to wire every
built-in handler into a :class:`TaskRunner`, or cherry-pick individual
handlers.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openot2.client import OT2Client
    from openot2.control.models import RunStep
    from openot2.control.runner import TaskRunner
    from openot2.operations import OT2Operations

logger = logging.getLogger("openot2.webapp.handlers")


class OT2StepHandlers:
    """Step handlers backed by an :class:`OT2Client` and :class:`OT2Operations`.

    When *client* is ``None`` (UI-only mode), handlers that require the
    robot return ``{"skipped": True}`` instead of raising.

    Parameters
    ----------
    client:
        Connected :class:`OT2Client`, or ``None`` for dry-run mode.
    ops:
        :class:`OT2Operations` wrapping *client*, or ``None``.
    """

    def __init__(
        self,
        client: OT2Client | None = None,
        ops: OT2Operations | None = None,
    ) -> None:
        self.client = client
        self.ops = ops

    # ------------------------------------------------------------------
    # Bulk registration
    # ------------------------------------------------------------------

    #: Maps step ``kind`` strings to the method name that handles them.
    HANDLER_MAP: dict[str, str] = {
        "home": "handle_home",
        "aspirate": "handle_aspirate",
        "dispense": "handle_dispense",
        "move": "handle_move",
        "pick_up_tip": "handle_pick_up_tip",
        "drop_tip": "handle_drop_tip",
        "blow_out": "handle_blow_out",
        "use_pipette": "handle_use_pipette",
        "transfer": "handle_transfer",
        "mix": "handle_mix",
    }

    def register_all(self, runner: TaskRunner) -> None:
        """Register every built-in handler with *runner*."""
        for kind, method_name in self.HANDLER_MAP.items():
            runner.register_handler(kind, getattr(self, method_name))

    # ------------------------------------------------------------------
    # Primitive handlers
    # ------------------------------------------------------------------

    def handle_home(self, step: RunStep, context=None) -> dict:
        if self.client:
            self.client.home()
        return {"action": "home"}

    def handle_aspirate(self, step: RunStep, context=None) -> dict:
        if not self.client:
            return {"skipped": True, "reason": "no client"}
        p = step.params
        labware_id = self.client.get_labware_id(p["slot"])
        self.client.aspirate(
            volume=p["volume"],
            labware_id=labware_id,
            well=p.get("well", "A1"),
        )
        return {"aspirated": p["volume"]}

    def handle_dispense(self, step: RunStep, context=None) -> dict:
        if not self.client:
            return {"skipped": True, "reason": "no client"}
        p = step.params
        labware_id = self.client.get_labware_id(p["slot"])
        self.client.dispense(
            volume=p["volume"],
            labware_id=labware_id,
            well=p.get("well", "A1"),
        )
        return {"dispensed": p["volume"]}

    def handle_move(self, step: RunStep, context=None) -> dict:
        if not self.client:
            return {"skipped": True, "reason": "no client"}
        p = step.params
        labware_id = self.client.get_labware_id(p["slot"])
        offset = tuple(p["offset"]) if "offset" in p else None
        self.client.move_to_well(
            labware_id=labware_id,
            well=p.get("well", "A1"),
            offset=offset,
        )
        return {"moved_to": p.get("well", "A1")}

    def handle_pick_up_tip(self, step: RunStep, context=None) -> dict:
        if not self.client:
            return {"skipped": True, "reason": "no client"}
        p = step.params
        labware_id = self.client.get_labware_id(p["slot"])
        self.client.pick_up_tip(labware_id=labware_id, well=p.get("well", "A1"))
        return {"tip": "picked_up"}

    def handle_drop_tip(self, step: RunStep, context=None) -> dict:
        if not self.client:
            return {"skipped": True, "reason": "no client"}
        self.client.drop_tip_in_trash()
        return {"tip": "dropped"}

    def handle_blow_out(self, step: RunStep, context=None) -> dict:
        if not self.client:
            return {"skipped": True, "reason": "no client"}
        self.client.blow_out()
        return {"action": "blow_out"}

    def handle_use_pipette(self, step: RunStep, context=None) -> dict:
        if not self.client:
            return {"skipped": True, "reason": "no client"}
        mount = step.params["mount"]
        self.client.use_pipette(mount)
        return {"active_pipette": mount}

    # ------------------------------------------------------------------
    # Composite handlers (require OT2Operations)
    # ------------------------------------------------------------------

    def handle_transfer(self, step: RunStep, context=None) -> dict:
        if not self.ops:
            return {"skipped": True, "reason": "no client"}
        p = step.params
        client = self.client
        self.ops.transfer(
            tiprack_id=client.get_labware_id(p["tiprack_slot"]),
            source_id=client.get_labware_id(p["source_slot"]),
            dest_id=client.get_labware_id(p["dest_slot"]),
            tip_well=p.get("tip_well", "A1"),
            source_well=p.get("source_well", "A1"),
            dest_well=p.get("dest_well", "A1"),
            volume=p["volume"],
            cleaning_id=(
                client.get_labware_id(p["cleaning_slot"])
                if "cleaning_slot" in p
                else None
            ),
            rinse_col=p.get("rinse_well"),
        )
        return {"transferred": p["volume"], "to": p.get("dest_well")}

    def handle_mix(self, step: RunStep, context=None) -> dict:
        if not self.ops:
            return {"skipped": True, "reason": "no client"}
        p = step.params
        client = self.client
        self.ops.mix(
            tiprack_id=client.get_labware_id(p["tiprack_slot"]),
            labware_id=client.get_labware_id(p["plate_slot"]),
            tip_well=p.get("tip_well", "A4"),
            mix_well=p.get("mix_well", "A1"),
            cycles=p.get("cycles", 3),
            volume=p.get("volume", 150),
            cleaning_id=(
                client.get_labware_id(p["cleaning_slot"])
                if "cleaning_slot" in p
                else None
            ),
            rinse_col=p.get("rinse_well"),
        )
        return {"mixed": p.get("mix_well"), "cycles": p.get("cycles", 3)}
