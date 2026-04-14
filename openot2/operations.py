"""High-level robot operations built on OT2Client.

Provides :class:`OT2Operations`, a composable wrapper around
:class:`OT2Client` with reusable pipetting workflows: transfer,
mix, and rinse — each handling the full pick-up → action → clean →
return cycle.

Example::

    from openot2 import OT2Client
    from openot2.operations import OT2Operations

    client = OT2Client("169.254.8.56")
    client.create_run()

    ops = OT2Operations(client, rinse_cycles=3, rinse_volume=250)

    # Load equipment
    ops.client.load_pipette("p300_multi_gen2", "right")
    tiprack = ops.client.load_labware("opentrons_96_tiprack_300ul", "1")
    source  = ops.client.load_labware("nest_12_reservoir_15ml", "4")
    plate   = ops.client.load_labware("corning_96_wellplate_360ul_flat", "5")
    clean   = ops.client.load_labware("nest_12_reservoir_15ml", "6")

    ops.transfer(tiprack, source, plate, "A1", "A1", "A1", 100,
                 cleaning_id=clean, rinse_col="A1")
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

from openot2.client import OT2Client

logger = logging.getLogger("openot2.operations")

ProgressCallback = Callable[[dict], None]


class OT2Operations:
    """High-level pipetting operations that compose an :class:`OT2Client`.

    Args:
        client: A connected :class:`OT2Client` instance.
        rinse_cycles: Default number of aspirate/dispense rinse cycles.
        rinse_volume: Default volume (µL) per rinse cycle.
    """

    def __init__(
        self,
        client: OT2Client,
        rinse_cycles: int = 3,
        rinse_volume: float = 250.0,
    ) -> None:
        self.client = client
        self.rinse_cycles = rinse_cycles
        self.rinse_volume = rinse_volume

    def _emit_progress(
        self,
        progress_callback: Optional[ProgressCallback],
        *,
        action: str,
        detail: str,
        cycle: Optional[int] = None,
        total_cycles: Optional[int] = None,
    ) -> None:
        """Emit a sub-step progress update when a callback is configured."""
        if not progress_callback:
            return
        payload = {"action": action, "detail": detail}
        if cycle is not None:
            payload["cycle"] = cycle
        if total_cycles is not None:
            payload["total_cycles"] = total_cycles
        progress_callback(payload)

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def transfer(
        self,
        tiprack_id: str,
        source_id: str,
        dest_id: str,
        tip_well: str,
        source_well: str,
        dest_well: str,
        volume: float,
        cleaning_id: Optional[str] = None,
        tip_offset=None,
        source_offset=None,
        dest_offset=None,
        cleaning_offset=None,
        rinse_col: Optional[str] = None,
        rinse_cycles: Optional[int] = None,
        rinse_volume: Optional[float] = None,
        blow_out: bool = True,
        return_tip: bool = True,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> None:
        """Pick up tip → aspirate → dispense → optional rinse → return tip.

        Args:
            tiprack_id: Labware ID of the tiprack.
            source_id: Labware ID of the source reservoir/plate.
            dest_id: Labware ID of the destination plate.
            tip_well: Well to pick up the tip from (e.g. ``"A1"``).
            source_well: Well to aspirate from.
            dest_well: Well to dispense into.
            volume: Volume in µL.
            cleaning_id: Labware ID of the cleaning reservoir.
            rinse_col: Well in cleaning reservoir for rinsing.
            rinse_cycles: Override instance default rinse cycles.
            rinse_volume: Override instance default rinse volume.
            blow_out: Whether to blow out after dispensing.
            return_tip: Whether to return the tip to the tiprack.
        """
        logger.info(
            "transfer: %s -> %s (%.1f uL) tip=%s",
            source_well, dest_well, volume, tip_well,
        )

        self._emit_progress(
            progress_callback,
            action="pick_up_tip",
            detail=f"tip {tip_well}",
        )
        self.client.pick_up_tip(tiprack_id, tip_well, offset=tip_offset)
        self._emit_progress(
            progress_callback,
            action="aspirate",
            detail=f"{source_well} ({volume:.1f}uL)",
        )
        self.client.aspirate(volume, source_id, source_well, offset=source_offset)
        self._emit_progress(
            progress_callback,
            action="dispense",
            detail=f"{dest_well} ({volume:.1f}uL)",
        )
        self.client.dispense(volume, dest_id, dest_well, offset=dest_offset)

        if blow_out:
            self._emit_progress(
                progress_callback,
                action="blow_out",
                detail=dest_well,
            )
            self.client.blow_out(dest_id, dest_well, offset=dest_offset)

        if cleaning_id and rinse_col:
            self.rinse(
                cleaning_id, rinse_col,
                offset=cleaning_offset,
                cycles=rinse_cycles, volume=rinse_volume,
                progress_callback=progress_callback,
            )

        if return_tip:
            self._emit_progress(
                progress_callback,
                action="drop_tip",
                detail=f"tip {tip_well}",
            )
            self.client.drop_tip(tiprack_id, tip_well, offset=tip_offset)

    def mix(
        self,
        tiprack_id: str,
        labware_id: str,
        tip_well: str,
        mix_well: str,
        cycles: int = 3,
        volume: float = 200.0,
        cleaning_id: Optional[str] = None,
        tip_offset=None,
        labware_offset=None,
        cleaning_offset=None,
        rinse_col: Optional[str] = None,
        rinse_cycles: Optional[int] = None,
        rinse_volume: Optional[float] = None,
        return_tip: bool = True,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> None:
        """Pick up tip → aspirate/dispense cycles → optional rinse → return tip.

        Args:
            tiprack_id: Labware ID of the tiprack.
            labware_id: Labware ID of the plate to mix in.
            tip_well: Well to pick up the tip from.
            mix_well: Well to mix in.
            cycles: Number of aspirate/dispense mixing cycles.
            volume: Volume per mix cycle in µL.
            cleaning_id: Labware ID of cleaning reservoir.
            rinse_col: Well in cleaning reservoir for rinsing.
            rinse_cycles: Override instance default rinse cycles.
            rinse_volume: Override instance default rinse volume.
            return_tip: Whether to return the tip to the tiprack.
        """
        logger.info(
            "mix: %s x%d (%.1f uL) tip=%s",
            mix_well, cycles, volume, tip_well,
        )

        self._emit_progress(
            progress_callback,
            action="pick_up_tip",
            detail=f"tip {tip_well}",
        )
        self.client.pick_up_tip(tiprack_id, tip_well, offset=tip_offset)

        for idx in range(cycles):
            self._emit_progress(
                progress_callback,
                action="mix_aspirate",
                detail=f"{mix_well} ({volume:.1f}uL)",
                cycle=idx + 1,
                total_cycles=cycles,
            )
            self.client.aspirate(volume, labware_id, mix_well, offset=labware_offset)
            self._emit_progress(
                progress_callback,
                action="mix_dispense",
                detail=f"{mix_well} ({volume:.1f}uL)",
                cycle=idx + 1,
                total_cycles=cycles,
            )
            self.client.dispense(volume, labware_id, mix_well, offset=labware_offset)

        self._emit_progress(
            progress_callback,
            action="blow_out",
            detail=mix_well,
        )
        self.client.blow_out(labware_id, mix_well, offset=labware_offset)

        if cleaning_id and rinse_col:
            self.rinse(
                cleaning_id, rinse_col,
                offset=cleaning_offset,
                cycles=rinse_cycles, volume=rinse_volume,
                progress_callback=progress_callback,
            )

        if return_tip:
            self._emit_progress(
                progress_callback,
                action="drop_tip",
                detail=f"tip {tip_well}",
            )
            self.client.drop_tip(tiprack_id, tip_well, offset=tip_offset)

    def rinse(
        self,
        cleaning_id: str,
        well: str,
        offset=None,
        cycles: Optional[int] = None,
        volume: Optional[float] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> None:
        """Rinse the current tip(s) in a cleaning reservoir.

        Assumes a tip is already on the pipette. Does *not* pick up or
        drop the tip.

        Args:
            cleaning_id: Labware ID of the cleaning reservoir.
            well: Well/column in the cleaning reservoir.
            cycles: Number of rinse cycles (default: instance setting).
            volume: Volume per cycle in µL (default: instance setting).
        """
        n = cycles if cycles is not None else self.rinse_cycles
        v = volume if volume is not None else self.rinse_volume

        logger.info("rinse: %s x%d (%.1f uL)", well, n, v)

        for idx in range(n):
            self._emit_progress(
                progress_callback,
                action="rinse_aspirate",
                detail=f"{well} ({v:.1f}uL)",
                cycle=idx + 1,
                total_cycles=n,
            )
            self.client.aspirate(v, cleaning_id, well, offset=offset)
            self._emit_progress(
                progress_callback,
                action="rinse_dispense",
                detail=f"{well} ({v:.1f}uL)",
                cycle=idx + 1,
                total_cycles=n,
            )
            self.client.dispense(v, cleaning_id, well, offset=offset)
        self._emit_progress(
            progress_callback,
            action="blow_out",
            detail=well,
        )
        self.client.blow_out(cleaning_id, well, offset=offset)
