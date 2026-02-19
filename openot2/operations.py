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
from typing import Optional

from openot2.client import OT2Client

logger = logging.getLogger("openot2.operations")


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
        rinse_col: Optional[str] = None,
        rinse_cycles: Optional[int] = None,
        rinse_volume: Optional[float] = None,
        blow_out: bool = True,
        return_tip: bool = True,
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

        self.client.pick_up_tip(tiprack_id, tip_well)
        self.client.aspirate(volume, source_id, source_well)
        self.client.dispense(volume, dest_id, dest_well)

        if blow_out:
            self.client.blow_out(dest_id, dest_well)

        if cleaning_id and rinse_col:
            self.rinse(
                cleaning_id, rinse_col,
                cycles=rinse_cycles, volume=rinse_volume,
            )

        if return_tip:
            self.client.drop_tip(tiprack_id, tip_well)

    def mix(
        self,
        tiprack_id: str,
        labware_id: str,
        tip_well: str,
        mix_well: str,
        cycles: int = 3,
        volume: float = 200.0,
        cleaning_id: Optional[str] = None,
        rinse_col: Optional[str] = None,
        rinse_cycles: Optional[int] = None,
        rinse_volume: Optional[float] = None,
        return_tip: bool = True,
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

        self.client.pick_up_tip(tiprack_id, tip_well)

        for _ in range(cycles):
            self.client.aspirate(volume, labware_id, mix_well)
            self.client.dispense(volume, labware_id, mix_well)

        self.client.blow_out(labware_id, mix_well)

        if cleaning_id and rinse_col:
            self.rinse(
                cleaning_id, rinse_col,
                cycles=rinse_cycles, volume=rinse_volume,
            )

        if return_tip:
            self.client.drop_tip(tiprack_id, tip_well)

    def rinse(
        self,
        cleaning_id: str,
        well: str,
        cycles: Optional[int] = None,
        volume: Optional[float] = None,
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

        for _ in range(n):
            self.client.aspirate(v, cleaning_id, well)
            self.client.dispense(v, cleaning_id, well)
        self.client.blow_out(cleaning_id, well)
