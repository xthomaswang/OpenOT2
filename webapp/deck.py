"""Declarative deck configuration for OT-2 runs."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class DeckConfig(BaseModel):
    """Describes an OT-2 deck layout: pipettes, labware, and defaults.

    Can be constructed from a dict, or loaded from a YAML file via
    :meth:`from_yaml`.

    Example YAML::

        pipettes:
          left: p300_single_gen2
          right: p300_multi_gen2
        labware:
          "1": corning_96_wellplate_360ul_flat
          "4": nest_12_reservoir_15ml
          "10": opentrons_96_filtertiprack_200ul
        active_pipette: left
    """

    pipettes: dict[str, str] = Field(
        default_factory=dict,
        description="Mount ('left'/'right') -> pipette name",
    )
    labware: dict[str, str] = Field(
        default_factory=dict,
        description="Slot number (as string) -> labware name",
    )
    active_pipette: str = Field(
        default="left",
        description="Which mount to activate after loading",
    )

    @classmethod
    def from_yaml(cls, path: str | Path) -> DeckConfig:
        """Load a deck config from a YAML file.

        The YAML may have a top-level ``deck:`` key, or the fields can
        live at the root level.
        """
        import yaml  # lazy — yaml is only needed for file loading

        with open(Path(path)) as f:
            data = yaml.safe_load(f)

        if "deck" in data:
            data = data["deck"]

        # Normalise slot keys to strings
        if "labware" in data:
            data["labware"] = {str(k): v for k, v in data["labware"].items()}

        return cls(**data)

    def load_onto(self, client) -> None:
        """Load this deck config onto a connected :class:`OT2Client`.

        Creates a new run, loads all pipettes and labware, and activates
        the configured pipette.
        """
        client.create_run()

        for mount, name in self.pipettes.items():
            client.load_pipette(name, mount)

        for slot, name in self.labware.items():
            client.load_labware(name, slot)

        if self.active_pipette in self.pipettes:
            client.use_pipette(self.active_pipette)
