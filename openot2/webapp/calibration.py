"""Generic pre-run calibration service for aspirate / dispense tasks.

This module provides Pydantic models and utility helpers that let users
build, tweak, preview, and persist calibration profiles *without* any
assay-specific logic.  It is designed to work with :class:`openot2.client.OT2Client`.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from openot2.client import OT2Client

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

CalibrationAction = Literal["aspirate", "dispense", "pick_up_tip"]

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class Offset(BaseModel):
    """Three-axis offset in millimetres."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def as_tuple(self) -> tuple[float, float, float]:
        """Return the offset as an ``(x, y, z)`` tuple."""
        return (self.x, self.y, self.z)


class CalibrationTarget(BaseModel):
    """A single calibration point — one well + action combination."""

    name: str
    pipette_mount: str = "right"
    labware_slot: str
    well: str = "A1"
    action: CalibrationAction
    volume: float | None = None
    offset: Offset = Field(default_factory=Offset)


class CalibrationProfile(BaseModel):
    """An ordered collection of :class:`CalibrationTarget` items."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    name: str
    targets: list[CalibrationTarget] = []
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = {}


class CalibrationSession(BaseModel):
    """Lightweight session state for walking through a profile."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    profile_id: str | None = None
    status: Literal["idle", "active", "completed"] = "idle"
    current_target_index: int = 0
    notes: list[str] = []
    events: list[str] = []


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def nudge_offset(offset: Offset, axis: str, delta: float) -> Offset:
    """Return a *new* :class:`Offset` with *axis* adjusted by *delta*.

    Parameters
    ----------
    offset:
        The base offset to adjust.
    axis:
        One of ``"x"``, ``"y"``, or ``"z"``.
    delta:
        The amount to add to the chosen axis.

    Returns
    -------
    Offset
        A new offset instance.

    Raises
    ------
    ValueError
        If *axis* is not one of ``x``, ``y``, ``z``.
    """
    axis = axis.lower()
    if axis not in ("x", "y", "z"):
        raise ValueError(f"axis must be 'x', 'y', or 'z', got '{axis}'")
    data = offset.model_dump()
    data[axis] += delta
    return Offset(**data)


def build_target(
    name: str,
    labware_slot: str,
    well: str,
    action: CalibrationAction,
    *,
    pipette_mount: str = "right",
    volume: float | None = None,
    offset: Offset | None = None,
) -> CalibrationTarget:
    """Convenience factory for :class:`CalibrationTarget`."""
    return CalibrationTarget(
        name=name,
        labware_slot=labware_slot,
        well=well,
        action=action,
        pipette_mount=pipette_mount,
        volume=volume,
        offset=offset if offset is not None else Offset(),
    )


def _activate_target_pipette(client: OT2Client, target: CalibrationTarget) -> None:
    """Switch the robot to the pipette required by *target*."""
    client.use_pipette(target.pipette_mount)


def preview_target(client: OT2Client, target: CalibrationTarget) -> None:
    """Move the pipette to *target*'s well so the user can visually verify.

    Resolves the labware ID from the slot via
    :pymeth:`OT2Client.get_labware_id`.
    """
    _activate_target_pipette(client, target)
    labware_id = client.get_labware_id(target.labware_slot)
    client.move_to_well(
        labware_id=labware_id,
        well=target.well,
        offset=target.offset.as_tuple(),
    )


def test_aspirate(
    client: OT2Client,
    target: CalibrationTarget,
    volume: float | None = None,
) -> None:
    """Perform a test aspirate using *target*'s parameters.

    Parameters
    ----------
    volume:
        Overrides ``target.volume`` when provided.

    Raises
    ------
    ValueError
        If neither *volume* nor ``target.volume`` is set.
    """
    vol = volume if volume is not None else target.volume
    if vol is None:
        raise ValueError(
            "No volume specified — pass volume explicitly or set target.volume"
        )
    _activate_target_pipette(client, target)
    labware_id = client.get_labware_id(target.labware_slot)
    client.aspirate(
        volume=vol,
        labware_id=labware_id,
        well=target.well,
        offset=target.offset.as_tuple(),
    )


test_aspirate.__test__ = False  # prevent pytest collection


def test_pick_up_tip(client: OT2Client, target: CalibrationTarget) -> None:
    """Pick up a tip at *target*."""
    _activate_target_pipette(client, target)
    labware_id = client.get_labware_id(target.labware_slot)
    client.pick_up_tip(
        labware_id=labware_id,
        well=target.well,
        offset=target.offset.as_tuple(),
    )


test_pick_up_tip.__test__ = False  # prevent pytest collection


def test_drop_tip(client: OT2Client, target: CalibrationTarget) -> None:
    """Return the currently attached tip to *target*'s tip position."""
    _activate_target_pipette(client, target)
    labware_id = client.get_labware_id(target.labware_slot)
    client.drop_tip(
        labware_id=labware_id,
        well=target.well,
        offset=target.offset.as_tuple(),
    )


test_drop_tip.__test__ = False  # prevent pytest collection


def test_dispense(
    client: OT2Client,
    target: CalibrationTarget,
    source_target: CalibrationTarget,
    volume: float | None = None,
) -> None:
    """Perform a source→destination dispense calibration test.

    This exercises the full aspirate→dispense→blow-out path so the
    operator can verify real liquid placement.

    Parameters
    ----------
    target:
        The **destination** dispense target.
    source_target:
        The **source** aspirate target.  Required — dispensing from an
        empty pipette is not a useful calibration action.
    volume:
        Overrides ``target.volume`` (and ``source_target.volume``) when
        provided.  Falls back to ``target.volume``, then
        ``source_target.volume``.

    Raises
    ------
    ValueError
        If no volume can be resolved from any source.
    """
    if source_target.pipette_mount != target.pipette_mount:
        raise ValueError(
            "Dispense source and destination must use the same pipette mount"
        )

    vol = volume if volume is not None else (target.volume or source_target.volume)
    if vol is None:
        raise ValueError(
            "No volume specified — pass volume explicitly or set "
            "target.volume or source_target.volume"
        )

    _activate_target_pipette(client, target)

    # 1. Aspirate from source
    src_labware = client.get_labware_id(source_target.labware_slot)
    client.aspirate(
        volume=vol,
        labware_id=src_labware,
        well=source_target.well,
        offset=source_target.offset.as_tuple(),
    )

    # 2. Dispense to destination
    dst_labware = client.get_labware_id(target.labware_slot)
    client.dispense(
        volume=vol,
        labware_id=dst_labware,
        well=target.well,
        offset=target.offset.as_tuple(),
    )

    # 3. Blow out at destination to clear the tip
    client.blow_out(
        labware_id=dst_labware,
        well=target.well,
        offset=target.offset.as_tuple(),
    )


test_dispense.__test__ = False  # prevent pytest collection


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_profile(profile: CalibrationProfile, path: Path) -> None:
    """Serialise *profile* as JSON and write it to *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(profile.model_dump_json(indent=2), encoding="utf-8")


def load_profile(path: Path) -> CalibrationProfile:
    """Read and validate a :class:`CalibrationProfile` from a JSON file."""
    raw = path.read_text(encoding="utf-8")
    return CalibrationProfile.model_validate_json(raw)
