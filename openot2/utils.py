"""Logging, labware loading, and shared utilities for OpenOT2."""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

if TYPE_CHECKING:
    from openot2.client import OT2Client


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(level: int = logging.INFO, handler: Optional[logging.Handler] = None) -> None:
    """Configure logging for all ``openot2`` modules.

    Args:
        level: Logging level (default ``logging.INFO``).
        handler: Custom handler. If *None*, uses ``StreamHandler(sys.stdout)``.
    """
    root = logging.getLogger("openot2")
    root.setLevel(level)

    if not root.handlers:
        h = handler or logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
        root.addHandler(h)


def get_logger(name: str) -> logging.Logger:
    """Return a logger under the ``openot2`` namespace."""
    return logging.getLogger(f"openot2.{name}")


# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------

def create_output_path(
    base_dir: str,
    run_id: str,
    step_name: str,
    extension: str = ".jpg",
) -> str:
    """Create a timestamped output path for images.

    Returns:
        ``{base_dir}/experiment/{run_id}/{step_name}_{timestamp}{extension}``
    """
    run_folder = os.path.join(base_dir, "experiment", run_id)
    os.makedirs(run_folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(run_folder, f"{step_name}_{timestamp}{extension}")


# ---------------------------------------------------------------------------
# Labware helpers
# ---------------------------------------------------------------------------

@dataclass
class LabwareMap:
    """Typed result of loading labware from a config dict."""

    pipette_id: Optional[str] = None
    tiprack_id: Optional[str] = None
    imaging_labware_id: Optional[str] = None
    sources: Dict[str, str] = field(default_factory=dict)
    dispenses: Dict[str, str] = field(default_factory=dict)

    @property
    def dispense_labware_id(self) -> Optional[str]:
        """Convenience accessor when there is exactly one dispense labware."""
        if len(self.dispenses) == 1:
            return next(iter(self.dispenses.values()))
        return None


def load_labware_from_config(client: "OT2Client", config: Dict[str, Any]) -> LabwareMap:
    """Load all equipment defined in *config* onto the robot.

    Args:
        client: Connected :class:`OT2Client`.
        config: Dict with optional keys ``pipette``, ``tiprack``, ``sources``,
                ``imaging``, ``dispense``.

    Returns:
        Populated :class:`LabwareMap`.
    """
    logger = get_logger("utils")
    result = LabwareMap()

    # Pipette
    pipette_conf = config.get("pipette")
    if pipette_conf is not None:
        result.pipette_id = client.load_pipette(
            pipette_conf["name"],
            mount=pipette_conf.get("mount", "right"),
        )
        logger.info("Pipette loaded: %s", result.pipette_id)

    # Tiprack
    tiprack_conf = config.get("tiprack")
    if tiprack_conf is not None:
        result.tiprack_id = client.load_labware(
            tiprack_conf["name"],
            slot=tiprack_conf["slot"],
        )
        logger.info("Tiprack loaded: %s (slot %s)", result.tiprack_id, tiprack_conf["slot"])

    # Sources (multiple slots, one labware type)
    sources_conf = config.get("sources")
    if sources_conf is not None:
        for slot in sources_conf["slots"]:
            labware_id = client.load_labware(sources_conf["name"], slot=slot)
            result.sources[slot] = labware_id
            logger.info("Source loaded: %s (slot %s)", labware_id, slot)

    # Imaging labware
    imaging_conf = config.get("imaging")
    if imaging_conf is not None:
        result.imaging_labware_id = client.load_labware(
            imaging_conf["name"],
            slot=imaging_conf["slot"],
        )
        logger.info("Imaging labware loaded: %s (slot %s)",
                     result.imaging_labware_id, imaging_conf["slot"])

    # Dispense labware (single or multiple plates)
    dispense_conf = config.get("dispense")
    if dispense_conf is not None:
        slots = dispense_conf.get("slots") or [dispense_conf["slot"]]
        for slot in slots:
            labware_id = client.load_labware(dispense_conf["name"], slot=slot)
            result.dispenses[slot] = labware_id
            logger.info("Dispense labware loaded: %s (slot %s)", labware_id, slot)

    return result


# ---------------------------------------------------------------------------
# Well mapping utilities  (migrated from ptc_utils.py, logic unchanged)
# ---------------------------------------------------------------------------

def build_source_wells_by_slot(
    source_slots: List[str],
    source_wells: List[List[str]],
) -> Dict[str, List[str]]:
    """Build mapping: source slot -> list of source wells."""
    if not source_slots or not source_wells:
        raise ValueError("source_slots and source_wells must not be empty.")
    if len(source_slots) != len(source_wells):
        raise ValueError("source_slots and source_wells must have the same length.")

    mapping: Dict[str, List[str]] = {}
    for slot, wells in zip(source_slots, source_wells):
        if not isinstance(wells, list):
            raise ValueError("Each element of source_wells must be a list.")
        mapping[slot] = [str(w) for w in wells]
    return mapping


def build_dispense_wells_by_slot(
    dispense_slots: List[str],
    dispense_wells: Union[str, List[Union[str, List[str]]]],
) -> Dict[str, List[str]]:
    """Normalize dispense definition into ``{slot: [wells]}`` mapping.

    Supported formats:
        - ``"A1"`` → same well on all plates
        - ``["A1", "A2"]`` (single slot only) → flat list for one plate
        - ``[["C1", "C2"], ["D1"]]`` → list-of-list matching *dispense_slots*
    """
    if not dispense_slots:
        raise ValueError("dispense_slots must not be empty.")
    if dispense_wells is None:
        raise ValueError("dispense_wells must be provided.")

    # Single string → same well on all plates
    if isinstance(dispense_wells, str):
        return {slot: [dispense_wells] for slot in dispense_slots}

    if not isinstance(dispense_wells, list) or len(dispense_wells) == 0:
        raise ValueError("dispense_wells must be a non-empty string or list.")

    first = dispense_wells[0]

    # Flat list of strings
    if isinstance(first, str):
        if len(dispense_slots) != 1:
            raise ValueError(
                "Flat list dispense_wells is only allowed with exactly one dispense slot."
            )
        return {dispense_slots[0]: [str(w) for w in dispense_wells]}

    # List of lists
    if isinstance(first, (list, tuple)):
        if len(dispense_wells) != len(dispense_slots):
            raise ValueError(
                f"List-of-list length ({len(dispense_wells)}) must match "
                f"number of dispense_slots ({len(dispense_slots)})."
            )
        return {
            slot: [str(w) for w in wells]
            for slot, wells in zip(dispense_slots, dispense_wells)
        }

    raise ValueError("Unsupported dispense_wells format.")
