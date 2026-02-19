"""OT-2 HTTP API client with session management and retry logic."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger("openot2.client")

HEADERS = {"Opentrons-Version": "*"}

Offset = Tuple[float, float, float]
"""Type alias for a 3D offset ``(x, y, z)``."""


class OT2Client:
    """Stateful client for a single OT-2 robot.

    All robot state (run ID, pipette ID, labware mapping) is held as
    instance attributes — never as module globals.

    Args:
        robot_ip: IPv4 address of the OT-2.
        port: HTTP API port (default ``31950``).
        timeout: Request timeout in seconds (default ``120``).
        max_retries: Number of retries on transient failures (default ``3``).
    """

    def __init__(
        self,
        robot_ip: str,
        port: int = 31950,
        timeout: float = 120.0,
        max_retries: int = 3,
    ) -> None:
        self._base_url = f"http://{robot_ip}:{port}"
        self._timeout = timeout

        self._session = requests.Session()
        retry = Retry(
            total=max_retries,
            backoff_factor=0.5,
            status_forcelist=[502, 503, 504],
        )
        self._session.mount("http://", HTTPAdapter(max_retries=retry))
        self._session.headers.update(HEADERS)

        # Run state
        self._run_id: Optional[str] = None
        self._commands_url: Optional[str] = None
        self._pipette_id: Optional[str] = None
        self._labware_by_slot: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def run_id(self) -> Optional[str]:
        return self._run_id

    @property
    def pipette_id(self) -> Optional[str]:
        return self._pipette_id

    @property
    def labware_by_slot(self) -> Dict[str, str]:
        return dict(self._labware_by_slot)

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def create_run(self) -> str:
        """Create a new run on the robot. Returns *run_id*."""
        url = f"{self._base_url}/runs"
        logger.info("POST %s", url)

        r = self._session.post(url, timeout=self._timeout)
        r.raise_for_status()

        data = r.json()["data"]
        self._run_id = data["id"]
        self._commands_url = f"{url}/{self._run_id}/commands"
        logger.info("Run created: %s", self._run_id)
        return self._run_id

    def reconnect_last_run(self, prefer_mount: str = "right") -> str:
        """Reconnect to the most recent run and recover state.

        Recovers ``pipette_id`` and ``labware_by_slot``.
        Returns *run_id*.
        """
        url = f"{self._base_url}/runs"
        logger.info("GET %s", url)
        r = self._session.get(url, timeout=self._timeout)
        r.raise_for_status()

        runs = r.json().get("data", [])
        if not runs:
            raise RuntimeError("No runs found on the robot.")

        last_run = runs[-1]
        self._run_id = last_run["id"]
        self._commands_url = f"{url}/{self._run_id}/commands"
        logger.info("Reconnected to run: %s", self._run_id)

        # Recover pipette
        self._pipette_id = None
        pipettes = last_run.get("pipettes", [])
        chosen = None
        for p in pipettes:
            if p.get("mount") == prefer_mount:
                chosen = p
                break
        if chosen is None and pipettes:
            chosen = pipettes[0]
        if chosen is not None:
            self._pipette_id = chosen.get("id")
            logger.info("Recovered pipette_id: %s", self._pipette_id)

        # Recover labware
        self._labware_by_slot.clear()
        for lw in last_run.get("labware", []):
            lw_id = lw.get("id")
            location = lw.get("location") or {}
            slot = location.get("slotName") or location.get("slot_name")
            if lw_id and slot:
                self._labware_by_slot[slot] = lw_id
        logger.info("Labware recovered: %s", self._labware_by_slot)

        return self._run_id

    # ------------------------------------------------------------------
    # Equipment loading
    # ------------------------------------------------------------------

    def load_pipette(self, pipette_name: str, mount: str = "right") -> str:
        """Load a pipette. Returns *pipette_id*."""
        cmd = {
            "data": {
                "commandType": "loadPipette",
                "params": {"pipetteName": pipette_name, "mount": mount},
                "intent": "setup",
            }
        }
        logger.info("loadPipette: %s (%s)", pipette_name, mount)
        data = self._post_command(cmd)
        self._pipette_id = data["result"]["pipetteId"]
        logger.info("Pipette ID: %s", self._pipette_id)
        return self._pipette_id

    def load_labware(
        self,
        load_name: str,
        slot: str,
        namespace: str = "opentrons",
        version: int = 1,
    ) -> str:
        """Load labware into a deck slot. Returns *labware_id*."""
        cmd = {
            "data": {
                "commandType": "loadLabware",
                "params": {
                    "location": {"slotName": slot},
                    "loadName": load_name,
                    "namespace": namespace,
                    "version": version,
                },
                "intent": "setup",
            }
        }
        logger.info("loadLabware: %s in slot %s", load_name, slot)
        data = self._post_command(cmd)
        labware_id = data["result"]["labwareId"]
        self._labware_by_slot[slot] = labware_id
        logger.info("Labware ID: %s", labware_id)
        return labware_id

    def get_labware_id(self, slot: str) -> str:
        """Resolve a slot name to its labware ID."""
        if slot in self._labware_by_slot:
            return self._labware_by_slot[slot]
        raise KeyError(
            f"No labware recorded for slot '{slot}'. "
            f"Known slots: {list(self._labware_by_slot.keys())}"
        )

    # ------------------------------------------------------------------
    # Pipette operations
    # ------------------------------------------------------------------

    def _require_pipette(self) -> str:
        if self._pipette_id is None:
            raise RuntimeError("No pipette loaded. Call load_pipette() first.")
        return self._pipette_id

    @staticmethod
    def _offset_dict(offset: Optional[Offset]) -> Dict[str, float]:
        off = offset or (0.0, 0.0, 0.0)
        return {"x": off[0], "y": off[1], "z": off[2]}

    def pick_up_tip(
        self,
        labware_id: str,
        well: str = "A1",
        offset: Optional[Offset] = None,
    ) -> None:
        """Pick up a tip from *labware_id* at *well*."""
        pid = self._require_pipette()
        cmd = {
            "data": {
                "commandType": "pickUpTip",
                "params": {
                    "labwareId": labware_id,
                    "wellName": well,
                    "wellLocation": {"origin": "top", "offset": self._offset_dict(offset)},
                    "pipetteId": pid,
                },
                "intent": "setup",
            }
        }
        logger.info("pickUpTip: labware=%s well=%s", labware_id, well)
        self._post_command(cmd)

    def move_to_well(
        self,
        labware_id: str,
        well: str = "A1",
        offset: Optional[Offset] = None,
    ) -> None:
        """Move pipette to *well* on *labware_id*."""
        pid = self._require_pipette()
        cmd = {
            "data": {
                "commandType": "moveToWell",
                "params": {
                    "labwareId": labware_id,
                    "wellName": well,
                    "wellLocation": {"origin": "top", "offset": self._offset_dict(offset)},
                    "pipetteId": pid,
                },
                "intent": "setup",
            }
        }
        logger.info("moveToWell: labware=%s well=%s", labware_id, well)
        self._post_command(cmd)

    def aspirate(
        self,
        volume: float,
        labware_id: str,
        well: str,
        offset: Optional[Offset] = None,
        origin: str = "bottom",
        flow_rate: float = 150.0,
    ) -> None:
        """Aspirate *volume* uL from *well*."""
        pid = self._require_pipette()
        off = offset or (0.0, 0.0, 1.0)
        cmd = {
            "data": {
                "commandType": "aspirate",
                "params": {
                    "pipetteId": pid,
                    "volume": volume,
                    "flowRate": flow_rate,
                    "labwareId": labware_id,
                    "wellName": well,
                    "wellLocation": {"origin": origin, "offset": self._offset_dict(off)},
                },
                "intent": "setup",
            }
        }
        logger.info("aspirate: %suL from %s origin=%s", volume, well, origin)
        self._post_command(cmd)

    def dispense(
        self,
        volume: float,
        labware_id: str,
        well: str,
        offset: Optional[Offset] = None,
        origin: str = "bottom",
        flow_rate: float = 150.0,
    ) -> None:
        """Dispense *volume* uL into *well*."""
        pid = self._require_pipette()
        off = offset or (0.0, 0.0, 1.0)
        cmd = {
            "data": {
                "commandType": "dispense",
                "params": {
                    "pipetteId": pid,
                    "volume": volume,
                    "flowRate": flow_rate,
                    "labwareId": labware_id,
                    "wellName": well,
                    "wellLocation": {"origin": origin, "offset": self._offset_dict(off)},
                },
                "intent": "setup",
            }
        }
        logger.info("dispense: %suL into %s origin=%s", volume, well, origin)
        self._post_command(cmd)

    def drop_tip(
        self,
        labware_id: Optional[str] = None,
        well: str = "A1",
        offset: Optional[Offset] = None,
    ) -> None:
        """Drop tip into *labware_id* (or ``fixedTrash`` if *None*)."""
        pid = self._require_pipette()
        target = labware_id or "fixedTrash"
        cmd = {
            "data": {
                "commandType": "dropTip",
                "params": {
                    "labwareId": target,
                    "wellName": well,
                    "wellLocation": {"origin": "top", "offset": self._offset_dict(offset)},
                    "pipetteId": pid,
                },
                "intent": "setup",
            }
        }
        logger.info("dropTip: labware=%s well=%s", target, well)
        self._post_command(cmd)

    def drop_tip_in_trash(
        self,
        well: str = "A1",
        offset: Optional[Offset] = None,
    ) -> None:
        """Drop tip into fixed trash (two-step: move then eject)."""
        pid = self._require_pipette()
        off = self._offset_dict(offset)

        move_cmd = {
            "data": {
                "commandType": "moveToAddressableAreaForDropTip",
                "params": {
                    "pipetteId": pid,
                    "addressableAreaName": "fixedTrash",
                    "wellName": well,
                    "wellLocation": {"origin": "default", "offset": off},
                    "alternateDropLocation": False,
                },
                "intent": "setup",
            }
        }
        logger.info("moveToAddressableAreaForDropTip -> fixedTrash")
        self._post_command(move_cmd)

        drop_cmd = {
            "data": {
                "commandType": "dropTipInPlace",
                "params": {"pipetteId": pid},
                "intent": "setup",
            }
        }
        logger.info("dropTipInPlace")
        self._post_command(drop_cmd)

    def blow_out(
        self,
        labware_id: Optional[str] = None,
        well: Optional[str] = None,
        offset: Optional[Offset] = None,
        flow_rate: float = 100.0,
    ) -> None:
        """Blow out remaining liquid."""
        pid = self._require_pipette()
        params: Dict[str, Any] = {"pipetteId": pid, "flowRate": flow_rate}

        if labware_id is not None and well is not None:
            params.update({
                "labwareId": labware_id,
                "wellName": well,
                "wellLocation": {"origin": "top", "offset": self._offset_dict(offset)},
            })
        elif labware_id is not None or well is not None:
            raise ValueError("blow_out() requires both labware_id and well, or neither.")

        cmd = {"data": {"commandType": "blowout", "params": params, "intent": "setup"}}
        logger.info("blowout")
        self._post_command(cmd)

    def home(self) -> None:
        """Home the robot."""
        cmd = {"data": {"commandType": "home", "params": {}, "intent": "setup"}}
        logger.info("home")
        self._post_command(cmd)

    def pause(self, message: str = "Paused.") -> None:
        """Pause the current run."""
        cmd = {
            "data": {
                "commandType": "pause",
                "params": {"message": message},
                "intent": "protocol",
            }
        }
        logger.info("pause: %s", message)
        self._post_command(cmd)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _post_command(self, command_dict: dict, wait: bool = True) -> dict:
        """Post a command to the current run's command endpoint."""
        if not self._commands_url:
            raise RuntimeError("No active run. Call create_run() first.")

        params = {"waitUntilComplete": True} if wait else None

        r = self._session.post(
            url=self._commands_url,
            json=command_dict,
            params=params,
            timeout=self._timeout,
        )

        if not r.ok:
            logger.error("HTTP %s: %s", r.status_code, r.text[:500])
            r.raise_for_status()

        return r.json()["data"]
