"""High-level WebApp facade that wires deck, handlers, and server together.

Typical usage::

    from webapp import WebApp

    app = WebApp.from_yaml("deck.yaml", robot_ip="169.254.8.56")
    app.run(port=8000)

Or programmatically::

    from webapp import WebApp
    from webapp.deck import DeckConfig

    deck = DeckConfig(
        pipettes={"left": "p300_single_gen2"},
        labware={"1": "corning_96_wellplate_360ul_flat"},
    )
    app = WebApp(deck=deck)
    app.run()
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

from openot2.client import OT2Client
from openot2.control.runner import TaskRunner
from openot2.control.store import JsonRunStore
from webapp.web import create_app
from openot2.operations import OT2Operations
from webapp.deck import DeckConfig
from webapp.handlers import OT2StepHandlers

logger = logging.getLogger("openot2.webapp")


class WebApp:
    """One-stop facade: configure, connect, and serve an OT-2 web controller.

    Parameters
    ----------
    deck:
        Declarative deck layout.  When ``None`` and a robot is connected,
        you must load labware yourself via ``self.client``.
    robot_ip:
        OT-2 IP address.  ``None`` for UI-only (no hardware) mode.
    reconnect:
        If ``True``, reuse the last run on the robot instead of creating
        a fresh one.
    data_dir:
        Directory for JSON run / event persistence.
    rinse_cycles:
        Default rinse cycles for :class:`OT2Operations`.
    rinse_volume:
        Default rinse volume (uL) for :class:`OT2Operations`.
    timeout:
        HTTP timeout in seconds for robot connection.
    """

    def __init__(
        self,
        deck: DeckConfig | None = None,
        robot_ip: str | None = None,
        reconnect: bool = False,
        data_dir: str | Path = "./run_data",
        rinse_cycles: int = 3,
        rinse_volume: float = 250.0,
        timeout: float = 180.0,
    ) -> None:
        self.deck = deck
        self.data_dir = Path(data_dir)

        # Persistence + runner
        self.store = JsonRunStore(base_dir=self.data_dir)
        self.runner = TaskRunner(store=self.store)

        # Robot connection
        self.client: OT2Client | None = None
        self.ops: OT2Operations | None = None

        if robot_ip:
            self._connect(robot_ip, reconnect, timeout, rinse_cycles, rinse_volume)
        else:
            logger.info("No robot IP — running in UI-only mode (no hardware).")

        # Handlers
        self.handlers = OT2StepHandlers(client=self.client, ops=self.ops)
        self.handlers.register_all(self.runner)

    # ------------------------------------------------------------------
    # Alternative constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(
        cls,
        config_path: str | Path,
        robot_ip: str | None = None,
        **kwargs: Any,
    ) -> WebApp:
        """Create a :class:`WebApp` from a YAML config file.

        The YAML file defines the deck layout.  All other keyword
        arguments are forwarded to the constructor.
        """
        deck = DeckConfig.from_yaml(config_path)
        return cls(deck=deck, robot_ip=robot_ip, **kwargs)

    # ------------------------------------------------------------------
    # Custom handler registration
    # ------------------------------------------------------------------

    def register_handler(self, kind: str, handler: Callable) -> None:
        """Register an additional (or override) step handler.

        Use this for domain-specific step kinds that are not covered by
        the built-in set (e.g. ``"photograph"``, ``"incubate"``).
        """
        self.runner.register_handler(kind, handler)

    # ------------------------------------------------------------------
    # Serve
    # ------------------------------------------------------------------

    def create_fastapi_app(self):
        """Build and return the :class:`FastAPI` application."""
        return create_app(
            store=self.store,
            runner=self.runner,
            client=self.client,
        )

    def run(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        """Start the web server (blocking)."""
        import uvicorn

        app = self.create_fastapi_app()
        logger.info("Starting web controller on http://localhost:%d", port)
        uvicorn.run(app, host=host, port=port)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _connect(
        self,
        ip: str,
        reconnect: bool,
        timeout: float,
        rinse_cycles: int,
        rinse_volume: float,
    ) -> None:
        """Connect to the OT-2 and optionally load the deck."""
        logger.info("Connecting to OT-2 at %s ...", ip)
        client = OT2Client(ip, timeout=timeout)

        health = client.health(timeout=30)
        logger.info("Robot healthy: %s", health.get("name", "OK"))

        if reconnect:
            run_id = client.reconnect_last_run()
            logger.info("Reconnected to existing run: %s", run_id)
            if not client.labware_by_slot:
                logger.warning(
                    "Reconnected run has no labware — loading fresh deck."
                )
                reconnect = False

        if not reconnect:
            if self.deck:
                self.deck.load_onto(client)
                logger.info("New run created and deck loaded from config.")
            else:
                client.create_run()
                logger.info("New run created (no deck config — load labware manually).")

        self.client = client
        self.ops = OT2Operations(
            client,
            rinse_cycles=rinse_cycles,
            rinse_volume=rinse_volume,
        )
        logger.info("Robot ready.")
