"""High-level WebApp facade that wires deck, handlers, and server together.

Typical usage::

    from openot2.webapp import WebApp

    app = WebApp.from_yaml("deck.yaml", robot_ip="169.254.8.56")
    app.run(port=8000)

Or programmatically::

    from openot2.webapp import WebApp
    from openot2.webapp.deck import DeckConfig

    deck = DeckConfig(
        pipettes={"left": "p300_single_gen2"},
        labware={"1": "corning_96_wellplate_360ul_flat"},
    )
    app = WebApp(deck=deck)
    app.run()

With a task plugin::

    from openot2.webapp import WebApp

    app = WebApp(plugin=my_plugin, config_path="experiment.yaml")
    app.run()
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

from fastapi import FastAPI
from openot2.client import OT2Client
from openot2.control.runner import TaskRunner
from openot2.control.store import JsonRunStore
from openot2.webapp.web import create_app
from openot2.operations import OT2Operations
from openot2.webapp.deck import DeckConfig
from openot2.webapp.handlers import OT2StepHandlers

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
    plugin:
        Optional :class:`~openot2.task_api.plugin.TaskPlugin` instance.
        When set, the web shell exposes generic task routes that delegate
        to the plugin for plan, status, run construction, etc.
    config_path:
        Path to the task configuration file.  Required when *plugin* is
        provided.
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
        plugin: Any | None = None,
        config_path: str | Path | None = None,
    ) -> None:
        self.deck = deck
        self.data_dir = Path(data_dir)

        # Plugin support
        self.plugin = plugin
        self.config_path = str(config_path) if config_path else None
        self._task_config: Any = None
        self._task_state: dict | None = None

        # If plugin provided, load config and derive deck if not explicit
        if plugin and config_path and deck is None:
            self._task_config = plugin.load_config(str(config_path))
            deck_obj = plugin.build_deck_config(self._task_config)
            if isinstance(deck_obj, DeckConfig):
                self.deck = deck_obj
            elif isinstance(deck_obj, dict):
                self.deck = DeckConfig(
                    pipettes=deck_obj.get("pipettes", {}),
                    labware=deck_obj.get("labware", {}),
                )

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

        # Extension points
        self._sub_apps: list[tuple[str, FastAPI]] = []
        self._nav_links: list[dict] = []
        self.calibration_profile = None
        self.calibration_session = None

    # ------------------------------------------------------------------
    # Plugin accessors
    # ------------------------------------------------------------------

    @property
    def task_config(self) -> Any:
        """Return the task config, loading lazily if needed."""
        if self._task_config is None and self.plugin and self.config_path:
            self._task_config = self.plugin.load_config(self.config_path)
        return self._task_config

    @task_config.setter
    def task_config(self, value: Any) -> None:
        self._task_config = value

    @property
    def task_state(self) -> dict | None:
        """Return the current task state (managed by the task routes)."""
        return self._task_state

    @task_state.setter
    def task_state(self, value: dict | None) -> None:
        self._task_state = value

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

    def mount_app(self, path: str, app: "FastAPI") -> None:
        """Register a FastAPI sub-application to be mounted at *path*."""
        self._sub_apps.append((path, app))

    def add_nav_link(self, title: str, path: str, icon: str = "&#9671;") -> None:
        """Add a navigation link to the sidebar."""
        self._nav_links.append({"title": title, "path": path, "icon": icon})

    def set_calibration_profile(self, profile, session=None) -> None:
        """Set the calibration profile used by the calibration UI and runtime."""
        self.calibration_profile = profile
        self.calibration_session = session
        self.handlers.set_calibration_profile(profile)

    # ------------------------------------------------------------------
    # Serve
    # ------------------------------------------------------------------

    def create_fastapi_app(self):
        """Build and return the :class:`FastAPI` application."""
        app = create_app(
            store=self.store,
            runner=self.runner,
            client=self.client,
            initial_calibration_profile=self.calibration_profile,
            initial_calibration_session=self.calibration_session,
            calibration_profile_sync=self.handlers.set_calibration_profile,
            nav_links=self._nav_links or None,
            plugin=self.plugin,
            webapp=self,
        )
        for path, sub in self._sub_apps:
            app.mount(path, sub)
        return app

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
