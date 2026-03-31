"""OpenOT2 Web Application — configurable, extensible web controller.

Quick start::

    from webapp import WebApp

    # From YAML config
    app = WebApp.from_yaml("deck.yaml", robot_ip="169.254.8.56")
    app.run(port=8000)

    # UI-only (no robot)
    app = WebApp.from_yaml("deck.yaml")
    app.run()
"""

from webapp.app import WebApp
from webapp.deck import DeckConfig
from webapp.handlers import OT2StepHandlers

__all__ = ["WebApp", "DeckConfig", "OT2StepHandlers"]
