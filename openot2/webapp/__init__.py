"""OpenOT2 Web Application — configurable, extensible web controller.

Quick start::

    from openot2.webapp import WebApp

    # From YAML config
    app = WebApp.from_yaml("deck.yaml", robot_ip="169.254.8.56")
    app.run(port=8000)

    # UI-only (no robot)
    app = WebApp.from_yaml("deck.yaml")
    app.run()

    # With a task plugin
    app = WebApp(plugin=my_plugin, config_path="experiment.yaml")
    app.run()
"""

from openot2.webapp.app import WebApp
from openot2.webapp.deck import DeckConfig
from openot2.webapp.handlers import OT2StepHandlers

__all__ = ["WebApp", "DeckConfig", "OT2StepHandlers"]
