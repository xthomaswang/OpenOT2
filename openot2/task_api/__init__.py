"""Task plugin API — the canonical boundary between tasks and the framework."""

from openot2.task_api.loader import PluginLoadError, load_plugin, load_plugin_class
from openot2.task_api.models import (
    current_iteration,
    current_phase,
    is_terminal,
    make_state,
)
from openot2.task_api.plugin import TaskPlugin, TaskWebExtension
from openot2.task_api.registry import PluginNotFoundError, PluginRegistry

__all__ = [
    # protocols
    "TaskPlugin",
    "TaskWebExtension",
    # state helpers
    "make_state",
    "is_terminal",
    "current_phase",
    "current_iteration",
    # registry
    "PluginRegistry",
    "PluginNotFoundError",
    # loader
    "load_plugin",
    "load_plugin_class",
    "PluginLoadError",
]
