"""Plugin loader — instantiate a plugin from a dotted class path.

Usage::

    from openot2.task_api.loader import load_plugin

    plugin = load_plugin("src.tasks.color_mixing.ColorMixingPlugin")
"""

from __future__ import annotations

import importlib
from typing import Any

from openot2.task_api.plugin import TaskPlugin


class PluginLoadError(Exception):
    """Raised when a plugin class cannot be imported or instantiated."""


def load_plugin_class(class_path: str) -> type:
    """Import and return the class located at *class_path*.

    *class_path* is a dotted string like ``"pkg.module.ClassName"``.
    """
    module_path, _, class_name = class_path.rpartition(".")
    if not module_path or not class_name:
        raise PluginLoadError(
            f"Invalid class path '{class_path}': expected 'module.ClassName'"
        )
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise PluginLoadError(
            f"Cannot import module '{module_path}': {exc}"
        ) from exc

    try:
        cls = getattr(module, class_name)
    except AttributeError:
        raise PluginLoadError(
            f"Module '{module_path}' has no attribute '{class_name}'"
        )
    return cls


def load_plugin(class_path: str, **kwargs: Any) -> TaskPlugin:
    """Import, instantiate, and return a :class:`TaskPlugin`.

    *kwargs* are forwarded to the plugin constructor.
    """
    cls = load_plugin_class(class_path)
    try:
        instance = cls(**kwargs)
    except TypeError as exc:
        raise PluginLoadError(
            f"Cannot instantiate '{class_path}': {exc}"
        ) from exc
    if not isinstance(instance, TaskPlugin):
        raise PluginLoadError(
            f"'{class_path}' does not satisfy the TaskPlugin protocol"
        )
    return instance
