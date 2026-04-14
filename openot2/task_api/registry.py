"""Plugin registry — register, look up, and iterate task plugins.

Usage::

    from openot2.task_api.registry import PluginRegistry

    registry = PluginRegistry()
    registry.register(my_plugin)          # uses plugin.name
    registry.register(my_plugin, "alias") # explicit name override

    plugin = registry.get("color_mixing")
"""

from __future__ import annotations

from typing import Iterator

from openot2.task_api.plugin import TaskPlugin


class PluginNotFoundError(KeyError):
    """Raised when a requested plugin name is not in the registry."""


class PluginRegistry:
    """Thread-unsafe, in-process plugin registry.

    Plugins are keyed by their ``name`` attribute (or an explicit override).
    """

    def __init__(self) -> None:
        self._plugins: dict[str, TaskPlugin] = {}

    # -- mutators ------------------------------------------------------------

    def register(self, plugin: TaskPlugin, name: str | None = None) -> None:
        """Register *plugin* under *name* (defaults to ``plugin.name``)."""
        key = name if name is not None else plugin.name
        self._plugins[key] = plugin

    def unregister(self, name: str) -> None:
        """Remove the plugin registered under *name*.

        Raises :class:`PluginNotFoundError` if *name* is not registered.
        """
        try:
            del self._plugins[name]
        except KeyError:
            raise PluginNotFoundError(name) from None

    # -- queries -------------------------------------------------------------

    def get(self, name: str) -> TaskPlugin:
        """Return the plugin registered under *name*.

        Raises :class:`PluginNotFoundError` if not found.
        """
        try:
            return self._plugins[name]
        except KeyError:
            raise PluginNotFoundError(name) from None

    def has(self, name: str) -> bool:
        """Return ``True`` if *name* is registered."""
        return name in self._plugins

    def names(self) -> list[str]:
        """Return a sorted list of registered plugin names."""
        return sorted(self._plugins)

    def __iter__(self) -> Iterator[tuple[str, TaskPlugin]]:
        """Iterate over ``(name, plugin)`` pairs."""
        yield from self._plugins.items()

    def __len__(self) -> int:
        return len(self._plugins)
