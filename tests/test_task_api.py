"""Tests for openot2.task_api — plugin protocol, registry, loader, and state helpers."""

from __future__ import annotations

import tempfile
from typing import Any

import pytest

from openot2.control.models import RunStep, TaskRun
from openot2.control.runner import TaskRunner
from openot2.control.store import JsonRunStore
from openot2.task_api import (
    PluginLoadError,
    PluginNotFoundError,
    PluginRegistry,
    TaskPlugin,
    TaskWebExtension,
    current_iteration,
    current_phase,
    is_terminal,
    load_plugin,
    make_state,
)


# ---------------------------------------------------------------------------
# Concrete test doubles that satisfy the protocols
# ---------------------------------------------------------------------------


class _StubWebExtension:
    def extra_routes(self):
        return []

    def ui_payload(self, config, state):
        return {"task_ui": True}

    def extra_status(self, config, state):
        return {"extra": "status"}


class _StubPlugin:
    name = "stub_task"

    def load_config(self, path: str):
        return {"loaded_from": path}

    def build_deck_config(self, config):
        return {"deck": True}

    def initial_state(self, config, mode: str):
        return make_state(phase="idle", iteration=0)

    def build_plan(self, config, state, mode: str):
        return {"iterations": 5, "mode": mode}

    def build_iteration_run(self, config, state, iteration: int, mode: str):
        return TaskRun(
            name=f"stub_iter_{iteration}",
            steps=[
                RunStep(key="mix", name="Mix", kind="liquid", params={"vol": 10}),
                RunStep(key="wait", name="Wait", kind="delay", params={"seconds": 5}),
                RunStep(
                    key="capture",
                    name="Capture",
                    kind="camera",
                    params={"slot": 1},
                ),
                RunStep(
                    key="analyse",
                    name="Analyse",
                    kind="analysis",
                    params={"image": {"$ref": "capture.output.image_path"}},
                ),
            ],
        )

    def build_calibration_run(self, config, state):
        return TaskRun(name="stub_calibration", steps=[])

    def build_tip_check_run(self, config, state):
        return TaskRun(name="stub_tip_check", steps=[])

    def apply_run_result(self, config, state, run, mode: str):
        state = dict(state)
        state["iteration"] = state.get("iteration", 0) + 1
        return state

    def build_calibration_targets(self, config):
        return [{"label": "target_1"}]

    def status_payload(self, config, state):
        return {"phase": state.get("phase"), "iteration": state.get("iteration")}

    def web_extension(self, config):
        return _StubWebExtension()


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestProtocolConformance:
    def test_stub_satisfies_task_plugin(self):
        assert isinstance(_StubPlugin(), TaskPlugin)

    def test_stub_web_extension_satisfies_protocol(self):
        assert isinstance(_StubWebExtension(), TaskWebExtension)

    def test_plain_object_does_not_satisfy_task_plugin(self):
        assert not isinstance(object(), TaskPlugin)

    def test_plain_object_does_not_satisfy_web_extension(self):
        assert not isinstance(object(), TaskWebExtension)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestPluginRegistry:
    def test_register_and_get(self):
        reg = PluginRegistry()
        p = _StubPlugin()
        reg.register(p)
        assert reg.get("stub_task") is p

    def test_register_with_explicit_name(self):
        reg = PluginRegistry()
        p = _StubPlugin()
        reg.register(p, "my_alias")
        assert reg.get("my_alias") is p
        assert not reg.has("stub_task")

    def test_get_missing_raises(self):
        reg = PluginRegistry()
        with pytest.raises(PluginNotFoundError):
            reg.get("no_such_plugin")

    def test_has(self):
        reg = PluginRegistry()
        reg.register(_StubPlugin())
        assert reg.has("stub_task")
        assert not reg.has("missing")

    def test_unregister(self):
        reg = PluginRegistry()
        reg.register(_StubPlugin())
        reg.unregister("stub_task")
        assert not reg.has("stub_task")

    def test_unregister_missing_raises(self):
        reg = PluginRegistry()
        with pytest.raises(PluginNotFoundError):
            reg.unregister("nope")

    def test_names(self):
        reg = PluginRegistry()
        reg.register(_StubPlugin(), "beta")
        reg.register(_StubPlugin(), "alpha")
        assert reg.names() == ["alpha", "beta"]

    def test_iter(self):
        reg = PluginRegistry()
        p = _StubPlugin()
        reg.register(p, "x")
        items = list(reg)
        assert items == [("x", p)]

    def test_len(self):
        reg = PluginRegistry()
        assert len(reg) == 0
        reg.register(_StubPlugin())
        assert len(reg) == 1


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


class TestLoader:
    def test_load_plugin_invalid_path(self):
        with pytest.raises(PluginLoadError, match="Invalid class path"):
            load_plugin("no_dot_here")

    def test_load_plugin_missing_module(self):
        with pytest.raises(PluginLoadError, match="Cannot import"):
            load_plugin("nonexistent.module.Class")

    def test_load_plugin_missing_class(self):
        with pytest.raises(PluginLoadError, match="has no attribute"):
            load_plugin("openot2.task_api.plugin.NoSuchClass")

    def test_load_plugin_not_conforming(self):
        with pytest.raises(PluginLoadError, match="does not satisfy"):
            load_plugin("openot2.task_api.registry.PluginRegistry")


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------


class TestStateHelpers:
    def test_make_state_defaults(self):
        s = make_state()
        assert s["phase"] == "idle"
        assert s["iteration"] == 0
        assert s["history"] == []
        assert s["metrics"] == {}
        assert s["artifacts"] == {}
        assert s["cache"] == {}
        assert s["terminal"] is False

    def test_make_state_custom(self):
        s = make_state(phase="running", iteration=3, terminal=True, custom_key="val")
        assert s["phase"] == "running"
        assert s["iteration"] == 3
        assert s["terminal"] is True
        assert s["custom_key"] == "val"

    def test_is_terminal(self):
        assert is_terminal({"terminal": True})
        assert not is_terminal({"terminal": False})
        assert not is_terminal({})

    def test_current_phase(self):
        assert current_phase({"phase": "calibrating"}) == "calibrating"
        assert current_phase({}) == "idle"

    def test_current_iteration(self):
        assert current_iteration({"iteration": 7}) == 7
        assert current_iteration({}) == 0


# ---------------------------------------------------------------------------
# Plugin produces TaskRun compatible with runner/store
# ---------------------------------------------------------------------------


class TestPluginRunnerIntegration:
    """Verify that a TaskRun produced by a plugin can flow through the
    existing runner and store without modification."""

    def test_plugin_run_persists_through_store(self):
        plugin = _StubPlugin()
        config = plugin.load_config("dummy.yaml")
        state = plugin.initial_state(config, "run")
        run = plugin.build_iteration_run(config, state, iteration=0, mode="run")

        with tempfile.TemporaryDirectory() as tmp:
            store = JsonRunStore(tmp)
            store.create_run(run)
            loaded = store.load_run(run.id)

        assert loaded.name == run.name
        assert len(loaded.steps) == 4
        assert loaded.steps[0].key == "mix"
        assert loaded.steps[3].params == {"image": {"$ref": "capture.output.image_path"}}

    def test_plugin_run_executes_with_runner(self):
        plugin = _StubPlugin()
        config = plugin.load_config("dummy.yaml")
        state = plugin.initial_state(config, "run")
        run = plugin.build_iteration_run(config, state, iteration=0, mode="run")

        with tempfile.TemporaryDirectory() as tmp:
            store = JsonRunStore(tmp)
            store.create_run(run)

            runner = TaskRunner(store)
            runner.register_handler("liquid", lambda step, ctx: {"done": True})
            runner.register_handler("delay", lambda step, ctx: {"waited": True})
            runner.register_handler(
                "camera", lambda step, ctx: {"image_path": "/tmp/img.png"}
            )
            runner.register_handler(
                "analysis", lambda step, ctx: {"rgb": [128, 64, 32]}
            )

            result = runner.run_until_pause_or_done(run.id)

        assert result.status.value == "completed"
        assert result.steps[3].output == {"rgb": [128, 64, 32]}
        # $ref should have been resolved before the analysis handler ran
        assert result.steps[3].params == {"image": "/tmp/img.png"}

    def test_apply_run_result_advances_state(self):
        plugin = _StubPlugin()
        config = plugin.load_config("dummy.yaml")
        state = plugin.initial_state(config, "run")
        assert state["iteration"] == 0

        run = plugin.build_iteration_run(config, state, iteration=0, mode="run")
        new_state = plugin.apply_run_result(config, state, run, "run")
        assert new_state["iteration"] == 1

    def test_web_extension_methods(self):
        plugin = _StubPlugin()
        config = plugin.load_config("dummy.yaml")
        ext = plugin.web_extension(config)
        assert ext is not None

        state = plugin.initial_state(config, "run")
        assert ext.extra_routes() == []
        assert ext.ui_payload(config, state) == {"task_ui": True}
        assert ext.extra_status(config, state) == {"extra": "status"}

    def test_status_payload(self):
        plugin = _StubPlugin()
        config = plugin.load_config("dummy.yaml")
        state = make_state(phase="running", iteration=2)
        payload = plugin.status_payload(config, state)
        assert payload == {"phase": "running", "iteration": 2}
