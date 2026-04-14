"""Tests for protocol executor and error recovery (mocked client)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from openot2.protocol.recovery import ErrorRecovery, RecoveryContext
from openot2.protocol.generator import ProtocolGenerator, DryRunResult


# ---------------------------------------------------------------------------
# RecoveryContext
# ---------------------------------------------------------------------------

class TestRecoveryContext:
    def test_resolve_ctx_value(self):
        ctx = RecoveryContext(
            run_id="run-1",
            source_well="A1",
            volume=100.0,
        )
        assert ctx.resolve("ctx.source_well") == "A1"
        assert ctx.resolve("ctx.volume") == 100.0
        assert ctx.resolve("ctx.run_id") == "run-1"

    def test_resolve_non_ctx_passthrough(self):
        ctx = RecoveryContext(run_id="run-1")
        assert ctx.resolve("hello") == "hello"
        assert ctx.resolve(42) == 42

    def test_resolve_missing_key_raises(self):
        ctx = RecoveryContext(run_id="run-1")
        with pytest.raises(ValueError, match="not found"):
            ctx.resolve("ctx.nonexistent_key")

    def test_resolve_none_value_raises(self):
        ctx = RecoveryContext(run_id="run-1", source_well=None)
        with pytest.raises(ValueError, match="is None"):
            ctx.resolve("ctx.source_well")

    def test_resolve_extra_field(self):
        ctx = RecoveryContext(run_id="run-1", extra={"custom_field": "value"})
        assert ctx.resolve("ctx.custom_field") == "value"


# ---------------------------------------------------------------------------
# ErrorRecovery
# ---------------------------------------------------------------------------

class TestErrorRecovery:
    def test_default_pickup_recovery(self):
        plan = ErrorRecovery.default_pickup_recovery()
        assert len(plan) == 2
        assert plan[0]["type"] == "drop"
        assert plan[1]["type"] == "home"

    def test_default_liquid_recovery(self):
        plan = ErrorRecovery.default_liquid_recovery()
        assert len(plan) == 4
        types = [t["type"] for t in plan]
        assert types == ["dispense", "blow_out", "drop", "home"]

    def test_execute_calls_client_methods(self):
        mock_client = MagicMock()
        recovery = ErrorRecovery(mock_client)

        plan = [
            {"type": "dispense", "labware_id": "ctx.source_labware_id",
             "well": "ctx.source_well", "volume": "ctx.volume"},
            {"type": "home"},
        ]
        ctx = RecoveryContext(
            run_id="run-1",
            source_labware_id="lw-1",
            source_well="A1",
            volume=100.0,
        )

        result = recovery.execute(plan, ctx)
        assert result is True
        mock_client.dispense.assert_called_once_with(
            volume=100.0, labware_id="lw-1", well="A1", origin="bottom",
        )
        mock_client.home.assert_called_once()

    def test_execute_handles_failure(self):
        mock_client = MagicMock()
        mock_client.dispense.side_effect = RuntimeError("Connection lost")
        recovery = ErrorRecovery(mock_client)

        plan = [{"type": "dispense", "labware_id": "lw-1", "well": "A1", "volume": 100}]
        ctx = RecoveryContext(run_id="run-1")

        result = recovery.execute(plan, ctx)
        assert result is False
        # Should attempt to home as emergency fallback
        mock_client.home.assert_called()


# ---------------------------------------------------------------------------
# Protocol Config Validation (via generator.validate)
# ---------------------------------------------------------------------------

class TestProtocolValidation:
    def test_valid_config_passes(self, sample_config):
        errors = ProtocolGenerator.validate(sample_config)
        assert errors == []

    def test_missing_labware_fails(self):
        config = {"settings": {}, "tasks": []}
        errors = ProtocolGenerator.validate(config)
        assert len(errors) > 0

    def test_missing_pipette_name_fails(self):
        config = {
            "labware": {
                "pipette": {"mount": "right"},  # missing "name"
                "tiprack": {"name": "rack", "slot": "11"},
            },
            "settings": {},
            "tasks": [],
        }
        errors = ProtocolGenerator.validate(config)
        assert any("name" in e.lower() or "required" in e.lower() for e in errors)


# ---------------------------------------------------------------------------
# Dry Run
# ---------------------------------------------------------------------------

class TestDryRun:
    def test_valid_protocol(self, sample_config):
        result = ProtocolGenerator.dry_run(sample_config)
        assert result.success is True
        assert len(result.errors) == 0
        assert len(result.steps_executed) > 0

    def test_aspirate_without_tip(self):
        config = {
            "labware": {
                "pipette": {"name": "p300_multi_gen2"},
                "tiprack": {"name": "rack", "slot": "11"},
                "sources": {"name": "source", "slots": ["8"]},
                "dispense": {"name": "plate", "slot": "10"},
            },
            "settings": {},
            "tasks": [
                # Transfer without pickup → should error
                {
                    "type": "transfer",
                    "source_slot": "8", "source_well": "A1",
                    "dest_slot": "10", "dest_well": "A1",
                    "volume": 100,
                },
            ],
        }
        result = ProtocolGenerator.dry_run(config)
        assert result.success is False
        assert any("no tip" in e.lower() for e in result.errors)

    def test_double_pickup(self):
        config = {
            "labware": {
                "pipette": {"name": "p300"},
                "tiprack": {"name": "rack", "slot": "11"},
            },
            "settings": {},
            "tasks": [
                {"type": "pickup", "well": "A1"},
                {"type": "pickup", "well": "A2"},  # double pickup
            ],
        }
        result = ProtocolGenerator.dry_run(config)
        assert result.success is False
        assert any("already holding" in e.lower() for e in result.errors)

    def test_drop_without_tip(self):
        config = {
            "labware": {
                "pipette": {"name": "p300"},
                "tiprack": {"name": "rack", "slot": "11"},
            },
            "settings": {},
            "tasks": [
                {"type": "drop", "well": "A1"},  # no tip to drop
            ],
        }
        result = ProtocolGenerator.dry_run(config)
        assert result.success is False

    def test_warning_tip_not_dropped(self):
        config = {
            "labware": {
                "pipette": {"name": "p300"},
                "tiprack": {"name": "rack", "slot": "11"},
            },
            "settings": {},
            "tasks": [
                {"type": "pickup", "well": "A1"},
                # No drop → warning
            ],
        }
        result = ProtocolGenerator.dry_run(config)
        assert any("tip still mounted" in w.lower() for w in result.warnings)

    def test_slot_collision(self):
        config = {
            "labware": {
                "pipette": {"name": "p300"},
                "tiprack": {"name": "rack", "slot": "11"},
                "sources": {"name": "source", "slots": ["11"]},  # same as tiprack!
            },
            "settings": {},
            "tasks": [],
        }
        result = ProtocolGenerator.dry_run(config)
        assert result.success is False
        assert any("already occupied" in e.lower() for e in result.errors)
