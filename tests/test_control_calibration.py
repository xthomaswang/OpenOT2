"""Tests for openot2.control.calibration — fully hardware-free."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from openot2.webapp.calibration import (
    CalibrationProfile,
    CalibrationSession,
    CalibrationTarget,
    Offset,
    build_target,
    load_profile,
    nudge_offset,
    preview_target,
    save_profile,
    test_aspirate,
    test_drop_tip,
    test_dispense,
    test_pick_up_tip,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client() -> MagicMock:
    """Return a MagicMock standing in for :class:`OT2Client`."""
    client = MagicMock()
    client.get_labware_id.return_value = "labware-123"
    return client


# ---------------------------------------------------------------------------
# Offset
# ---------------------------------------------------------------------------


class TestOffset:
    def test_defaults(self) -> None:
        o = Offset()
        assert o.x == 0.0
        assert o.y == 0.0
        assert o.z == 0.0

    def test_as_tuple(self) -> None:
        o = Offset(x=1.0, y=2.0, z=3.0)
        assert o.as_tuple() == (1.0, 2.0, 3.0)


# ---------------------------------------------------------------------------
# nudge_offset
# ---------------------------------------------------------------------------


class TestNudgeOffset:
    def test_nudge_x(self) -> None:
        o = Offset(x=1.0, y=2.0, z=3.0)
        result = nudge_offset(o, "x", 0.5)
        assert result.x == pytest.approx(1.5)
        assert result.y == 2.0
        assert result.z == 3.0

    def test_nudge_y(self) -> None:
        o = Offset(x=1.0, y=2.0, z=3.0)
        result = nudge_offset(o, "y", -1.0)
        assert result.y == pytest.approx(1.0)
        assert result.x == 1.0
        assert result.z == 3.0

    def test_nudge_z(self) -> None:
        o = Offset()
        result = nudge_offset(o, "z", 10.0)
        assert result.z == pytest.approx(10.0)

    def test_nudge_case_insensitive(self) -> None:
        o = Offset()
        result = nudge_offset(o, "Z", 5.0)
        assert result.z == pytest.approx(5.0)

    def test_nudge_invalid_axis(self) -> None:
        with pytest.raises(ValueError, match="axis must be"):
            nudge_offset(Offset(), "w", 1.0)

    def test_original_unchanged(self) -> None:
        o = Offset(x=1.0)
        nudge_offset(o, "x", 5.0)
        assert o.x == 1.0  # immutable-style


# ---------------------------------------------------------------------------
# build_target
# ---------------------------------------------------------------------------


class TestBuildTarget:
    def test_basic(self) -> None:
        t = build_target("src", "3", "B2", "aspirate", volume=50.0)
        assert t.name == "src"
        assert t.labware_slot == "3"
        assert t.well == "B2"
        assert t.action == "aspirate"
        assert t.volume == 50.0
        assert t.pipette_mount == "right"
        assert t.offset.as_tuple() == (0.0, 0.0, 0.0)

    def test_custom_offset_and_mount(self) -> None:
        off = Offset(x=1.0, y=2.0, z=3.0)
        t = build_target(
            "dst", "5", "A1", "dispense",
            pipette_mount="left", offset=off,
        )
        assert t.pipette_mount == "left"
        assert t.offset.as_tuple() == (1.0, 2.0, 3.0)


# ---------------------------------------------------------------------------
# preview_target
# ---------------------------------------------------------------------------


class TestPreviewTarget:
    def test_calls_move_to_well(self) -> None:
        client = _make_client()
        target = build_target("t", "3", "A1", "aspirate", pipette_mount="left")
        preview_target(client, target)

        client.use_pipette.assert_called_once_with("left")
        client.get_labware_id.assert_called_once_with("3")
        client.move_to_well.assert_called_once_with(
            labware_id="labware-123",
            well="A1",
            offset=(0.0, 0.0, 0.0),
        )


# ---------------------------------------------------------------------------
# test_aspirate
# ---------------------------------------------------------------------------


class TestTestAspirate:
    def test_uses_explicit_volume(self) -> None:
        client = _make_client()
        target = build_target(
            "t", "3", "A1", "aspirate", volume=100.0, pipette_mount="right"
        )
        test_aspirate(client, target, volume=50.0)

        client.use_pipette.assert_called_once_with("right")
        client.aspirate.assert_called_once_with(
            volume=50.0,
            labware_id="labware-123",
            well="A1",
            offset=(0.0, 0.0, 0.0),
        )

    def test_falls_back_to_target_volume(self) -> None:
        client = _make_client()
        target = build_target("t", "3", "A1", "aspirate", volume=75.0)
        test_aspirate(client, target)

        client.aspirate.assert_called_once()
        assert client.aspirate.call_args.kwargs["volume"] == 75.0

    def test_raises_without_volume(self) -> None:
        client = _make_client()
        target = build_target("t", "3", "A1", "aspirate")
        with pytest.raises(ValueError, match="No volume"):
            test_aspirate(client, target)


# ---------------------------------------------------------------------------
# test_dispense
# ---------------------------------------------------------------------------


class TestTestDispense:
    def _src(self, volume: float | None = 50.0) -> CalibrationTarget:
        return build_target("src", "3", "A1", "aspirate", volume=volume)

    def _dst(self, volume: float | None = None) -> CalibrationTarget:
        return build_target("dst", "5", "B2", "dispense", volume=volume)

    def test_source_to_dest_flow(self) -> None:
        """Full aspirate→dispense→blow_out path is exercised."""
        client = _make_client()
        # get_labware_id returns different IDs per slot
        client.get_labware_id.side_effect = lambda s: f"lw-{s}"
        src = build_target("src", "3", "A1", "aspirate", volume=50.0, pipette_mount="right")
        dst = build_target("dst", "5", "B2", "dispense", pipette_mount="right")

        test_dispense(client, dst, source_target=src)

        client.use_pipette.assert_called_once_with("right")
        client.aspirate.assert_called_once_with(
            volume=50.0,
            labware_id="lw-3",
            well="A1",
            offset=(0.0, 0.0, 0.0),
        )
        client.dispense.assert_called_once_with(
            volume=50.0,
            labware_id="lw-5",
            well="B2",
            offset=(0.0, 0.0, 0.0),
        )
        client.blow_out.assert_called_once_with(
            labware_id="lw-5",
            well="B2",
            offset=(0.0, 0.0, 0.0),
        )

    def test_rejects_cross_mount_dispense(self) -> None:
        client = _make_client()
        src = build_target("src", "3", "A1", "aspirate", volume=50.0, pipette_mount="left")
        dst = build_target("dst", "5", "B2", "dispense", pipette_mount="right")

        with pytest.raises(ValueError, match="same pipette mount"):
            test_dispense(client, dst, source_target=src)

    def test_explicit_volume_overrides(self) -> None:
        """Explicit volume param takes precedence over target volumes."""
        client = _make_client()
        src = self._src(volume=100.0)
        dst = self._dst(volume=80.0)

        test_dispense(client, dst, source_target=src, volume=25.0)

        assert client.aspirate.call_args.kwargs["volume"] == 25.0
        assert client.dispense.call_args.kwargs["volume"] == 25.0

    def test_falls_back_to_dest_volume(self) -> None:
        """Falls back to target.volume when no explicit volume."""
        client = _make_client()
        src = self._src(volume=None)
        dst = self._dst(volume=60.0)

        test_dispense(client, dst, source_target=src)

        assert client.aspirate.call_args.kwargs["volume"] == 60.0

    def test_falls_back_to_source_volume(self) -> None:
        """Falls back to source_target.volume as last resort."""
        client = _make_client()
        src = self._src(volume=75.0)
        dst = self._dst(volume=None)

        test_dispense(client, dst, source_target=src)

        assert client.aspirate.call_args.kwargs["volume"] == 75.0

    def test_raises_without_volume(self) -> None:
        client = _make_client()
        src = self._src(volume=None)
        dst = self._dst(volume=None)
        with pytest.raises(ValueError, match="No volume"):
            test_dispense(client, dst, source_target=src)


class TestTipCalibration:
    def test_pick_up_tip_uses_target_mount(self) -> None:
        client = _make_client()
        target = build_target("tips", "11", "A1", "pick_up_tip", pipette_mount="right")

        test_pick_up_tip(client, target)

        client.use_pipette.assert_called_once_with("right")
        client.pick_up_tip.assert_called_once_with(
            labware_id="labware-123",
            well="A1",
            offset=(0.0, 0.0, 0.0),
        )
        client.drop_tip_in_trash.assert_not_called()

    def test_drop_tip_uses_target_mount(self) -> None:
        client = _make_client()
        target = build_target("tips", "10", "A1", "pick_up_tip", pipette_mount="left")

        test_drop_tip(client, target)

        client.use_pipette.assert_called_once_with("left")
        client.get_labware_id.assert_called_once_with("10")
        client.drop_tip.assert_called_once_with(
            labware_id="labware-123",
            well="A1",
            offset=(0.0, 0.0, 0.0),
        )


# ---------------------------------------------------------------------------
# save_profile / load_profile round-trip
# ---------------------------------------------------------------------------


class TestProfilePersistence:
    def test_round_trip(self, tmp_path: Path) -> None:
        target = build_target("src", "3", "A1", "aspirate", volume=50.0)
        profile = CalibrationProfile(
            name="demo",
            targets=[target],
            metadata={"author": "test"},
        )
        fp = tmp_path / "profile.json"
        save_profile(profile, fp)

        loaded = load_profile(fp)
        assert loaded.name == profile.name
        assert loaded.id == profile.id
        assert len(loaded.targets) == 1
        assert loaded.targets[0].name == "src"
        assert loaded.metadata == {"author": "test"}

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        fp = tmp_path / "sub" / "dir" / "profile.json"
        profile = CalibrationProfile(name="nested")
        save_profile(profile, fp)
        assert fp.exists()


# ---------------------------------------------------------------------------
# CalibrationSession
# ---------------------------------------------------------------------------


class TestCalibrationSession:
    def test_defaults(self) -> None:
        s = CalibrationSession()
        assert s.status == "idle"
        assert s.current_target_index == 0
        assert s.notes == []
        assert s.events == []
        assert s.profile_id is None
        assert isinstance(s.id, str)
        assert len(s.id) > 0

    def test_custom_values(self) -> None:
        s = CalibrationSession(
            profile_id="p-123",
            status="active",
            current_target_index=2,
            notes=["adjusted z"],
            events=["started"],
        )
        assert s.profile_id == "p-123"
        assert s.status == "active"
        assert s.current_target_index == 2
        assert s.notes == ["adjusted z"]
        assert s.events == ["started"]

    def test_invalid_status_rejected(self) -> None:
        with pytest.raises(Exception):
            CalibrationSession(status="bogus")  # type: ignore[arg-type]
