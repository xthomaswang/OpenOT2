"""Tests for OT2Client (mocked HTTP, no real hardware)."""

from __future__ import annotations

import json

import pytest
import responses

from openot2.client import OT2Client

ROBOT_IP = "127.0.0.1"
BASE_URL = f"http://{ROBOT_IP}:31950"


class TestCreateRun:
    def test_returns_run_id(self, mock_client, mock_ot2_server):
        assert mock_client.run_id == mock_ot2_server["run_id"]

    def test_sets_internal_state(self, mock_client):
        assert mock_client._commands_url is not None
        assert mock_client.run_id in mock_client._commands_url


class TestReconnect:
    def test_reconnect_recovers_state(self, mock_ot2_server):
        client = OT2Client(ROBOT_IP)
        run_id = client.reconnect_last_run()
        assert run_id == mock_ot2_server["run_id"]
        assert client.pipette_id == mock_ot2_server["pipette_id"]
        assert "11" in client.labware_by_slot

    def test_reconnect_no_runs_raises(self):
        with responses.RequestsMock() as rsps:
            rsps.add(responses.GET, f"{BASE_URL}/runs", json={"data": []}, status=200)
            client = OT2Client(ROBOT_IP)
            with pytest.raises(RuntimeError, match="No runs found"):
                client.reconnect_last_run()


class TestLoadEquipment:
    def test_load_pipette(self, mock_client, mock_ot2_server):
        pid = mock_client.load_pipette("p300_multi_gen2", mount="right")
        assert pid == mock_ot2_server["pipette_id"]
        assert mock_client.pipette_id == pid

    def test_load_labware(self, mock_client):
        lid = mock_client.load_labware("opentrons_96_tiprack_300ul", slot="11")
        assert lid is not None
        assert mock_client.get_labware_id("11") == lid

    def test_get_labware_id_missing_raises(self, mock_client):
        with pytest.raises(KeyError, match="No labware recorded"):
            mock_client.get_labware_id("99")


class TestPipetteOps:
    def test_pick_up_tip_sends_correct_command(self, mock_client, mock_ot2_server):
        mock_client.load_pipette("p300_multi_gen2")
        mock_client.pick_up_tip("labware-123", well="B2", offset=(1.0, 2.0, 3.0))

        # Check the last request body
        last_req = mock_ot2_server["rsps"].calls[-1].request
        body = json.loads(last_req.body)
        assert body["data"]["commandType"] == "pickUpTip"
        assert body["data"]["params"]["wellName"] == "B2"
        assert body["data"]["params"]["wellLocation"]["offset"] == {"x": 1.0, "y": 2.0, "z": 3.0}

    def test_aspirate_without_pipette_raises(self):
        with responses.RequestsMock() as rsps:
            rsps.add(responses.POST, f"{BASE_URL}/runs", json={"data": {"id": "r1"}}, status=200)
            client = OT2Client(ROBOT_IP)
            client.create_run()
            with pytest.raises(RuntimeError, match="No pipette loaded"):
                client.aspirate(100, "lw1", "A1")

    def test_aspirate_sends_correct_command(self, mock_client, mock_ot2_server):
        mock_client.load_pipette("p300_multi_gen2")
        mock_client.aspirate(150, "lw-1", "A3", origin="top", flow_rate=200.0)

        last_req = mock_ot2_server["rsps"].calls[-1].request
        body = json.loads(last_req.body)
        assert body["data"]["commandType"] == "aspirate"
        assert body["data"]["params"]["volume"] == 150
        assert body["data"]["params"]["wellName"] == "A3"
        assert body["data"]["params"]["wellLocation"]["origin"] == "top"
        assert body["data"]["params"]["flowRate"] == 200.0

    def test_dispense_sends_correct_command(self, mock_client, mock_ot2_server):
        mock_client.load_pipette("p300_multi_gen2")
        mock_client.dispense(100, "lw-1", "B1")

        last_req = mock_ot2_server["rsps"].calls[-1].request
        body = json.loads(last_req.body)
        assert body["data"]["commandType"] == "dispense"
        assert body["data"]["params"]["volume"] == 100

    def test_blow_out_requires_both_or_neither(self, mock_client):
        mock_client.load_pipette("p300_multi_gen2")
        with pytest.raises(ValueError, match="both labware_id and well"):
            mock_client.blow_out(labware_id="lw-1")

    def test_home_sends_command(self, mock_client, mock_ot2_server):
        mock_client.home()
        last_req = mock_ot2_server["rsps"].calls[-1].request
        body = json.loads(last_req.body)
        assert body["data"]["commandType"] == "home"

    def test_no_run_raises(self):
        client = OT2Client(ROBOT_IP)
        with pytest.raises(RuntimeError, match="No active run"):
            client._post_command({"data": {}})


class TestRetry:
    def test_retry_adapter_configured(self):
        """Verify that the retry adapter is properly configured on the session."""
        client = OT2Client(ROBOT_IP, max_retries=5)
        adapter = client._session.get_adapter("http://")
        assert adapter.max_retries.total == 5
        assert 502 in adapter.max_retries.status_forcelist
        assert 503 in adapter.max_retries.status_forcelist
        assert 504 in adapter.max_retries.status_forcelist

    def test_http_error_raises(self):
        """Non-retryable errors should raise immediately."""
        with responses.RequestsMock() as rsps:
            rsps.add(responses.POST, f"{BASE_URL}/runs", json={"data": {"id": "r1"}}, status=200)
            rsps.add(
                responses.POST,
                f"{BASE_URL}/runs/r1/commands",
                json={"error": "not found"},
                status=404,
            )
            client = OT2Client(ROBOT_IP)
            client.create_run()
            with pytest.raises(Exception):
                client._post_command({"data": {"commandType": "home", "params": {}}})
