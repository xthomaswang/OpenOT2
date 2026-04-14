"""Tests for LLM protocol generator (mocked LLM API)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from openot2.protocol.generator import (
    DryRunResult,
    ProtocolGenerator,
    get_protocol_prompt,
)


class TestGetProtocolPrompt:
    def test_returns_string(self):
        prompt = get_protocol_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 100

    def test_contains_format_description(self):
        prompt = get_protocol_prompt()
        assert "labware" in prompt
        assert "tasks" in prompt
        assert "pickup" in prompt
        assert "transfer" in prompt
        assert "pipette" in prompt

    def test_contains_labware_examples(self):
        prompt = get_protocol_prompt()
        assert "p300_multi_gen2" in prompt
        assert "opentrons_96_tiprack_300ul" in prompt


class TestValidate:
    def test_valid_config(self, sample_config):
        errors = ProtocolGenerator.validate(sample_config)
        assert errors == []

    def test_missing_tasks(self):
        config = {
            "labware": {
                "pipette": {"name": "p300", "mount": "right"},
                "tiprack": {"name": "rack", "slot": "11"},
            },
            "settings": {},
            # missing "tasks"
        }
        errors = ProtocolGenerator.validate(config)
        assert len(errors) > 0

    def test_empty_tasks_is_valid(self):
        config = {
            "labware": {
                "pipette": {"name": "p300", "mount": "right"},
                "tiprack": {"name": "rack", "slot": "11"},
            },
            "settings": {},
            "tasks": [],
        }
        errors = ProtocolGenerator.validate(config)
        assert errors == []


class TestDryRun:
    def test_simple_valid_protocol(self, sample_config):
        result = ProtocolGenerator.dry_run(sample_config)
        assert isinstance(result, DryRunResult)
        assert result.success is True

    def test_empty_tasks(self):
        config = {
            "labware": {"pipette": {"name": "p300"}, "tiprack": {"name": "r", "slot": "1"}},
            "settings": {},
            "tasks": [],
        }
        result = ProtocolGenerator.dry_run(config)
        assert result.success is True
        assert len(result.errors) == 0


class TestGenerate:
    def test_generate_with_mock_llm(self):
        """Test generate() with a mocked OpenAI client."""
        sample_response = json.dumps({
            "labware": {
                "pipette": {"name": "p300_multi_gen2", "mount": "right"},
                "tiprack": {"name": "opentrons_96_tiprack_300ul", "slot": "11"},
                "sources": {"name": "nest_12_reservoir_15ml", "slots": ["8"]},
                "dispense": {"name": "corning_96_wellplate_360ul_flat", "slot": "10"},
            },
            "settings": {"imaging_well": "A1", "base_dir": "output"},
            "tasks": [
                {"type": "pickup", "well": "A1"},
                {"type": "transfer", "source_slot": "8", "source_well": "A1",
                 "dest_slot": "10", "dest_well": "A1", "volume": 100},
                {"type": "drop", "well": "A1"},
            ],
        })

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = sample_response

        mock_openai = MagicMock()
        mock_openai.return_value.chat.completions.create.return_value = mock_response

        with patch("openot2.protocol.generator.ProtocolGenerator.__init__", return_value=None):
            gen = ProtocolGenerator.__new__(ProtocolGenerator)
            gen._client = mock_openai()
            gen._model = "gpt-4o"

            config = gen.generate("Transfer 100uL from reservoir to plate")

        assert "labware" in config
        assert "tasks" in config
        assert len(config["tasks"]) == 3

    def test_generate_strips_markdown_fences(self):
        """LLM sometimes wraps JSON in ```json ... ``` — generator should handle it."""
        raw_json = {
            "labware": {"pipette": {"name": "p300"}, "tiprack": {"name": "r", "slot": "1"}},
            "settings": {},
            "tasks": [],
        }
        fenced = f"```json\n{json.dumps(raw_json)}\n```"

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = fenced

        with patch("openot2.protocol.generator.ProtocolGenerator.__init__", return_value=None):
            gen = ProtocolGenerator.__new__(ProtocolGenerator)
            gen._client = MagicMock()
            gen._client.chat.completions.create.return_value = mock_response
            gen._model = "gpt-4o"

            config = gen.generate("Simple test")

        assert config["labware"]["pipette"]["name"] == "p300"

    def test_generate_invalid_json_raises(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "This is not JSON at all"

        with patch("openot2.protocol.generator.ProtocolGenerator.__init__", return_value=None):
            gen = ProtocolGenerator.__new__(ProtocolGenerator)
            gen._client = MagicMock()
            gen._client.chat.completions.create.return_value = mock_response
            gen._model = "gpt-4o"

            with pytest.raises(ValueError, match="Failed to parse"):
                gen.generate("Test")


class TestGenerateAndValidate:
    def test_end_to_end_with_mock(self):
        valid_config = {
            "labware": {
                "pipette": {"name": "p300_multi_gen2", "mount": "right"},
                "tiprack": {"name": "opentrons_96_tiprack_300ul", "slot": "11"},
                "sources": {"name": "nest_12_reservoir_15ml", "slots": ["8"]},
                "dispense": {"name": "corning_96_wellplate_360ul_flat", "slot": "10"},
            },
            "settings": {"imaging_well": "A1"},
            "tasks": [
                {"type": "pickup", "well": "A1"},
                {"type": "transfer", "source_slot": "8", "source_well": "A1",
                 "dest_slot": "10", "dest_well": "A1", "volume": 100},
                {"type": "drop", "well": "A1"},
            ],
        }

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(valid_config)

        with patch("openot2.protocol.generator.ProtocolGenerator.__init__", return_value=None):
            gen = ProtocolGenerator.__new__(ProtocolGenerator)
            gen._client = MagicMock()
            gen._client.chat.completions.create.return_value = mock_response
            gen._model = "gpt-4o"

            config, result = gen.generate_and_validate("Transfer 100uL")

        assert "labware" in config
        assert result.success is True
        assert len(result.errors) == 0
