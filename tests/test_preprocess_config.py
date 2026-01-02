"""Tests for config preprocessor."""

import os
import tempfile
import yaml
import pytest

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from preprocess_config import preprocess_config


@pytest.fixture
def sample_config():
    """Sample LiteLLM config with OpenRouter models."""
    return {
        "model_list": [
            {
                "model_name": "groq/llama-4",
                "litellm_params": {
                    "model": "groq/llama-4",
                    "api_key": "os.environ/GROQ_API_KEY",
                },
            },
            {
                "model_name": "openrouter/llama-4",
                "litellm_params": {
                    "model": "openrouter/meta-llama/llama-4",
                    "api_key": "os.environ/OPENROUTER_API_KEY",
                },
            },
            {
                "model_name": "default",
                "litellm_params": {
                    "model": "openrouter/meta-llama/llama-4",
                    "api_key": "os.environ/OPENROUTER_API_KEY",
                },
            },
        ],
    }


class TestPreprocessConfig:
    """Tests for preprocess_config function."""

    def test_applies_ignore_providers_to_openrouter_models(self, sample_config, monkeypatch):
        """Should add ignore list to all OpenRouter models."""
        monkeypatch.setenv("OPENROUTER_IGNORE_PROVIDERS", "Friendli,Google")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(sample_config, f)
            input_path = f.name

        output_path = f.name.replace(".yaml", ".processed.yaml")

        try:
            preprocess_config(input_path, output_path)

            with open(output_path) as f:
                result = yaml.safe_load(f)

            # OpenRouter models should have ignore list
            openrouter_model = result["model_list"][1]["litellm_params"]
            assert openrouter_model["extra_body"]["provider"]["ignore"] == ["Friendli", "Google"]

            default_model = result["model_list"][2]["litellm_params"]
            assert default_model["extra_body"]["provider"]["ignore"] == ["Friendli", "Google"]

            # Non-OpenRouter models should be unchanged
            groq_model = result["model_list"][0]["litellm_params"]
            assert "extra_body" not in groq_model
        finally:
            os.unlink(input_path)
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_handles_spaces_in_provider_list(self, sample_config, monkeypatch):
        """Should trim whitespace from provider names."""
        monkeypatch.setenv("OPENROUTER_IGNORE_PROVIDERS", "Friendli , Google , DeepInfra")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(sample_config, f)
            input_path = f.name

        output_path = f.name.replace(".yaml", ".processed.yaml")

        try:
            preprocess_config(input_path, output_path)

            with open(output_path) as f:
                result = yaml.safe_load(f)

            openrouter_model = result["model_list"][1]["litellm_params"]
            assert openrouter_model["extra_body"]["provider"]["ignore"] == [
                "Friendli",
                "Google",
                "DeepInfra",
            ]
        finally:
            os.unlink(input_path)
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_no_changes_when_env_var_not_set(self, sample_config, monkeypatch):
        """Should pass through config unchanged when env var is empty."""
        monkeypatch.delenv("OPENROUTER_IGNORE_PROVIDERS", raising=False)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(sample_config, f)
            input_path = f.name

        output_path = f.name.replace(".yaml", ".processed.yaml")

        try:
            preprocess_config(input_path, output_path)

            with open(output_path) as f:
                result = yaml.safe_load(f)

            # Models should be unchanged (no extra_body added)
            openrouter_model = result["model_list"][1]["litellm_params"]
            assert "extra_body" not in openrouter_model
        finally:
            os.unlink(input_path)
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_preserves_existing_extra_body(self, sample_config, monkeypatch):
        """Should merge with existing extra_body settings."""
        monkeypatch.setenv("OPENROUTER_IGNORE_PROVIDERS", "Friendli")

        # Add existing extra_body
        sample_config["model_list"][1]["litellm_params"]["extra_body"] = {
            "transforms": ["middle-out"]
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(sample_config, f)
            input_path = f.name

        output_path = f.name.replace(".yaml", ".processed.yaml")

        try:
            preprocess_config(input_path, output_path)

            with open(output_path) as f:
                result = yaml.safe_load(f)

            openrouter_model = result["model_list"][1]["litellm_params"]
            # Should have both transforms and provider.ignore
            assert openrouter_model["extra_body"]["transforms"] == ["middle-out"]
            assert openrouter_model["extra_body"]["provider"]["ignore"] == ["Friendli"]
        finally:
            os.unlink(input_path)
            if os.path.exists(output_path):
                os.unlink(output_path)
