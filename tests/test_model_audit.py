"""
Tests for model_audit.py - configured models vs. what providers actually serve.

Regression context: Groq retired llama-4-scout and llama-4-maverick while both
were still named in the routing chain, and Together retired the Maverick FP8
build that backed two aliases. Nothing noticed until a human checked by hand.
"""

import os

import pytest
import yaml
from hypothesis import given, settings
from hypothesis import strategies as st

from hooks.model_audit import (
    DEFAULT_PROVIDER,
    audit_models,
    describe_missing,
    extract_available_models,
    find_dangling_fallbacks,
    load_configured_models,
    parse_configured_models,
    resolve_config_path,
    split_model_string,
)

# =============================================================================
# split_model_string
# =============================================================================


class TestSplitModelString:
    """Tests for LiteLLM model string parsing."""

    @pytest.mark.parametrize(
        "model,expected",
        [
            ("deepinfra/Qwen/Qwen3.6-35B-A3B", ("deepinfra", "Qwen/Qwen3.6-35B-A3B")),
            ("groq/qwen/qwen3.6-27b", ("groq", "qwen/qwen3.6-27b")),
            ("openrouter/qwen/qwen3.6-35b-a3b", ("openrouter", "qwen/qwen3.6-35b-a3b")),
            (
                "together_ai/meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
                ("together", "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"),
            ),
            # Unprefixed models are OpenAI's
            ("gpt-4o-mini", (DEFAULT_PROVIDER, "gpt-4o-mini")),
            # Unknown prefix is not silently treated as a provider
            ("bedrock/anthropic.claude", (DEFAULT_PROVIDER, "bedrock/anthropic.claude")),
        ],
    )
    def test_splits_known_prefixes(self, model, expected):
        assert split_model_string(model) == expected

    @given(model=st.text(min_size=1, max_size=80))
    @settings(max_examples=100)
    def test_never_raises(self, model):
        """Property: any string parses into a (provider, model) pair."""
        provider, upstream = split_model_string(model)
        assert isinstance(provider, str) and provider
        assert isinstance(upstream, str)


# =============================================================================
# parse_configured_models
# =============================================================================


class TestParseConfiguredModels:
    """Tests for extracting the configured model set from a LiteLLM config."""

    def test_groups_by_provider_and_collects_aliases(self):
        config = {
            "model_list": [
                {
                    "model_name": "default",
                    "litellm_params": {"model": "deepinfra/Qwen/Qwen3.6-35B-A3B"},
                },
                {
                    "model_name": "qwen-3.6-35b",
                    "litellm_params": {"model": "deepinfra/Qwen/Qwen3.6-35B-A3B"},
                },
                {
                    "model_name": "fast",
                    "litellm_params": {"model": "groq/qwen/qwen3.6-27b"},
                },
            ]
        }

        configured = parse_configured_models(config)

        assert configured["deepinfra"] == {
            "Qwen/Qwen3.6-35B-A3B": ["default", "qwen-3.6-35b"]
        }
        assert configured["groq"] == {"qwen/qwen3.6-27b": ["fast"]}

    def test_ignores_malformed_entries(self):
        config = {
            "model_list": [
                {"model_name": "no-params"},
                {"model_name": "empty", "litellm_params": {}},
                {"model_name": "not-a-string", "litellm_params": {"model": 42}},
                {"model_name": "ok", "litellm_params": {"model": "groq/qwen/qwen3.6-27b"}},
            ]
        }

        assert parse_configured_models(config) == {"groq": {"qwen/qwen3.6-27b": ["ok"]}}

    @pytest.mark.parametrize("config", [{}, {"model_list": None}, {"model_list": []}])
    def test_empty_configs(self, config):
        assert parse_configured_models(config) == {}


# =============================================================================
# extract_available_models
# =============================================================================


class TestExtractAvailableModels:
    """Tests for parsing provider /models responses."""

    def test_openai_style_envelope(self):
        """Groq, OpenRouter and DeepInfra all use {"data": [...]}."""
        payload = {"data": [{"id": "a"}, {"id": "b"}]}
        assert extract_available_models(payload) == {"a", "b"}

    def test_bare_array(self):
        """Together AI returns a top-level array."""
        payload = [{"id": "a"}, {"id": "b"}]
        assert extract_available_models(payload) == {"a", "b"}

    def test_skips_entries_without_string_ids(self):
        payload = {"data": [{"id": "a"}, {"name": "b"}, {"id": 3}]}
        assert extract_available_models(payload) == {"a"}

    @pytest.mark.parametrize(
        "payload",
        [None, "text", 42, {}, {"data": []}, {"data": "nope"}, [], [{"name": "x"}]],
    )
    def test_unusable_payloads_return_none(self, payload):
        """None means 'not audited' — never 'nothing available'."""
        assert extract_available_models(payload) is None


# =============================================================================
# audit_models
# =============================================================================


class TestAuditModels:
    """Tests for the configured-vs-available comparison."""

    def test_all_available(self):
        audit = audit_models({"a": ["alias-a"], "b": ["alias-b"]}, {"a", "b", "c"})

        assert audit == {"configured": 2, "available": 2, "missing": []}

    def test_reports_missing_with_aliases(self):
        """The alias list is what makes the failure actionable."""
        configured = {
            "meta-llama/llama-4-scout-17b-16e-instruct": ["llama-4-scout", "fast"],
            "openai/gpt-oss-20b": ["gpt-oss-20b"],
        }
        audit = audit_models(configured, {"openai/gpt-oss-20b"})

        assert audit["configured"] == 2
        assert audit["available"] == 1
        assert audit["missing"] == [
            {
                "model": "meta-llama/llama-4-scout-17b-16e-instruct",
                "aliases": ["llama-4-scout", "fast"],
            }
        ]

    def test_unusable_response_is_not_an_audit(self):
        assert audit_models({"a": ["x"]}, None) is None

    def test_nothing_configured_is_not_an_audit(self):
        assert audit_models({}, {"a"}) is None

    @given(
        configured=st.dictionaries(
            st.text(min_size=1, max_size=20), st.just(["alias"]), min_size=1, max_size=8
        ),
        available=st.sets(st.text(min_size=1, max_size=20), max_size=8),
    )
    @settings(max_examples=100)
    def test_property_counts_are_consistent(self, configured, available):
        """Property: available + missing always equals configured."""
        audit = audit_models(configured, available or None)
        if audit is None:
            return
        assert audit["available"] + len(audit["missing"]) == audit["configured"]
        assert audit["configured"] == len(configured)


class TestDescribeMissing:
    """Tests for the human-readable summary attached to a status error."""

    def test_singular(self):
        audit = {"missing": [{"model": "a", "aliases": []}]}
        assert describe_missing(audit) == "1 configured model unavailable: a"

    def test_plural(self):
        audit = {"missing": [{"model": "a", "aliases": []}, {"model": "b", "aliases": []}]}
        assert describe_missing(audit) == "2 configured models unavailable: a, b"


# =============================================================================
# find_dangling_fallbacks
# =============================================================================


class TestFindDanglingFallbacks:
    """Removing a model must not leave fallback chains pointing at nothing."""

    @staticmethod
    def _config(fallbacks, aliases=("default", "backup")):
        return {
            "model_list": [
                {"model_name": a, "litellm_params": {"model": f"groq/{a}"}} for a in aliases
            ],
            "router_settings": {"fallbacks": fallbacks},
        }

    def test_clean_config(self):
        config = self._config([{"default": ["backup"]}])
        assert find_dangling_fallbacks(config) == []

    def test_detects_missing_target(self):
        """The failure mode: chain names a model that was deleted."""
        config = self._config([{"default": ["backup", "groq/llama-4-scout"]}])

        assert find_dangling_fallbacks(config) == [
            {
                "source": "default",
                "undefined_source": False,
                "missing_targets": ["groq/llama-4-scout"],
            }
        ]

    def test_detects_undefined_source(self):
        config = self._config([{"llama-4-scout": ["backup"]}])

        assert find_dangling_fallbacks(config) == [
            {"source": "llama-4-scout", "undefined_source": True, "missing_targets": []}
        ]

    @pytest.mark.parametrize(
        "fallbacks", [None, [], ["not-a-dict"], [{"default": None}], [{"default": []}]]
    )
    def test_tolerates_odd_shapes(self, fallbacks):
        assert find_dangling_fallbacks(self._config(fallbacks)) == []

    def test_no_router_settings(self):
        assert find_dangling_fallbacks({"model_list": []}) == []

    def test_shipped_config_is_clean(self):
        """The config we ship must never name an alias it does not define."""
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "litellm_config.yaml")
        with open(path, "r") as f:
            config = yaml.safe_load(f)

        assert find_dangling_fallbacks(config) == []


# =============================================================================
# Config loading
# =============================================================================


class TestConfigLoading:
    """Tests for locating and loading the active config."""

    def test_resolve_explicit_path(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("model_list: []\n")

        assert resolve_config_path(str(config_file)) == str(config_file)

    def test_resolve_missing_explicit_path(self, tmp_path):
        assert resolve_config_path(str(tmp_path / "nope.yaml")) is None

    def test_resolve_falls_back_to_repo_config(self):
        """A source checkout resolves without any env var set."""
        resolved = resolve_config_path()
        assert resolved is not None
        assert resolved.endswith(".yaml")

    def test_load_returns_configured_models(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            yaml.safe_dump(
                {
                    "model_list": [
                        {
                            "model_name": "default",
                            "litellm_params": {"model": "groq/qwen/qwen3.6-27b"},
                        }
                    ]
                }
            )
        )

        assert load_configured_models(str(config_file)) == {
            "groq": {"qwen/qwen3.6-27b": ["default"]}
        }

    def test_unreadable_config_is_not_a_failure(self, tmp_path):
        """An audit we cannot perform must not be reported as a defect."""
        assert load_configured_models(str(tmp_path / "missing.yaml")) == {}

    def test_malformed_yaml_is_not_a_failure(self, tmp_path):
        config_file = tmp_path / "bad.yaml"
        config_file.write_text("model_list: [unclosed\n")

        assert load_configured_models(str(config_file)) == {}

    def test_shipped_config_matches_monitored_providers(self):
        """Every provider we route to must also be health-checked (issue #7)."""
        from hooks.status_handler import PROVIDERS

        configured = load_configured_models()
        routed = {p for p in configured if p != DEFAULT_PROVIDER}
        monitored = {p for p, c in PROVIDERS.items() if c["type"] == "llm"}

        assert routed <= monitored, f"routed but unmonitored: {routed - monitored}"
