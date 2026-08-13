"""
Tests for hooks/verify_models.py - the model availability CLI.

This CLI is a CI gate (the `config-audit` job and the daily
model-availability workflow), so its exit codes and its skip-vs-fail
semantics are load-bearing: a false failure blocks every PR, and a false pass
is the silent retirement it exists to catch.
"""

import os
from unittest.mock import patch

import httpx
import pytest
import respx
import yaml

from hooks.status_handler import PROVIDERS
from hooks.verify_models import (
    audit_provider,
    fetch_available,
    main,
    report,
    report_fallbacks,
    run,
)

GROQ_URL = PROVIDERS["groq"]["check_url"]


def _models(*ids):
    return httpx.Response(200, json={"data": [{"id": i} for i in ids]})


# =============================================================================
# fetch_available
# =============================================================================


class TestFetchAvailable:
    """Tests for the provider /models fetch."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_returns_model_ids(self):
        respx.get(GROQ_URL).mock(return_value=_models("a", "b"))

        with patch.dict(os.environ, {"GROQ_API_KEY": "secret"}):
            async with httpx.AsyncClient() as client:
                assert await fetch_available(client, PROVIDERS["groq"]) == {"a", "b"}

    @pytest.mark.asyncio
    @respx.mock
    async def test_sends_auth_header_when_key_present(self):
        route = respx.get(GROQ_URL).mock(return_value=_models("a"))

        with patch.dict(os.environ, {"GROQ_API_KEY": "secret"}):
            async with httpx.AsyncClient() as client:
                await fetch_available(client, PROVIDERS["groq"])

        assert route.calls.last.request.headers["Authorization"] == "Bearer secret"

    @pytest.mark.asyncio
    @respx.mock
    async def test_omits_auth_header_without_key(self):
        """Some providers publish the list unauthenticated — still worth asking."""
        route = respx.get(GROQ_URL).mock(return_value=_models("a"))

        with patch.dict(os.environ, {"GROQ_API_KEY": ""}):
            async with httpx.AsyncClient() as client:
                await fetch_available(client, PROVIDERS["groq"])

        assert "authorization" not in route.calls.last.request.headers

    @pytest.mark.asyncio
    @respx.mock
    async def test_raises_on_http_error(self):
        respx.get(GROQ_URL).mock(return_value=httpx.Response(401))

        with patch.dict(os.environ, {"GROQ_API_KEY": "bad"}):
            async with httpx.AsyncClient() as client:
                with pytest.raises(httpx.HTTPStatusError):
                    await fetch_available(client, PROVIDERS["groq"])


# =============================================================================
# audit_provider
# =============================================================================


class TestAuditProvider:
    """Skip-vs-fail semantics: an audit we could not perform is not a defect."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_reports_missing_models(self):
        respx.get(GROQ_URL).mock(return_value=_models("qwen/qwen3.6-27b"))

        with patch.dict(os.environ, {"GROQ_API_KEY": "secret"}):
            async with httpx.AsyncClient() as client:
                result = await audit_provider(
                    client,
                    "groq",
                    PROVIDERS["groq"],
                    {"qwen/qwen3.6-27b": ["fast"], "gone": ["llama-4-scout"]},
                )

        assert result["provider"] == "groq"
        assert result["missing"] == [{"model": "gone", "aliases": ["llama-4-scout"]}]

    @pytest.mark.asyncio
    @respx.mock
    async def test_clean_provider(self):
        respx.get(GROQ_URL).mock(return_value=_models("qwen/qwen3.6-27b"))

        with patch.dict(os.environ, {"GROQ_API_KEY": "secret"}):
            async with httpx.AsyncClient() as client:
                result = await audit_provider(
                    client, "groq", PROVIDERS["groq"], {"qwen/qwen3.6-27b": ["fast"]}
                )

        assert result["missing"] == []
        assert result["available"] == 1

    @pytest.mark.asyncio
    @respx.mock
    async def test_missing_key_is_a_skip_naming_the_variable(self):
        respx.get(GROQ_URL).mock(return_value=httpx.Response(401))

        with patch.dict(os.environ, {"GROQ_API_KEY": ""}):
            async with httpx.AsyncClient() as client:
                result = await audit_provider(
                    client, "groq", PROVIDERS["groq"], {"m": ["alias"]}
                )

        assert result["skipped"] == "GROQ_API_KEY not set"
        assert "missing" not in result

    @pytest.mark.asyncio
    @respx.mock
    async def test_transport_failure_with_key_is_a_skip(self):
        respx.get(GROQ_URL).mock(side_effect=httpx.ConnectError("boom"))

        with patch.dict(os.environ, {"GROQ_API_KEY": "secret"}):
            async with httpx.AsyncClient() as client:
                result = await audit_provider(
                    client, "groq", PROVIDERS["groq"], {"m": ["alias"]}
                )

        assert "ConnectError" in result["skipped"]

    @pytest.mark.asyncio
    @respx.mock
    async def test_unusable_body_is_a_skip(self):
        respx.get(GROQ_URL).mock(return_value=httpx.Response(200, json={"data": []}))

        with patch.dict(os.environ, {"GROQ_API_KEY": "secret"}):
            async with httpx.AsyncClient() as client:
                result = await audit_provider(
                    client, "groq", PROVIDERS["groq"], {"m": ["alias"]}
                )

        assert result["skipped"] == "no usable /models response"


# =============================================================================
# run
# =============================================================================


class TestRun:
    """Only LLM providers with configured models are audited."""

    @pytest.mark.asyncio
    async def test_skips_billing_and_unconfigured_providers(self, tmp_path):
        config = tmp_path / "config.yaml"
        config.write_text(
            yaml.safe_dump(
                {
                    "model_list": [
                        {
                            "model_name": "fast",
                            "litellm_params": {"model": "groq/qwen/qwen3.6-27b"},
                        }
                    ]
                }
            )
        )

        with respx.mock:
            respx.get(GROQ_URL).mock(return_value=_models("qwen/qwen3.6-27b"))
            with patch.dict(os.environ, {"GROQ_API_KEY": "secret"}):
                results = await run(str(config))

        assert [r["provider"] for r in results] == ["groq"]


# =============================================================================
# Reporting and exit codes
# =============================================================================


class TestReport:
    """The report's return value becomes the process exit code."""

    def test_clean_run_returns_zero(self, capsys):
        results = [{"provider": "groq", "configured": 2, "available": 2, "missing": []}]

        assert report(results) == 0
        assert "ok groq" in capsys.readouterr().out

    def test_missing_model_returns_one_and_names_aliases(self, capsys):
        results = [
            {
                "provider": "groq",
                "configured": 2,
                "available": 1,
                "missing": [{"model": "llama-4-scout", "aliases": ["fast"]}],
            }
        ]

        assert report(results) == 1
        out = capsys.readouterr().out
        assert "llama-4-scout" in out
        assert "fast" in out

    def test_missing_model_without_alias(self, capsys):
        results = [
            {
                "provider": "groq",
                "configured": 1,
                "available": 0,
                "missing": [{"model": "orphan", "aliases": []}],
            }
        ]

        assert report(results) == 1
        assert "(no alias)" in capsys.readouterr().out

    def test_skipped_provider_does_not_fail_the_run(self, capsys):
        results = [{"provider": "groq", "skipped": "GROQ_API_KEY not set"}]

        assert report(results) == 0
        assert "skipped" in capsys.readouterr().out


class TestReportFallbacks:
    """Dangling fallback detection, reported alongside the model audit."""

    def _write(self, tmp_path, config):
        path = tmp_path / "config.yaml"
        path.write_text(yaml.safe_dump(config))
        return str(path)

    def test_clean_config(self, tmp_path):
        path = self._write(
            tmp_path,
            {
                "model_list": [{"model_name": "a", "litellm_params": {"model": "groq/a"}}],
                "router_settings": {"fallbacks": [{"a": []}]},
            },
        )

        assert report_fallbacks(path) is False

    def test_missing_target_is_reported(self, tmp_path, capsys):
        path = self._write(
            tmp_path,
            {
                "model_list": [{"model_name": "a", "litellm_params": {"model": "groq/a"}}],
                "router_settings": {"fallbacks": [{"a": ["ghost"]}]},
            },
        )

        assert report_fallbacks(path) is True
        assert "ghost" in capsys.readouterr().out

    def test_undefined_source_is_reported(self, tmp_path, capsys):
        path = self._write(
            tmp_path,
            {
                "model_list": [{"model_name": "a", "litellm_params": {"model": "groq/a"}}],
                "router_settings": {"fallbacks": [{"ghost": ["a"]}]},
            },
        )

        assert report_fallbacks(path) is True
        assert "ghost" in capsys.readouterr().out

    def test_unreadable_config_is_not_a_failure(self, tmp_path, capsys):
        assert report_fallbacks(str(tmp_path / "missing.yaml")) is False
        assert "skipped" in capsys.readouterr().out


class TestMain:
    """End-to-end CLI behaviour, including exit codes."""

    def test_missing_config_exits_two(self, tmp_path, capsys):
        with patch("sys.argv", ["verify_models", "--config", str(tmp_path / "nope.yaml")]):
            assert main() == 2
        assert "no config found" in capsys.readouterr().err

    def test_clean_config_exits_zero(self, tmp_path, capsys):
        config = tmp_path / "config.yaml"
        config.write_text(
            yaml.safe_dump(
                {
                    "model_list": [
                        {
                            "model_name": "fast",
                            "litellm_params": {"model": "groq/qwen/qwen3.6-27b"},
                        }
                    ],
                    "router_settings": {"fallbacks": [{"fast": []}]},
                }
            )
        )

        with respx.mock:
            respx.get(GROQ_URL).mock(return_value=_models("qwen/qwen3.6-27b"))
            with patch.dict(os.environ, {"GROQ_API_KEY": "secret"}), patch(
                "sys.argv", ["verify_models", "--config", str(config)]
            ):
                assert main() == 0

        assert "ok groq" in capsys.readouterr().out

    def test_missing_model_exits_one(self, tmp_path):
        config = tmp_path / "config.yaml"
        config.write_text(
            yaml.safe_dump(
                {
                    "model_list": [
                        {
                            "model_name": "llama-4-scout",
                            "litellm_params": {
                                "model": "groq/meta-llama/llama-4-scout-17b-16e-instruct"
                            },
                        }
                    ]
                }
            )
        )

        with respx.mock:
            respx.get(GROQ_URL).mock(return_value=_models("qwen/qwen3.6-27b"))
            with patch.dict(os.environ, {"GROQ_API_KEY": "secret"}), patch(
                "sys.argv", ["verify_models", "--config", str(config)]
            ):
                assert main() == 1

    def test_dangling_fallback_alone_exits_one(self, tmp_path):
        """A clean model audit must not mask a broken chain."""
        config = tmp_path / "config.yaml"
        config.write_text(
            yaml.safe_dump(
                {
                    "model_list": [
                        {
                            "model_name": "fast",
                            "litellm_params": {"model": "groq/qwen/qwen3.6-27b"},
                        }
                    ],
                    "router_settings": {"fallbacks": [{"fast": ["ghost"]}]},
                }
            )
        )

        with respx.mock:
            respx.get(GROQ_URL).mock(return_value=_models("qwen/qwen3.6-27b"))
            with patch.dict(os.environ, {"GROQ_API_KEY": "secret"}), patch(
                "sys.argv", ["verify_models", "--config", str(config)]
            ):
                assert main() == 1

    def test_json_output_is_machine_readable(self, tmp_path, capsys):
        import json

        config = tmp_path / "config.yaml"
        config.write_text(
            yaml.safe_dump(
                {
                    "model_list": [
                        {
                            "model_name": "fast",
                            "litellm_params": {"model": "groq/qwen/qwen3.6-27b"},
                        }
                    ]
                }
            )
        )

        with respx.mock:
            respx.get(GROQ_URL).mock(return_value=_models("qwen/qwen3.6-27b"))
            with patch.dict(os.environ, {"GROQ_API_KEY": "secret"}), patch(
                "sys.argv", ["verify_models", "--config", str(config), "--json"]
            ):
                assert main() == 0

        payload = json.loads(capsys.readouterr().out)
        assert payload["providers"][0]["provider"] == "groq"
        assert payload["dangling_fallbacks"] == []

    def test_shipped_config_has_no_dangling_fallbacks(self, capsys):
        """Guards the repo's own config without needing any provider key."""
        with patch.dict(os.environ, {k: "" for k in
                                     ("DEEPINFRA_API_KEY", "GROQ_API_KEY",
                                      "TOGETHER_API_KEY", "OPENROUTER_API_KEY")}), \
             respx.mock, patch("sys.argv", ["verify_models"]):
            respx.route(host__in=[
                "api.deepinfra.com", "openrouter.ai", "api.groq.com", "api.together.xyz",
            ]).mock(return_value=httpx.Response(401))

            assert main() == 0

        assert "not set" in capsys.readouterr().out
