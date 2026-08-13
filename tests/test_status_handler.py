"""
Tests for status_handler.py - Provider health checks.

Uses:
- pytest parametrize for data variations
- hypothesis for property-based testing
- respx for HTTP mocking
"""

import os
import time
from unittest.mock import patch

import httpx
import pytest
import respx
from hypothesis import given, settings
from hypothesis import strategies as st

from hooks.status_handler import (
    POOLS,
    PROVIDERS,
    ROLE_FALLBACK,
    ROLE_PRIMARY,
    STATUS_DEGRADED,
    STATUS_OPERATIONAL,
    STATUS_OUTAGE,
    STATUS_UNKNOWN,
    _build_auth_headers,
    _build_check_url,
    _evaluate_response,
    _get_api_key,
    _init_provider_result,
    _summarize_pool,
    _worst,
    check_provider,
    get_status,
)

# =============================================================================
# Helper Function Tests
# =============================================================================


class TestInitProviderResult:
    """Tests for _init_provider_result helper."""

    def test_creates_result_with_defaults(self):
        """Test basic result creation."""
        config = {"name": "TestProvider", "type": "llm"}
        result = _init_provider_result("test", config)

        assert result["provider"] == "test"
        assert result["name"] == "TestProvider"
        assert result["type"] == "llm"
        assert result["status"] == STATUS_OUTAGE
        assert result["latency_ms"] is None
        assert result["error"] is None
        assert "checked_at" in result

    @given(
        provider_id=st.text(min_size=1, max_size=50),
        name=st.text(min_size=1, max_size=100),
        ptype=st.sampled_from(["llm", "internal", "search"]),
    )
    @settings(max_examples=50)
    def test_handles_various_inputs(self, provider_id, name, ptype):
        """Property: result always contains required fields."""
        config = {"name": name, "type": ptype}
        result = _init_provider_result(provider_id, config)

        assert result["provider"] == provider_id
        assert result["name"] == name
        assert result["type"] == ptype
        assert "status" in result
        assert "checked_at" in result


class TestGetApiKey:
    """Tests for _get_api_key helper."""

    def test_returns_key_when_set(self):
        """Test key retrieval when environment variable is set."""
        with patch.dict(os.environ, {"TEST_API_KEY": "secret123"}):
            config = {"env_key": "TEST_API_KEY"}
            assert _get_api_key(config) == "secret123"

    def test_returns_none_when_not_set(self):
        """Test returns None when env var is missing."""
        config = {"env_key": "NONEXISTENT_KEY_12345"}
        assert _get_api_key(config) is None

    def test_returns_none_for_empty_string(self):
        """Test returns None when env var is empty string."""
        with patch.dict(os.environ, {"EMPTY_KEY": ""}):
            config = {"env_key": "EMPTY_KEY"}
            assert _get_api_key(config) is None


class TestBuildCheckUrl:
    """Tests for _build_check_url helper."""

    def test_returns_static_url(self):
        """Test returns check_url when no env_url."""
        config = {"check_url": "https://api.example.com/health"}
        assert _build_check_url(config) == "https://api.example.com/health"

    def test_builds_url_from_env(self):
        """Test builds URL from environment variable."""
        with patch.dict(os.environ, {"API_URL": "https://api.example.com"}):
            config = {"env_url": "API_URL", "check_path": "/health"}
            assert _build_check_url(config) == "https://api.example.com/health"

    def test_strips_trailing_slash(self):
        """Test strips trailing slash from base URL."""
        with patch.dict(os.environ, {"API_URL": "https://api.example.com/"}):
            config = {"env_url": "API_URL", "check_path": "/health"}
            assert _build_check_url(config) == "https://api.example.com/health"

    def test_returns_none_when_env_not_set(self):
        """Test returns None when env URL not configured."""
        config = {"env_url": "NONEXISTENT_URL", "check_path": "/health"}
        assert _build_check_url(config) is None

    def test_default_check_path(self):
        """Test uses /health as default check path."""
        with patch.dict(os.environ, {"API_URL": "https://api.example.com"}):
            config = {"env_url": "API_URL"}  # No check_path
            assert _build_check_url(config) == "https://api.example.com/health"


class TestBuildAuthHeaders:
    """Tests for _build_auth_headers helper."""

    def test_builds_bearer_header(self):
        """Test builds Authorization: Bearer header."""
        config = {"auth_header": "Authorization", "auth_prefix": "Bearer "}
        headers = _build_auth_headers(config, "token123")
        assert headers == {"Authorization": "Bearer token123"}

    def test_builds_custom_header(self):
        """Test builds custom header without prefix."""
        config = {"auth_header": "X-Subscription-Token", "auth_prefix": ""}
        headers = _build_auth_headers(config, "token123")
        assert headers == {"X-Subscription-Token": "token123"}

    def test_returns_empty_when_no_auth_header(self):
        """Test returns empty dict when no auth_header configured."""
        config = {}
        headers = _build_auth_headers(config, "token123")
        assert headers == {}


class TestEvaluateResponse:
    """Tests for _evaluate_response helper."""

    @pytest.mark.parametrize(
        "latency_ms,status_code,expected_status,expected_error",
        [
            # Good latency, success
            (100, 200, STATUS_OPERATIONAL, None),
            (500, 200, STATUS_OPERATIONAL, None),
            (999, 200, STATUS_OPERATIONAL, None),
            # Degraded latency
            (1000, 200, STATUS_DEGRADED, None),
            (2000, 200, STATUS_DEGRADED, None),
            (2999, 200, STATUS_DEGRADED, None),
            # High latency (still degraded but with error)
            (3000, 200, STATUS_DEGRADED, "High latency: 3000ms"),
            (5000, 200, STATUS_DEGRADED, "High latency: 5000ms"),
            # Error status codes
            (100, 400, STATUS_OUTAGE, "HTTP 400"),
            (100, 401, STATUS_OUTAGE, "HTTP 401"),
            (100, 500, STATUS_OUTAGE, "HTTP 500"),
            (100, 503, STATUS_OUTAGE, "HTTP 503"),
        ],
    )
    def test_status_evaluation(self, latency_ms, status_code, expected_status, expected_error):
        """Test status evaluation for various latency/status code combinations."""
        result = {"status": None, "latency_ms": None, "error": None}
        _evaluate_response(result, latency_ms, status_code)

        assert result["status"] == expected_status
        assert result["latency_ms"] == latency_ms
        assert result["error"] == expected_error

    @given(latency_ms=st.integers(min_value=0, max_value=60000))
    @settings(max_examples=100)
    def test_latency_always_set(self, latency_ms):
        """Property: latency is always set regardless of status code."""
        result = {"status": None, "latency_ms": None, "error": None}
        _evaluate_response(result, latency_ms, 200)
        assert result["latency_ms"] == latency_ms


# =============================================================================
# check_provider Tests
# =============================================================================


class TestCheckProvider:
    """Tests for check_provider async function."""

    @pytest.fixture
    def mock_config(self):
        """Sample provider config."""
        return {
            "name": "TestProvider",
            "type": "llm",
            "check_url": "https://api.test.com/health",
            "env_key": "TEST_API_KEY",
            "auth_header": "Authorization",
            "auth_prefix": "Bearer ",
        }

    @pytest.mark.asyncio
    async def test_returns_error_when_no_api_key(self, mock_config):
        """Test returns error when API key not configured."""
        async with httpx.AsyncClient() as client:
            result = await check_provider(client, "test", mock_config)

        assert result["status"] == STATUS_OUTAGE
        assert result["error"] == "API key not configured"

    @pytest.mark.asyncio
    @respx.mock
    async def test_successful_health_check(self, mock_config):
        """Test successful health check returns operational status."""
        respx.get("https://api.test.com/health").mock(
            return_value=httpx.Response(200)
        )

        with patch.dict(os.environ, {"TEST_API_KEY": "secret"}):
            async with httpx.AsyncClient() as client:
                result = await check_provider(client, "test", mock_config)

        assert result["status"] == STATUS_OPERATIONAL
        assert result["error"] is None
        assert result["latency_ms"] is not None

    @pytest.mark.asyncio
    @respx.mock
    async def test_handles_http_error(self, mock_config):
        """Test handles HTTP error responses."""
        respx.get("https://api.test.com/health").mock(
            return_value=httpx.Response(500)
        )

        with patch.dict(os.environ, {"TEST_API_KEY": "secret"}):
            async with httpx.AsyncClient() as client:
                result = await check_provider(client, "test", mock_config)

        assert result["status"] == STATUS_OUTAGE
        assert result["error"] == "HTTP 500"

    @pytest.mark.asyncio
    @respx.mock
    async def test_handles_timeout(self, mock_config):
        """Test handles timeout exceptions."""
        respx.get("https://api.test.com/health").mock(
            side_effect=httpx.TimeoutException("Timeout")
        )

        with patch.dict(os.environ, {"TEST_API_KEY": "secret"}):
            async with httpx.AsyncClient() as client:
                result = await check_provider(client, "test", mock_config)

        assert result["status"] == STATUS_OUTAGE
        assert result["error"] == "Timeout"

    @pytest.mark.asyncio
    @respx.mock
    async def test_handles_connection_error(self, mock_config):
        """Test handles connection errors."""
        respx.get("https://api.test.com/health").mock(
            side_effect=httpx.ConnectError("Connection refused")
        )

        with patch.dict(os.environ, {"TEST_API_KEY": "secret"}):
            async with httpx.AsyncClient() as client:
                result = await check_provider(client, "test", mock_config)

        assert result["status"] == STATUS_OUTAGE
        assert "Connection error" in result["error"]

    @pytest.mark.asyncio
    async def test_dynamic_url_config(self):
        """Test provider with dynamic URL from environment."""
        config = {
            "name": "Billing",
            "type": "internal",
            "check_url": None,
            "env_key": "BILLING_KEY",
            "env_url": "BILLING_URL",
            "check_path": "/health",
        }

        # Missing URL should return error
        with patch.dict(os.environ, {"BILLING_KEY": "secret"}):
            async with httpx.AsyncClient() as client:
                result = await check_provider(client, "billing", config)

        assert result["error"] == "URL not configured"


# =============================================================================
# get_status Tests
# =============================================================================


class TestGetStatus:
    """Tests for get_status function."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_returns_aggregated_status(self):
        """Test returns status for all providers."""
        # Mock all provider endpoints
        for provider_id, config in PROVIDERS.items():
            if config.get("check_url"):
                respx.get(config["check_url"]).mock(
                    return_value=httpx.Response(200)
                )
            elif config.get("env_url"):
                # Skip dynamic URLs in this test
                pass

        # Set all API keys
        env_vars = {
            "OPENROUTER_API_KEY": "key1",
            "GROQ_API_KEY": "key2",
            "TOGETHER_API_KEY": "key3",
            "BILLING_API_KEY": "key4",
            "BILLING_API_URL": "https://billing.test.com",
            "BRAVE_API_KEY": "key5",
        }
        respx.get("https://billing.test.com/health").mock(
            return_value=httpx.Response(200)
        )

        with patch.dict(os.environ, env_vars):
            # Clear cache to force fresh check
            import hooks.status_handler as sh
            sh._status_cache = {}
            sh._cache_timestamp = 0

            status = await get_status()

        assert status["service"] == "cirisproxy"
        assert "status" in status
        assert "providers" in status
        assert len(status["providers"]) == len(PROVIDERS)

    @pytest.mark.asyncio
    async def test_cache_prevents_repeated_checks(self):
        """Test that cache prevents hammering upstream providers."""
        import hooks.status_handler as sh

        # Pre-populate cache
        sh._status_cache = {
            "service": "cirisproxy",
            "status": STATUS_OPERATIONAL,
            "providers": [],
        }
        sh._cache_timestamp = time.monotonic()  # Fresh cache

        # Should return cached result without making requests
        result = await get_status()

        assert result == sh._status_cache


# =============================================================================
# Model Audit Integration Tests
# =============================================================================


class TestModelAuditIntegration:
    """The health probe's /models body is reused to audit configured models."""

    @pytest.fixture
    def groq_config(self):
        return PROVIDERS["groq"]

    @staticmethod
    def _models_response(*ids):
        return httpx.Response(200, json={"data": [{"id": i} for i in ids]})

    @pytest.mark.asyncio
    @respx.mock
    async def test_missing_model_degrades_provider(self):
        """Reachable but no longer serving what we route there — the Groq case."""
        respx.get(PROVIDERS["groq"]["check_url"]).mock(
            return_value=self._models_response("qwen/qwen3.6-27b")
        )

        with patch.dict(os.environ, {"GROQ_API_KEY": "secret"}), patch(
            "hooks.status_handler.load_configured_models",
            return_value={
                "groq": {
                    "qwen/qwen3.6-27b": ["fast"],
                    "meta-llama/llama-4-scout-17b-16e-instruct": ["llama-4-scout"],
                }
            },
        ):
            async with httpx.AsyncClient() as client:
                result = await check_provider(client, "groq", PROVIDERS["groq"])

        assert result["status"] == STATUS_DEGRADED
        assert result["models"]["available"] == 1
        assert result["models"]["missing"][0]["aliases"] == ["llama-4-scout"]
        assert "llama-4-scout-17b-16e-instruct" in result["error"]

    @pytest.mark.asyncio
    @respx.mock
    async def test_all_models_present_stays_operational(self):
        respx.get(PROVIDERS["groq"]["check_url"]).mock(
            return_value=self._models_response("qwen/qwen3.6-27b")
        )

        with patch.dict(os.environ, {"GROQ_API_KEY": "secret"}), patch(
            "hooks.status_handler.load_configured_models",
            return_value={"groq": {"qwen/qwen3.6-27b": ["fast"]}},
        ):
            async with httpx.AsyncClient() as client:
                result = await check_provider(client, "groq", PROVIDERS["groq"])

        assert result["status"] == STATUS_OPERATIONAL
        assert result["models"] == {"configured": 1, "available": 1, "missing": []}
        assert result["error"] is None

    @pytest.mark.asyncio
    @respx.mock
    async def test_audit_never_upgrades_a_verdict(self):
        """A slow provider stays degraded-for-latency even with all models present."""
        respx.get(PROVIDERS["groq"]["check_url"]).mock(
            return_value=self._models_response("qwen/qwen3.6-27b")
        )

        with patch.dict(os.environ, {"GROQ_API_KEY": "secret"}), patch(
            "hooks.status_handler.load_configured_models",
            return_value={"groq": {"qwen/qwen3.6-27b": ["fast"]}},
        ), patch("hooks.status_handler._evaluate_response") as mock_eval:

            def slow(result, latency_ms, status_code):
                result["status"] = STATUS_DEGRADED
                result["error"] = "High latency: 4000ms"

            mock_eval.side_effect = slow

            async with httpx.AsyncClient() as client:
                result = await check_provider(client, "groq", PROVIDERS["groq"])

        assert result["status"] == STATUS_DEGRADED
        assert result["error"] == "High latency: 4000ms"

    @pytest.mark.asyncio
    @respx.mock
    async def test_unparseable_body_is_not_a_failure(self):
        """We must not invent an outage from a body we could not read."""
        respx.get(PROVIDERS["groq"]["check_url"]).mock(
            return_value=httpx.Response(200, text="not json")
        )

        with patch.dict(os.environ, {"GROQ_API_KEY": "secret"}), patch(
            "hooks.status_handler.load_configured_models",
            return_value={"groq": {"qwen/qwen3.6-27b": ["fast"]}},
        ):
            async with httpx.AsyncClient() as client:
                result = await check_provider(client, "groq", PROVIDERS["groq"])

        assert result["status"] == STATUS_OPERATIONAL
        assert "models" not in result

    @pytest.mark.asyncio
    @respx.mock
    async def test_billing_is_not_model_audited(self):
        """Non-LLM dependencies have no model list to audit."""
        respx.get("https://billing.test.com/health").mock(
            return_value=httpx.Response(200, json={"data": [{"id": "x"}]})
        )

        env = {"BILLING_API_KEY": "secret", "BILLING_API_URL": "https://billing.test.com"}
        with patch.dict(os.environ, env):
            async with httpx.AsyncClient() as client:
                result = await check_provider(client, "billing", PROVIDERS["billing"])

        assert result["status"] == STATUS_OPERATIONAL
        assert "models" not in result


# =============================================================================
# Provider Registry Tests
# =============================================================================


class TestProviderRegistry:
    """The monitored set must match the serving set (issue #7)."""

    def test_primary_provider_is_monitored(self):
        """DeepInfra serves the default chain, so it must be checked."""
        assert "deepinfra" in PROVIDERS
        assert PROVIDERS["deepinfra"]["role"] == ROLE_PRIMARY

    def test_exactly_one_primary_per_pool(self):
        """A pool with two primaries (or none) makes `primary_available` meaningless."""
        for pool_id in POOLS:
            primaries = [
                p for p in PROVIDERS.values()
                if p["type"] == pool_id and p.get("role") == ROLE_PRIMARY
            ]
            assert len(primaries) == 1, f"pool {pool_id} has {len(primaries)} primaries"

    def test_pooled_providers_declare_a_role(self):
        """Every pool member carries primary/fallback; non-pooled deps do not."""
        for provider_id, config in PROVIDERS.items():
            if config["type"] in POOLS:
                assert config.get("role") in (ROLE_PRIMARY, ROLE_FALLBACK), provider_id
            else:
                assert config.get("role") is None, provider_id

    def test_no_metered_search_provider(self):
        """Regression guard for PR #6 — search APIs bill per probe."""
        assert not any(c["type"] == "search" for c in PROVIDERS.values())


# =============================================================================
# Rollup Tests
# =============================================================================


def _member(provider_id, status, role=ROLE_FALLBACK, ptype="llm"):
    """Build a provider result for rollup tests."""
    return {"provider": provider_id, "type": ptype, "role": role, "status": status}


class TestWorst:
    """Tests for _worst severity ordering."""

    @pytest.mark.parametrize(
        "statuses,expected",
        [
            ([], STATUS_OPERATIONAL),
            ([STATUS_OPERATIONAL], STATUS_OPERATIONAL),
            ([STATUS_OPERATIONAL, STATUS_DEGRADED], STATUS_DEGRADED),
            ([STATUS_DEGRADED, STATUS_OUTAGE], STATUS_OUTAGE),
            ([STATUS_OUTAGE, STATUS_OPERATIONAL], STATUS_OUTAGE),
            ([STATUS_UNKNOWN, STATUS_OPERATIONAL], STATUS_UNKNOWN),
            ([STATUS_UNKNOWN, STATUS_OUTAGE], STATUS_OUTAGE),
        ],
    )
    def test_severity_ordering(self, statuses, expected):
        assert _worst(statuses) == expected


class TestSummarizePool:
    """Tests for _summarize_pool — CIRISStatus FSD §2.2 semantics."""

    @pytest.fixture
    def config(self):
        return {"label": "LLM providers", "min_available": 1}

    def test_all_healthy(self, config):
        results = [
            _member("deepinfra", STATUS_OPERATIONAL, ROLE_PRIMARY),
            _member("groq", STATUS_OPERATIONAL),
        ]
        pool = _summarize_pool("llm", config, results)

        assert pool["status"] == STATUS_OPERATIONAL
        assert pool["available"] == 2
        assert pool["primary_available"] is True
        assert pool["members"] == ["deepinfra", "groq"]

    def test_one_member_down_is_still_operational(self, config):
        """The whole point of issue #7: a dead fallback is routed around."""
        results = [
            _member("deepinfra", STATUS_OPERATIONAL, ROLE_PRIMARY),
            _member("together", STATUS_OUTAGE),
        ]
        pool = _summarize_pool("llm", config, results)

        assert pool["status"] == STATUS_OPERATIONAL
        assert pool["available"] == 1

    def test_degraded_member_counts_as_unavailable(self, config):
        """A degraded member is routed around, so it does not count toward the threshold."""
        results = [
            _member("deepinfra", STATUS_DEGRADED, ROLE_PRIMARY),
            _member("groq", STATUS_OPERATIONAL),
        ]
        pool = _summarize_pool("llm", config, results)

        assert pool["available"] == 1
        assert pool["status"] == STATUS_OPERATIONAL
        assert pool["primary_available"] is False

    def test_serving_on_fallback_is_not_impairment(self, config):
        """Primary down, fallback up: operational, but the fact is recorded."""
        results = [
            _member("deepinfra", STATUS_OUTAGE, ROLE_PRIMARY),
            _member("openrouter", STATUS_OPERATIONAL),
        ]
        pool = _summarize_pool("llm", config, results)

        assert pool["status"] == STATUS_OPERATIONAL
        assert pool["primary_available"] is False

    def test_all_members_down_is_outage(self, config):
        results = [
            _member("deepinfra", STATUS_OUTAGE, ROLE_PRIMARY),
            _member("groq", STATUS_DEGRADED),
        ]
        pool = _summarize_pool("llm", config, results)

        assert pool["status"] == STATUS_OUTAGE
        assert pool["available"] == 0

    def test_quorum_below_threshold_is_degraded(self):
        """min_available > 1: still serving, but the margin is gone."""
        config = {"label": "LLM providers", "min_available": 2}
        results = [
            _member("deepinfra", STATUS_OPERATIONAL, ROLE_PRIMARY),
            _member("groq", STATUS_OUTAGE),
        ]
        pool = _summarize_pool("llm", config, results)

        assert pool["status"] == STATUS_DEGRADED
        assert pool["available"] == 1

    def test_empty_pool_is_unknown(self, config):
        """Never green by omission."""
        pool = _summarize_pool("llm", config, [_member("billing", STATUS_OPERATIONAL, None, "internal")])

        assert pool["status"] == STATUS_UNKNOWN
        assert pool["members"] == []
        assert pool["primary_available"] is None

    def test_ignores_other_pools_members(self, config):
        results = [
            _member("deepinfra", STATUS_OPERATIONAL, ROLE_PRIMARY),
            _member("billing", STATUS_OUTAGE, None, "internal"),
        ]
        pool = _summarize_pool("llm", config, results)

        assert pool["members"] == ["deepinfra"]
        assert pool["status"] == STATUS_OPERATIONAL


class TestServiceStatusRollup:
    """get_status must not inherit pooled member health (issue #7 D1)."""

    @staticmethod
    async def _status_with(statuses):
        """Run get_status with each provider forced to a given status."""
        import hooks.status_handler as sh

        async def fake_check(client, provider_id, config):
            result = _init_provider_result(provider_id, config)
            result["status"] = statuses[provider_id]
            return result

        sh._status_cache = {}
        sh._cache_timestamp = 0
        with patch.object(sh, "check_provider", fake_check):
            return await sh.get_status()

    @pytest.mark.asyncio
    async def test_healthy_fabric_is_operational(self):
        status = await self._status_with(
            {p: STATUS_OPERATIONAL for p in PROVIDERS}
        )

        assert status["status"] == STATUS_OPERATIONAL
        assert status["pools"]["llm"]["status"] == STATUS_OPERATIONAL
        assert status["pools"]["llm"]["available"] == 4

    @pytest.mark.asyncio
    async def test_single_llm_outage_does_not_degrade_service(self):
        """Four days of amber for one unused provider — the bug this fixes."""
        statuses = {p: STATUS_OPERATIONAL for p in PROVIDERS}
        statuses["together"] = STATUS_OUTAGE

        status = await self._status_with(statuses)

        assert status["status"] == STATUS_OPERATIONAL
        assert status["pools"]["llm"]["status"] == STATUS_OPERATIONAL
        # The member's own verdict is preserved, not overwritten
        together = next(p for p in status["providers"] if p["provider"] == "together")
        assert together["status"] == STATUS_OUTAGE

    @pytest.mark.asyncio
    async def test_slow_llm_provider_does_not_degrade_service(self):
        statuses = {p: STATUS_OPERATIONAL for p in PROVIDERS}
        statuses["openrouter"] = STATUS_DEGRADED

        status = await self._status_with(statuses)

        assert status["status"] == STATUS_OPERATIONAL

    @pytest.mark.asyncio
    async def test_primary_outage_is_visible_but_not_impairment(self):
        statuses = {p: STATUS_OPERATIONAL for p in PROVIDERS}
        statuses["deepinfra"] = STATUS_OUTAGE

        status = await self._status_with(statuses)

        assert status["status"] == STATUS_OPERATIONAL
        assert status["pools"]["llm"]["primary_available"] is False

    @pytest.mark.asyncio
    async def test_entire_pool_down_is_an_outage(self):
        """Nothing left to route to is real impairment."""
        statuses = {p: STATUS_OUTAGE for p in PROVIDERS if p != "billing"}
        statuses["billing"] = STATUS_OPERATIONAL

        status = await self._status_with(statuses)

        assert status["status"] == STATUS_OUTAGE
        assert status["pools"]["llm"]["status"] == STATUS_OUTAGE

    @pytest.mark.asyncio
    async def test_billing_outage_degrades_service(self):
        """Non-pooled dependency: nothing else serves it."""
        statuses = {p: STATUS_OPERATIONAL for p in PROVIDERS}
        statuses["billing"] = STATUS_OUTAGE

        status = await self._status_with(statuses)

        assert status["status"] == STATUS_OUTAGE

    @pytest.mark.asyncio
    async def test_billing_degraded_degrades_service(self):
        statuses = {p: STATUS_OPERATIONAL for p in PROVIDERS}
        statuses["billing"] = STATUS_DEGRADED

        status = await self._status_with(statuses)

        assert status["status"] == STATUS_DEGRADED

    @given(
        llm_statuses=st.lists(
            st.sampled_from([STATUS_OPERATIONAL, STATUS_DEGRADED, STATUS_OUTAGE]),
            min_size=4,
            max_size=4,
        )
    )
    @settings(max_examples=50, deadline=None)
    def test_property_healthy_billing_and_one_llm_is_operational(self, llm_statuses):
        """Property: any pool state with >=1 operational member leaves us operational."""
        import asyncio

        pool_ids = [p for p in PROVIDERS if p != "billing"]
        statuses = dict(zip(pool_ids, llm_statuses))
        statuses["billing"] = STATUS_OPERATIONAL

        status = asyncio.run(self._status_with(statuses))

        if STATUS_OPERATIONAL in llm_statuses:
            assert status["status"] == STATUS_OPERATIONAL
        else:
            assert status["status"] == STATUS_OUTAGE
