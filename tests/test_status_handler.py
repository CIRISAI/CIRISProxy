"""
Tests for status_handler.py - Provider health checks.

Uses:
- pytest parametrize for data variations
- hypothesis for property-based testing
- respx for HTTP mocking
"""

import os
import time
from unittest.mock import patch, MagicMock

import httpx
import pytest
import respx
from hypothesis import given, strategies as st, settings

from hooks.status_handler import (
    _init_provider_result,
    _get_api_key,
    _build_check_url,
    _build_auth_headers,
    _evaluate_response,
    check_provider,
    get_status,
    STATUS_OPERATIONAL,
    STATUS_DEGRADED,
    STATUS_OUTAGE,
    LATENCY_GOOD,
    LATENCY_DEGRADED,
    PROVIDERS,
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
