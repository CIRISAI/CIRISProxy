"""
Tests for search_handler.py - Web search with billing integration.

Uses:
- pytest fixtures for setup
- respx for HTTP mocking
- hypothesis for property-based testing
"""

import os
from unittest.mock import patch, AsyncMock

import httpx
import pytest
import respx
from hypothesis import given, strategies as st, settings

from hooks.search_handler import (
    SearchHandler,
    handle_search_request,
    BRAVE_API_URL,
    EXA_API_URL,
    _get_active_provider,
)


# =============================================================================
# SearchHandler Tests
# =============================================================================


class TestSearchHandler:
    """Tests for SearchHandler class."""

    @pytest.fixture
    def handler(self):
        """Create a fresh handler for each test."""
        return SearchHandler()

    @pytest.fixture
    def env_vars(self):
        """Common environment variables."""
        return {
            "BRAVE_API_KEY": "brave_key_123",
            "BILLING_API_URL": "https://billing.test.com",
            "BILLING_API_KEY": "billing_key_456",
        }


class TestSearchHandlerInit:
    """Tests for SearchHandler initialization."""

    def test_creates_with_no_client(self):
        """Test handler starts with no HTTP client."""
        handler = SearchHandler()
        assert handler._client is None


class TestGenerateIdempotencyKey:
    """Tests for _generate_idempotency_key method."""

    def test_generates_unique_keys(self):
        """Test generates different keys for different queries."""
        handler = SearchHandler()
        key1 = handler._generate_idempotency_key("user1", "query1")
        key2 = handler._generate_idempotency_key("user1", "query2")
        assert key1 != key2

    def test_key_format(self):
        """Test key has expected format."""
        handler = SearchHandler()
        key = handler._generate_idempotency_key("user123456", "test query")
        assert key.startswith("search-user1234-")  # First 8 chars of user_id

    @given(
        user_id=st.text(min_size=1, max_size=100),
        query=st.text(min_size=1, max_size=500),
    )
    @settings(max_examples=50)
    def test_always_produces_string(self, user_id, query):
        """Property: always produces a non-empty string key."""
        handler = SearchHandler()
        key = handler._generate_idempotency_key(user_id, query)
        assert isinstance(key, str)
        assert len(key) > 0
        assert key.startswith("search-")


class TestChargeCredits:
    """Tests for _charge_credits method."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_successful_charge(self):
        """Test successful credit charge."""
        import hooks.search_handler as sh
        original_url = sh.BILLING_API_URL
        sh.BILLING_API_URL = "https://billing.test.com"

        try:
            respx.post("https://billing.test.com/v1/tools/charge").mock(
                return_value=httpx.Response(200, json={"charged": True})
            )

            handler = SearchHandler()
            success, error = await handler._charge_credits(
                oauth_provider="oauth:google",
                external_id="user123",
                idempotency_key="test-key",
            )

            assert success is True
            assert error is None
        finally:
            sh.BILLING_API_URL = original_url

    @pytest.mark.asyncio
    @respx.mock
    async def test_insufficient_credits(self):
        """Test returns error when no credits available."""
        import hooks.search_handler as sh
        original_url = sh.BILLING_API_URL
        sh.BILLING_API_URL = "https://billing.test.com"

        try:
            respx.post("https://billing.test.com/v1/tools/charge").mock(
                return_value=httpx.Response(
                    402,
                    json={"detail": "No web_search credits available"}
                )
            )

            handler = SearchHandler()
            success, error = await handler._charge_credits(
                oauth_provider="oauth:google",
                external_id="user123",
                idempotency_key="test-key",
            )

            assert success is False
            assert "No web_search credits" in error
        finally:
            sh.BILLING_API_URL = original_url

    @pytest.mark.asyncio
    @respx.mock
    async def test_billing_service_error(self):
        """Test handles billing service errors."""
        import hooks.search_handler as sh
        original_url = sh.BILLING_API_URL
        sh.BILLING_API_URL = "https://billing.test.com"

        try:
            respx.post("https://billing.test.com/v1/tools/charge").mock(
                return_value=httpx.Response(500)
            )

            handler = SearchHandler()
            success, error = await handler._charge_credits(
                oauth_provider="oauth:google",
                external_id="user123",
                idempotency_key="test-key",
            )

            assert success is False
            assert "Billing error: 500" in error
        finally:
            sh.BILLING_API_URL = original_url

    @pytest.mark.asyncio
    @respx.mock
    async def test_billing_service_unavailable(self):
        """Test handles network errors."""
        import hooks.search_handler as sh
        original_url = sh.BILLING_API_URL
        sh.BILLING_API_URL = "https://billing.test.com"

        try:
            respx.post("https://billing.test.com/v1/tools/charge").mock(
                side_effect=httpx.ConnectError("Connection refused")
            )

            handler = SearchHandler()
            success, error = await handler._charge_credits(
                oauth_provider="oauth:google",
                external_id="user123",
                idempotency_key="test-key",
            )

            assert success is False
            assert "unavailable" in error.lower()
        finally:
            sh.BILLING_API_URL = original_url


class TestCallBraveSearch:
    """Tests for _call_brave_search method."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_successful_search(self):
        """Test successful Brave search."""
        respx.get(BRAVE_API_URL).mock(
            return_value=httpx.Response(200, json={
                "web": {"results": [{"title": "Test Result"}]}
            })
        )

        with patch.dict(os.environ, {"BRAVE_API_KEY": "brave_key"}):
            # Need to reimport to pick up env var
            import hooks.search_handler as sh
            sh.BRAVE_API_KEY = "brave_key"

            handler = SearchHandler()
            results, error = await handler._call_brave_search("test query")

        assert results is not None
        assert error is None
        assert "web" in results

    @pytest.mark.asyncio
    async def test_missing_api_key(self):
        """Test returns error when Brave API key not configured."""
        import hooks.search_handler as sh
        original_key = sh.BRAVE_API_KEY
        sh.BRAVE_API_KEY = ""

        try:
            handler = SearchHandler()
            results, error = await handler._call_brave_search("test query")

            assert results is None
            assert "not configured" in error.lower()
        finally:
            sh.BRAVE_API_KEY = original_key

    @pytest.mark.asyncio
    @respx.mock
    async def test_search_error_response(self):
        """Test handles Brave API error responses."""
        respx.get(BRAVE_API_URL).mock(
            return_value=httpx.Response(429)
        )

        import hooks.search_handler as sh
        sh.BRAVE_API_KEY = "brave_key"

        handler = SearchHandler()
        results, error = await handler._call_brave_search("test query")

        assert results is None
        assert "429" in error

    @pytest.mark.asyncio
    @respx.mock
    async def test_limits_count_to_20(self):
        """Test limits count parameter to Brave's maximum of 20."""
        respx.get(BRAVE_API_URL).mock(
            return_value=httpx.Response(200, json={"results": []})
        )

        import hooks.search_handler as sh
        sh.BRAVE_API_KEY = "brave_key"

        handler = SearchHandler()
        await handler._call_brave_search("test", count=50)

        # Check the request was made with count=20
        assert respx.calls.last.request.url.params.get("count") == "20"


class TestHandleSearch:
    """Tests for handle_search method."""

    @pytest.mark.asyncio
    async def test_invalid_token(self):
        """Test returns 401 for invalid token."""
        with patch("hooks.search_handler.verify_google_token", new_callable=AsyncMock) as mock:
            mock.return_value = None

            handler = SearchHandler()
            result = await handler.handle_search(
                token="invalid_token",
                query="test query",
            )

        assert result["status_code"] == 401
        assert "Invalid" in result["error"]

    @pytest.mark.asyncio
    @respx.mock
    async def test_insufficient_credits_returns_402(self):
        """Test returns 402 when no credits available."""
        import hooks.search_handler as sh
        original_url = sh.BILLING_API_URL
        sh.BILLING_API_URL = "https://billing.test.com"

        try:
            respx.post("https://billing.test.com/v1/tools/charge").mock(
                return_value=httpx.Response(
                    402,
                    json={"detail": "No credits"}
                )
            )

            with patch("hooks.search_handler.verify_google_token", new_callable=AsyncMock) as mock:
                mock.return_value = {"sub": "user123"}

                handler = SearchHandler()
                result = await handler.handle_search(
                    token="valid_token",
                    query="test query",
                )

            assert result["status_code"] == 402
        finally:
            sh.BILLING_API_URL = original_url

    @pytest.mark.asyncio
    @respx.mock
    async def test_successful_search_flow(self):
        """Test complete successful search flow."""
        import hooks.search_handler as sh
        original_url = sh.BILLING_API_URL
        original_key = sh.BRAVE_API_KEY
        sh.BILLING_API_URL = "https://billing.test.com"
        sh.BRAVE_API_KEY = "brave_key"

        try:
            # Mock billing
            respx.post("https://billing.test.com/v1/tools/charge").mock(
                return_value=httpx.Response(200, json={"charged": True})
            )
            # Mock Brave
            respx.get(BRAVE_API_URL).mock(
                return_value=httpx.Response(200, json={"results": ["test"]})
            )

            with patch("hooks.search_handler.verify_google_token", new_callable=AsyncMock) as mock:
                mock.return_value = {"sub": "user123"}

                handler = SearchHandler()
                result = await handler.handle_search(
                    token="valid_token",
                    query="test query",
                )

            assert result["status_code"] == 200
            assert "results" in result
        finally:
            sh.BILLING_API_URL = original_url
            sh.BRAVE_API_KEY = original_key


class TestHandleSearchRequest:
    """Tests for handle_search_request function."""

    @pytest.mark.asyncio
    async def test_missing_query_returns_400(self):
        """Test returns 400 when query is missing."""
        result = await handle_search_request({}, "token")
        assert result["status_code"] == 400
        assert "Missing 'q'" in result["error"]

    @pytest.mark.asyncio
    async def test_empty_query_returns_400(self):
        """Test returns 400 when query is empty."""
        result = await handle_search_request({"q": ""}, "token")
        assert result["status_code"] == 400

    @pytest.mark.asyncio
    async def test_extracts_query_and_count(self):
        """Test extracts query and count from request."""
        with patch("hooks.search_handler.verify_google_token", new_callable=AsyncMock) as mock:
            mock.return_value = None  # Will fail auth but tests extraction

            result = await handle_search_request(
                {"q": "test query", "count": 5},
                "token"
            )

        assert result["status_code"] == 401  # Auth failed but params were extracted


class TestSearchHandlerClose:
    """Tests for SearchHandler.close method."""

    @pytest.mark.asyncio
    async def test_close_closes_client(self):
        """Test close properly closes HTTP client."""
        handler = SearchHandler()
        # Initialize client
        await handler._get_client()
        assert handler._client is not None

        await handler.close()
        assert handler._client is None

    @pytest.mark.asyncio
    async def test_close_handles_no_client(self):
        """Test close handles case where client was never created."""
        handler = SearchHandler()
        await handler.close()  # Should not raise


# =============================================================================
# Provider Selection Tests
# =============================================================================


class TestGetActiveProvider:
    """Tests for _get_active_provider function."""

    def test_returns_exa_when_exa_key_set_and_auto(self):
        """Test returns 'exa' when EXA_API_KEY is set in auto mode."""
        import hooks.search_handler as sh
        original_exa = sh.EXA_API_KEY
        original_brave = sh.BRAVE_API_KEY
        original_provider = sh.SEARCH_PROVIDER

        try:
            sh.EXA_API_KEY = "exa_key"
            sh.BRAVE_API_KEY = "brave_key"
            sh.SEARCH_PROVIDER = "auto"

            assert _get_active_provider() == "exa"
        finally:
            sh.EXA_API_KEY = original_exa
            sh.BRAVE_API_KEY = original_brave
            sh.SEARCH_PROVIDER = original_provider

    def test_returns_brave_when_only_brave_key_set(self):
        """Test returns 'brave' when only BRAVE_API_KEY is set."""
        import hooks.search_handler as sh
        original_exa = sh.EXA_API_KEY
        original_brave = sh.BRAVE_API_KEY
        original_provider = sh.SEARCH_PROVIDER

        try:
            sh.EXA_API_KEY = ""
            sh.BRAVE_API_KEY = "brave_key"
            sh.SEARCH_PROVIDER = "auto"

            assert _get_active_provider() == "brave"
        finally:
            sh.EXA_API_KEY = original_exa
            sh.BRAVE_API_KEY = original_brave
            sh.SEARCH_PROVIDER = original_provider

    def test_returns_none_when_no_keys(self):
        """Test returns 'none' when no API keys configured."""
        import hooks.search_handler as sh
        original_exa = sh.EXA_API_KEY
        original_brave = sh.BRAVE_API_KEY
        original_provider = sh.SEARCH_PROVIDER

        try:
            sh.EXA_API_KEY = ""
            sh.BRAVE_API_KEY = ""
            sh.SEARCH_PROVIDER = "auto"

            assert _get_active_provider() == "none"
        finally:
            sh.EXA_API_KEY = original_exa
            sh.BRAVE_API_KEY = original_brave
            sh.SEARCH_PROVIDER = original_provider

    def test_explicit_brave_selection(self):
        """Test explicit 'brave' provider selection."""
        import hooks.search_handler as sh
        original_exa = sh.EXA_API_KEY
        original_brave = sh.BRAVE_API_KEY
        original_provider = sh.SEARCH_PROVIDER

        try:
            sh.EXA_API_KEY = "exa_key"
            sh.BRAVE_API_KEY = "brave_key"
            sh.SEARCH_PROVIDER = "brave"

            assert _get_active_provider() == "brave"
        finally:
            sh.EXA_API_KEY = original_exa
            sh.BRAVE_API_KEY = original_brave
            sh.SEARCH_PROVIDER = original_provider

    def test_explicit_exa_selection(self):
        """Test explicit 'exa' provider selection."""
        import hooks.search_handler as sh
        original_exa = sh.EXA_API_KEY
        original_brave = sh.BRAVE_API_KEY
        original_provider = sh.SEARCH_PROVIDER

        try:
            sh.EXA_API_KEY = "exa_key"
            sh.BRAVE_API_KEY = "brave_key"
            sh.SEARCH_PROVIDER = "exa"

            assert _get_active_provider() == "exa"
        finally:
            sh.EXA_API_KEY = original_exa
            sh.BRAVE_API_KEY = original_brave
            sh.SEARCH_PROVIDER = original_provider


# =============================================================================
# Exa Search Tests
# =============================================================================


class TestCallExaSearch:
    """Tests for _call_exa_search method."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_successful_exa_search(self):
        """Test successful Exa search."""
        import hooks.search_handler as sh
        original_key = sh.EXA_API_KEY
        sh.EXA_API_KEY = "exa_test_key"

        try:
            respx.post(EXA_API_URL).mock(
                return_value=httpx.Response(200, json={
                    "requestId": "req-123",
                    "resolvedSearchType": "neural",
                    "results": [
                        {
                            "title": "Test Result",
                            "url": "https://example.com",
                            "text": "This is test content",
                            "publishedDate": "2025-01-01",
                            "author": "Test Author",
                            "highlights": ["highlight 1"],
                        }
                    ]
                })
            )

            handler = SearchHandler()
            results, error = await handler._call_exa_search("test query")

            assert results is not None
            assert error is None
            assert results["provider"] == "exa"
            assert results["web"]["results"][0]["title"] == "Test Result"
        finally:
            sh.EXA_API_KEY = original_key

    @pytest.mark.asyncio
    async def test_exa_missing_api_key(self):
        """Test returns error when Exa API key not configured."""
        import hooks.search_handler as sh
        original_key = sh.EXA_API_KEY
        sh.EXA_API_KEY = ""

        try:
            handler = SearchHandler()
            results, error = await handler._call_exa_search("test query")

            assert results is None
            assert "not configured" in error.lower()
        finally:
            sh.EXA_API_KEY = original_key

    @pytest.mark.asyncio
    @respx.mock
    async def test_exa_error_response(self):
        """Test handles Exa API error responses."""
        import hooks.search_handler as sh
        original_key = sh.EXA_API_KEY
        sh.EXA_API_KEY = "exa_test_key"

        try:
            respx.post(EXA_API_URL).mock(
                return_value=httpx.Response(429)
            )

            handler = SearchHandler()
            results, error = await handler._call_exa_search("test query")

            assert results is None
            assert "429" in error
        finally:
            sh.EXA_API_KEY = original_key

    @pytest.mark.asyncio
    @respx.mock
    async def test_exa_limits_count_to_100(self):
        """Test limits count parameter to Exa's maximum of 100."""
        import hooks.search_handler as sh
        original_key = sh.EXA_API_KEY
        sh.EXA_API_KEY = "exa_test_key"

        try:
            respx.post(EXA_API_URL).mock(
                return_value=httpx.Response(200, json={
                    "requestId": "req-123",
                    "results": []
                })
            )

            handler = SearchHandler()
            await handler._call_exa_search("test", count=150)

            # Check the request was made with numResults=100
            request_body = respx.calls.last.request.content
            import json
            body = json.loads(request_body)
            assert body["numResults"] == 100
        finally:
            sh.EXA_API_KEY = original_key

    @pytest.mark.asyncio
    @respx.mock
    async def test_exa_includes_content_request(self):
        """Test Exa request includes text and highlights content."""
        import hooks.search_handler as sh
        original_key = sh.EXA_API_KEY
        sh.EXA_API_KEY = "exa_test_key"

        try:
            respx.post(EXA_API_URL).mock(
                return_value=httpx.Response(200, json={
                    "requestId": "req-123",
                    "results": []
                })
            )

            handler = SearchHandler()
            await handler._call_exa_search("test")

            import json
            body = json.loads(respx.calls.last.request.content)
            assert "contents" in body
            assert "text" in body["contents"]
            assert "highlights" in body["contents"]
        finally:
            sh.EXA_API_KEY = original_key


class TestNormalizeExaResponse:
    """Tests for _normalize_exa_response method."""

    def test_normalizes_basic_response(self):
        """Test normalizes basic Exa response."""
        handler = SearchHandler()
        exa_response = {
            "requestId": "req-123",
            "resolvedSearchType": "neural",
            "results": [
                {
                    "title": "Test",
                    "url": "https://example.com",
                    "text": "Content text",
                }
            ]
        }

        normalized = handler._normalize_exa_response(exa_response)

        assert normalized["provider"] == "exa"
        assert normalized["request_id"] == "req-123"
        assert normalized["search_type"] == "neural"
        assert len(normalized["web"]["results"]) == 1
        assert normalized["web"]["results"][0]["title"] == "Test"

    def test_truncates_description(self):
        """Test truncates description to 500 chars."""
        handler = SearchHandler()
        long_text = "x" * 1000
        exa_response = {
            "results": [{"text": long_text, "title": "T", "url": "http://x"}]
        }

        normalized = handler._normalize_exa_response(exa_response)

        assert len(normalized["web"]["results"][0]["description"]) == 500

    def test_handles_empty_results(self):
        """Test handles empty results array."""
        handler = SearchHandler()
        exa_response = {"results": [], "requestId": "123"}

        normalized = handler._normalize_exa_response(exa_response)

        assert normalized["web"]["results"] == []
        assert normalized["provider"] == "exa"


# =============================================================================
# Execute Search with Fallback Tests
# =============================================================================


class TestExecuteSearch:
    """Tests for _execute_search method with provider fallback."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_fallback_to_brave_when_exa_fails(self):
        """Test falls back to Brave when Exa fails."""
        import hooks.search_handler as sh
        original_exa = sh.EXA_API_KEY
        original_brave = sh.BRAVE_API_KEY
        original_provider = sh.SEARCH_PROVIDER

        try:
            sh.EXA_API_KEY = "exa_key"
            sh.BRAVE_API_KEY = "brave_key"
            sh.SEARCH_PROVIDER = "auto"

            # Exa fails
            respx.post(EXA_API_URL).mock(
                return_value=httpx.Response(500)
            )
            # Brave succeeds
            respx.get(BRAVE_API_URL).mock(
                return_value=httpx.Response(200, json={
                    "web": {"results": [{"title": "Brave Result"}]}
                })
            )

            handler = SearchHandler()
            results, error = await handler._execute_search("test query")

            assert results is not None
            assert results["provider"] == "brave"
        finally:
            sh.EXA_API_KEY = original_exa
            sh.BRAVE_API_KEY = original_brave
            sh.SEARCH_PROVIDER = original_provider

    @pytest.mark.asyncio
    async def test_returns_error_when_no_provider(self):
        """Test returns error when no provider configured."""
        import hooks.search_handler as sh
        original_exa = sh.EXA_API_KEY
        original_brave = sh.BRAVE_API_KEY
        original_provider = sh.SEARCH_PROVIDER

        try:
            sh.EXA_API_KEY = ""
            sh.BRAVE_API_KEY = ""
            sh.SEARCH_PROVIDER = "auto"

            handler = SearchHandler()
            results, error = await handler._execute_search("test query")

            assert results is None
            assert "No search provider configured" in error
        finally:
            sh.EXA_API_KEY = original_exa
            sh.BRAVE_API_KEY = original_brave
            sh.SEARCH_PROVIDER = original_provider
