"""
Tests for server.py - Custom FastAPI endpoints.

Uses FastAPI TestClient pattern from LiteLLM testing approach.
"""

import sys
import os

# Add hooks directory to path so server.py can find its imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "hooks"))

import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from fastapi import FastAPI
from fastapi.testclient import TestClient

# Import the function that adds routes
from server import add_custom_routes


@pytest.fixture
def app():
    """Create a test FastAPI app with custom routes."""
    test_app = FastAPI()
    add_custom_routes(test_app)
    return test_app


@pytest.fixture
def client(app):
    """Create a test client."""
    return TestClient(app)


# =============================================================================
# /v1/status endpoint tests
# =============================================================================


class TestStatusEndpoint:
    """Tests for /v1/status endpoint."""

    def test_returns_status_on_success(self, client):
        """Test returns provider status on success."""
        mock_status = {
            "service": "cirisproxy",
            "status": "operational",
            "providers": [
                {"provider": "openrouter", "status": "operational"},
            ],
        }

        with patch("server.get_status", new_callable=AsyncMock) as mock:
            mock.return_value = mock_status

            response = client.get("/v1/status")

        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "cirisproxy"
        assert data["status"] == "operational"

    def test_returns_500_on_error(self, client):
        """Test returns 500 with safe error message on exception."""
        with patch("server.get_status", new_callable=AsyncMock) as mock:
            mock.side_effect = Exception("Database connection failed with password=secret")

            response = client.get("/v1/status")

        assert response.status_code == 500
        data = response.json()
        assert data["status"] == "outage"
        assert data["error"] == "Internal error"
        # Should not leak sensitive info
        assert "password" not in str(data)
        assert "secret" not in str(data)


class TestStatusSimpleEndpoint:
    """Tests for /v1/status/simple endpoint."""

    def test_returns_ok(self, client):
        """Test returns simple OK status."""
        response = client.get("/v1/status/simple")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["service"] == "cirisproxy"


# =============================================================================
# /v1/web/search endpoint tests
# =============================================================================


class TestSearchEndpoint:
    """Tests for /v1/web/search endpoint."""

    def test_returns_401_without_auth(self, client):
        """Test returns 401 when Authorization header missing."""
        response = client.post("/v1/web/search", json={"q": "test"})

        assert response.status_code == 401
        data = response.json()
        assert "Authorization" in data["error"]

    def test_returns_401_with_invalid_auth_format(self, client):
        """Test returns 401 when Authorization header has wrong format."""
        response = client.post(
            "/v1/web/search",
            json={"q": "test"},
            headers={"Authorization": "Basic abc123"},
        )

        assert response.status_code == 401

    def test_returns_400_on_invalid_json(self, client):
        """Test returns 400 on invalid JSON body."""
        response = client.post(
            "/v1/web/search",
            content="not valid json",
            headers={
                "Authorization": "Bearer token123",
                "Content-Type": "application/json",
            },
        )

        assert response.status_code == 400
        data = response.json()
        assert "Invalid JSON" in data["error"]

    def test_successful_search(self, client):
        """Test successful search flow."""
        mock_result = {
            "results": {"web": {"results": [{"title": "Test"}]}},
            "status_code": 200,
        }

        with patch("server.handle_search_request", new_callable=AsyncMock) as mock:
            mock.return_value = mock_result

            response = client.post(
                "/v1/web/search",
                json={"q": "test query", "count": 5},
                headers={"Authorization": "Bearer valid-token"},
            )

        assert response.status_code == 200
        data = response.json()
        assert "results" in data

        # Verify handler was called with correct args
        mock.assert_called_once()
        call_args = mock.call_args
        assert call_args[0][0] == {"q": "test query", "count": 5}
        assert call_args[0][1] == "valid-token"

    def test_returns_402_on_insufficient_credits(self, client):
        """Test returns 402 when user has no credits."""
        mock_result = {
            "error": "No web_search credits available",
            "status_code": 402,
        }

        with patch("server.handle_search_request", new_callable=AsyncMock) as mock:
            mock.return_value = mock_result

            response = client.post(
                "/v1/web/search",
                json={"q": "test"},
                headers={"Authorization": "Bearer token"},
            )

        assert response.status_code == 402
        data = response.json()
        assert "credits" in data["error"].lower()

    def test_strips_bearer_prefix(self, client):
        """Test correctly strips Bearer prefix from token."""
        mock_result = {"results": {}, "status_code": 200}

        with patch("server.handle_search_request", new_callable=AsyncMock) as mock:
            mock.return_value = mock_result

            client.post(
                "/v1/web/search",
                json={"q": "test"},
                headers={"Authorization": "Bearer my-google-token-123"},
            )

        # Token should have Bearer stripped
        call_args = mock.call_args
        assert call_args[0][1] == "my-google-token-123"


# =============================================================================
# Integration-style tests
# =============================================================================


class TestEndpointIntegration:
    """Integration tests for endpoint behavior."""

    def test_status_simple_always_works(self, client):
        """Test /v1/status/simple works even if providers are down."""
        # This endpoint should always return OK as long as the server is running
        response = client.get("/v1/status/simple")
        assert response.status_code == 200

    def test_endpoints_return_json(self, client):
        """Test all endpoints return JSON content type."""
        with patch("server.get_status", new_callable=AsyncMock) as mock:
            mock.return_value = {"service": "cirisproxy", "status": "operational"}

            response = client.get("/v1/status")
            assert response.headers["content-type"] == "application/json"

        response = client.get("/v1/status/simple")
        assert response.headers["content-type"] == "application/json"

        response = client.post("/v1/web/search", json={"q": "test"})
        assert response.headers["content-type"] == "application/json"
