"""
Pytest fixtures for CIRISProxy tests.
"""

import os
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Set test environment variables before importing hooks
os.environ.setdefault("BILLING_API_URL", "https://billing.test.ciris.ai")
os.environ.setdefault("BILLING_API_KEY", "test-billing-key")
os.environ.setdefault("GOOGLE_CLIENT_ID", "test-client-id.apps.googleusercontent.com")


@pytest.fixture
def billing_url():
    """Billing API URL for tests."""
    return "https://billing.test.ciris.ai"


@pytest.fixture
def billing_key():
    """Billing API key for tests."""
    return "test-billing-key"


@pytest.fixture
def sample_user_api_key_dict():
    """Sample user API key dict as passed by LiteLLM."""
    return {
        "api_key": "google:test-user-123",
        "user_id": None,
        "team_id": None,
    }


@pytest.fixture
def sample_request_data():
    """Sample request data for LLM calls."""
    return {
        "model": "groq/llama-3.1-70b",
        "messages": [{"role": "user", "content": "Hello"}],
        "metadata": {
            "interaction_id": "int-test-456",
        },
    }


@pytest.fixture
def sample_response_obj():
    """Sample LiteLLM response object."""
    response = MagicMock()
    response.usage = MagicMock()
    response.usage.prompt_tokens = 100
    response.usage.completion_tokens = 50
    response._hidden_params = {"response_cost": 0.0012}  # $0.0012 = 0.12 cents
    return response


@pytest.fixture
def mock_litellm_params():
    """Mock litellm_params dict for post-call hooks."""
    return {
        "metadata": {
            "_ciris_oauth_provider": "oauth:google",
            "_ciris_external_id": "test-user-123",
            "_ciris_interaction_id": "int-test-456",
        }
    }


# =============================================================================
# Google ID Token Fixtures
# =============================================================================

@pytest.fixture
def google_client_id():
    """Google OAuth client ID for tests."""
    return "test-client-id.apps.googleusercontent.com"


@pytest.fixture
def sample_google_user_id():
    """Sample Google user ID (sub claim)."""
    return "118234567890123456789"


@pytest.fixture
def sample_google_id_token_claims(sample_google_user_id, google_client_id):
    """Sample decoded Google ID token claims."""
    return {
        "iss": "https://accounts.google.com",
        "azp": google_client_id,
        "aud": google_client_id,
        "sub": sample_google_user_id,
        "email": "testuser@gmail.com",
        "email_verified": True,
        "name": "Test User",
        "picture": "https://lh3.googleusercontent.com/a/default-user",
        "given_name": "Test",
        "family_name": "User",
        "iat": int(time.time()),
        "exp": int(time.time()) + 3600,  # Valid for 1 hour
    }


@pytest.fixture
def expired_google_id_token_claims(sample_google_user_id, google_client_id):
    """Sample expired Google ID token claims."""
    return {
        "iss": "https://accounts.google.com",
        "azp": google_client_id,
        "aud": google_client_id,
        "sub": sample_google_user_id,
        "email": "testuser@gmail.com",
        "email_verified": True,
        "iat": int(time.time()) - 7200,  # 2 hours ago
        "exp": int(time.time()) - 3600,  # Expired 1 hour ago
    }


@pytest.fixture
def wrong_audience_token_claims(sample_google_user_id):
    """Token claims with wrong audience."""
    return {
        "iss": "https://accounts.google.com",
        "azp": "wrong-client-id.apps.googleusercontent.com",
        "aud": "wrong-client-id.apps.googleusercontent.com",
        "sub": sample_google_user_id,
        "email": "testuser@gmail.com",
        "iat": int(time.time()),
        "exp": int(time.time()) + 3600,
    }


@pytest.fixture
def mock_google_verify_success(sample_google_id_token_claims):
    """Mock google.oauth2.id_token.verify_oauth2_token to return valid claims."""
    with patch("hooks.custom_auth.id_token") as mock_id_token:
        mock_id_token.verify_oauth2_token.return_value = sample_google_id_token_claims
        yield mock_id_token


@pytest.fixture
def mock_google_verify_expired():
    """Mock google.oauth2.id_token.verify_oauth2_token to raise expired error."""
    with patch("hooks.custom_auth.id_token") as mock_id_token:
        mock_id_token.verify_oauth2_token.side_effect = ValueError(
            "Token has expired"
        )
        yield mock_id_token


@pytest.fixture
def mock_google_verify_invalid_signature():
    """Mock google.oauth2.id_token.verify_oauth2_token to raise signature error."""
    with patch("hooks.custom_auth.id_token") as mock_id_token:
        mock_id_token.verify_oauth2_token.side_effect = ValueError(
            "Could not verify token signature"
        )
        yield mock_id_token


@pytest.fixture
def mock_google_verify_wrong_audience():
    """Mock google.oauth2.id_token.verify_oauth2_token to raise audience error."""
    with patch("hooks.custom_auth.id_token") as mock_id_token:
        mock_id_token.verify_oauth2_token.side_effect = ValueError(
            "Token has wrong audience"
        )
        yield mock_id_token


@pytest.fixture
def mock_fastapi_request():
    """Mock FastAPI Request object."""
    request = MagicMock()
    request.url.path = "/v1/chat/completions"
    request.method = "POST"
    request.client.host = "192.168.1.1"
    return request


@pytest.fixture
def sample_jwt_token():
    """A sample JWT-like string (not valid, just for format testing)."""
    # This is a fake JWT with 3 parts separated by dots
    return "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMTgyMzQ1Njc4OTAxMjM0NTY3ODkiLCJhdWQiOiJ0ZXN0LWNsaWVudC1pZC5hcHBzLmdvb2dsZXVzZXJjb250ZW50LmNvbSIsImlzcyI6Imh0dHBzOi8vYWNjb3VudHMuZ29vZ2xlLmNvbSIsImlhdCI6MTcwMDAwMDAwMCwiZXhwIjoxNzAwMDAzNjAwfQ.fake_signature_here"
