"""
Pytest fixtures for CIRISProxy tests.
"""

import os
from unittest.mock import AsyncMock, MagicMock

import pytest

# Set test environment variables before importing hooks
os.environ.setdefault("BILLING_API_URL", "https://billing.test.ciris.ai")
os.environ.setdefault("BILLING_API_KEY", "test-billing-key")


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
