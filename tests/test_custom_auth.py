"""
Unit tests for custom_auth.py - Google ID Token verification.
"""

import sys
import time
from unittest.mock import MagicMock, patch

import pytest

from hooks.custom_auth import (
    GOOGLE_CLIENT_ID,
    _cleanup_cache,
    _token_cache,
    user_api_key_auth,
)


# Create mock modules for google.oauth2.id_token and google.auth.transport.requests
# These need to be created before the tests run since the imports happen inside the function
def create_mock_google_modules():
    """Create mock google auth modules and inject them into sys.modules."""
    mock_id_token = MagicMock()
    mock_google_requests = MagicMock()
    mock_request_class = MagicMock()
    mock_google_requests.Request = mock_request_class

    return mock_id_token, mock_google_requests


class TestUserApiKeyAuth:
    """Tests for user_api_key_auth function."""

    @pytest.fixture(autouse=True)
    def clear_cache(self):
        """Clear token cache before each test."""
        _token_cache.clear()
        yield
        _token_cache.clear()

    @pytest.fixture
    def mock_google_auth(self):
        """Set up mock google auth modules."""
        mock_id_token = MagicMock()
        mock_google_requests = MagicMock()

        # Store original modules if they exist
        orig_id_token = sys.modules.get('google.oauth2.id_token')
        orig_google_requests = sys.modules.get('google.auth.transport.requests')

        # Inject mocks
        sys.modules['google.oauth2.id_token'] = mock_id_token
        sys.modules['google.auth.transport.requests'] = mock_google_requests

        yield mock_id_token, mock_google_requests

        # Restore original modules
        if orig_id_token is not None:
            sys.modules['google.oauth2.id_token'] = orig_id_token
        else:
            sys.modules.pop('google.oauth2.id_token', None)

        if orig_google_requests is not None:
            sys.modules['google.auth.transport.requests'] = orig_google_requests
        else:
            sys.modules.pop('google.auth.transport.requests', None)

    @pytest.mark.asyncio
    async def test_missing_token_raises_401(self, mock_fastapi_request):
        """Test that missing token raises ProxyException with 401."""
        from litellm.proxy.proxy_server import ProxyException

        with pytest.raises(ProxyException) as exc_info:
            await user_api_key_auth(mock_fastapi_request, "")

        # ProxyException.code can be int or string depending on version
        assert str(exc_info.value.code) == "401"
        assert "Missing" in exc_info.value.message

    @pytest.mark.asyncio
    async def test_valid_token_returns_user_api_key_auth(
        self,
        mock_fastapi_request,
        sample_jwt_token,
        sample_google_user_id,
        sample_google_id_token_claims,
        mock_google_auth,
    ):
        """Test that valid token returns UserAPIKeyAuth with google:{user_id}."""
        mock_id_token, mock_google_requests = mock_google_auth
        mock_id_token.verify_oauth2_token.return_value = sample_google_id_token_claims

        result = await user_api_key_auth(mock_fastapi_request, sample_jwt_token)

        assert result.api_key == f"google:{sample_google_user_id}"
        assert result.user_id == sample_google_user_id

    @pytest.mark.asyncio
    async def test_token_cached_after_verification(
        self,
        mock_fastapi_request,
        sample_jwt_token,
        sample_google_user_id,
        sample_google_id_token_claims,
        mock_google_auth,
    ):
        """Test that verified token is cached."""
        mock_id_token, mock_google_requests = mock_google_auth
        mock_id_token.verify_oauth2_token.return_value = sample_google_id_token_claims

        # First call - should verify
        await user_api_key_auth(mock_fastapi_request, sample_jwt_token)

        # Token should be cached
        assert sample_jwt_token in _token_cache
        cached_user_id, cached_expiry = _token_cache[sample_jwt_token]
        assert cached_user_id == sample_google_user_id
        assert cached_expiry > time.time()

    @pytest.mark.asyncio
    async def test_cached_token_skips_verification(
        self,
        mock_fastapi_request,
        sample_jwt_token,
        sample_google_user_id,
        mock_google_auth,
    ):
        """Test that cached token skips Google verification."""
        mock_id_token, mock_google_requests = mock_google_auth

        # Pre-populate cache
        _token_cache[sample_jwt_token] = (sample_google_user_id, time.time() + 3600)

        result = await user_api_key_auth(mock_fastapi_request, sample_jwt_token)

        # Should NOT have called verify
        mock_id_token.verify_oauth2_token.assert_not_called()

        # Should still return correct result
        assert result.api_key == f"google:{sample_google_user_id}"

    @pytest.mark.asyncio
    async def test_expired_cache_triggers_reverification(
        self,
        mock_fastapi_request,
        sample_jwt_token,
        sample_google_user_id,
        sample_google_id_token_claims,
        mock_google_auth,
    ):
        """Test that expired cache entry triggers re-verification."""
        mock_id_token, mock_google_requests = mock_google_auth
        mock_id_token.verify_oauth2_token.return_value = sample_google_id_token_claims

        # Pre-populate cache with expired entry
        _token_cache[sample_jwt_token] = (sample_google_user_id, time.time() - 100)

        await user_api_key_auth(mock_fastapi_request, sample_jwt_token)

        # Should have called verify because cache was expired
        mock_id_token.verify_oauth2_token.assert_called_once()

    @pytest.mark.asyncio
    async def test_expired_token_raises_401(
        self,
        mock_fastapi_request,
        sample_jwt_token,
        mock_google_auth,
    ):
        """Test that expired token raises ProxyException with 401."""
        from litellm.proxy.proxy_server import ProxyException

        mock_id_token, mock_google_requests = mock_google_auth
        mock_id_token.verify_oauth2_token.side_effect = ValueError("Token has expired")

        with pytest.raises(ProxyException) as exc_info:
            await user_api_key_auth(mock_fastapi_request, sample_jwt_token)

        assert str(exc_info.value.code) == "401"
        assert "expired" in exc_info.value.message.lower()

    @pytest.mark.asyncio
    async def test_invalid_signature_raises_401(
        self,
        mock_fastapi_request,
        sample_jwt_token,
        mock_google_auth,
    ):
        """Test that invalid signature raises ProxyException with 401."""
        from litellm.proxy.proxy_server import ProxyException

        mock_id_token, mock_google_requests = mock_google_auth
        mock_id_token.verify_oauth2_token.side_effect = ValueError(
            "Could not verify token signature"
        )

        with pytest.raises(ProxyException) as exc_info:
            await user_api_key_auth(mock_fastapi_request, sample_jwt_token)

        assert str(exc_info.value.code) == "401"
        assert "Invalid Google ID token" in exc_info.value.message

    @pytest.mark.asyncio
    async def test_wrong_audience_raises_401(
        self,
        mock_fastapi_request,
        sample_jwt_token,
        mock_google_auth,
    ):
        """Test that wrong audience raises ProxyException with 401."""
        from litellm.proxy.proxy_server import ProxyException

        mock_id_token, mock_google_requests = mock_google_auth
        mock_id_token.verify_oauth2_token.side_effect = ValueError(
            "Token has wrong audience"
        )

        with pytest.raises(ProxyException) as exc_info:
            await user_api_key_auth(mock_fastapi_request, sample_jwt_token)

        assert str(exc_info.value.code) == "401"
        assert "audience" in exc_info.value.message.lower()

    @pytest.mark.asyncio
    async def test_missing_sub_claim_raises_401(
        self,
        mock_fastapi_request,
        sample_jwt_token,
        sample_google_id_token_claims,
        mock_google_auth,
    ):
        """Test that token without sub claim raises ProxyException with 401."""
        from litellm.proxy.proxy_server import ProxyException

        mock_id_token, mock_google_requests = mock_google_auth

        # Remove sub claim
        claims_without_sub = {k: v for k, v in sample_google_id_token_claims.items() if k != "sub"}
        mock_id_token.verify_oauth2_token.return_value = claims_without_sub

        with pytest.raises(ProxyException) as exc_info:
            await user_api_key_auth(mock_fastapi_request, sample_jwt_token)

        assert str(exc_info.value.code) == "401"
        assert "missing user ID" in exc_info.value.message


class TestCacheCleanup:
    """Tests for cache cleanup functionality."""

    @pytest.fixture(autouse=True)
    def clear_cache(self):
        """Clear token cache before each test."""
        _token_cache.clear()
        yield
        _token_cache.clear()

    def test_cleanup_removes_expired_entries(self):
        """Test that cleanup removes expired entries."""
        # Add expired entry
        _token_cache["expired_token"] = ("user1", time.time() - 100)
        # Add valid entry
        _token_cache["valid_token"] = ("user2", time.time() + 3600)

        # Force cleanup by exceeding max size
        from hooks.custom_auth import _MAX_CACHE_SIZE
        for i in range(_MAX_CACHE_SIZE):
            _token_cache[f"token_{i}"] = ("user", time.time() + 3600)

        _cleanup_cache()

        # Expired should be removed
        assert "expired_token" not in _token_cache

    def test_cleanup_skips_when_under_limit(self):
        """Test that cleanup doesn't run when under size limit."""
        # Add expired entry
        _token_cache["expired_token"] = ("user1", time.time() - 100)
        # Add valid entry
        _token_cache["valid_token"] = ("user2", time.time() + 3600)

        _cleanup_cache()

        # Expired should still be there (under limit)
        assert "expired_token" in _token_cache


class TestTokenCaching:
    """Tests for token caching behavior."""

    @pytest.fixture(autouse=True)
    def clear_cache(self):
        """Clear token cache before each test."""
        _token_cache.clear()
        yield
        _token_cache.clear()

    @pytest.fixture
    def mock_google_auth(self):
        """Set up mock google auth modules."""
        mock_id_token = MagicMock()
        mock_google_requests = MagicMock()

        # Store original modules if they exist
        orig_id_token = sys.modules.get('google.oauth2.id_token')
        orig_google_requests = sys.modules.get('google.auth.transport.requests')

        # Inject mocks
        sys.modules['google.oauth2.id_token'] = mock_id_token
        sys.modules['google.auth.transport.requests'] = mock_google_requests

        yield mock_id_token, mock_google_requests

        # Restore original modules
        if orig_id_token is not None:
            sys.modules['google.oauth2.id_token'] = orig_id_token
        else:
            sys.modules.pop('google.oauth2.id_token', None)

        if orig_google_requests is not None:
            sys.modules['google.auth.transport.requests'] = orig_google_requests
        else:
            sys.modules.pop('google.auth.transport.requests', None)

    @pytest.mark.asyncio
    async def test_multiple_tokens_cached_independently(
        self,
        mock_fastapi_request,
        sample_google_id_token_claims,
        mock_google_auth,
    ):
        """Test that different tokens are cached independently."""
        mock_id_token, mock_google_requests = mock_google_auth

        token1 = "token1.payload.sig"
        token2 = "token2.payload.sig"

        claims1 = {**sample_google_id_token_claims, "sub": "user1"}
        claims2 = {**sample_google_id_token_claims, "sub": "user2"}

        # First token
        mock_id_token.verify_oauth2_token.return_value = claims1
        result1 = await user_api_key_auth(mock_fastapi_request, token1)

        # Second token
        mock_id_token.verify_oauth2_token.return_value = claims2
        result2 = await user_api_key_auth(mock_fastapi_request, token2)

        assert result1.api_key == "google:user1"
        assert result2.api_key == "google:user2"

        # Both should be cached
        assert token1 in _token_cache
        assert token2 in _token_cache

    @pytest.mark.asyncio
    async def test_cache_expiry_respects_token_exp(
        self,
        mock_fastapi_request,
        sample_jwt_token,
        sample_google_id_token_claims,
        mock_google_auth,
    ):
        """Test that cache expiry is set based on token exp claim."""
        mock_id_token, mock_google_requests = mock_google_auth

        expected_exp = int(time.time()) + 1800  # 30 minutes
        claims = {**sample_google_id_token_claims, "exp": expected_exp}
        mock_id_token.verify_oauth2_token.return_value = claims

        await user_api_key_auth(mock_fastapi_request, sample_jwt_token)

        _, cached_expiry = _token_cache[sample_jwt_token]
        # Should be exp - 60 seconds (buffer)
        assert cached_expiry == expected_exp - 60


class TestEnvironmentConfig:
    """Tests for environment configuration."""

    def test_google_client_id_from_env(self):
        """Test that GOOGLE_CLIENT_ID is loaded from environment."""
        import os

        # The conftest sets this
        expected = os.environ.get("GOOGLE_CLIENT_ID")
        assert GOOGLE_CLIENT_ID == expected

    def test_default_google_client_id(self):
        """Test the default client ID value."""
        # The hooks/custom_auth.py has a default value
        import hooks.custom_auth as auth_module

        # Re-read the module to check default
        assert "apps.googleusercontent.com" in auth_module.GOOGLE_CLIENT_ID
