"""
Unit tests for custom_auth helper functions and main auth functions.

Uses LiteLLM testing patterns:
- AsyncMock for async dependency mocking
- patch for isolating Google auth
- parametrize for multiple auth scenarios
"""

import time
from unittest.mock import MagicMock, patch, AsyncMock

import pytest
from hypothesis import given, strategies as st, settings

# Mock the litellm imports before importing custom_auth
import sys
mock_litellm_types = MagicMock()
mock_litellm_proxy = MagicMock()


class MockUserAPIKeyAuth:
    """Mock UserAPIKeyAuth for testing."""

    def __init__(self, api_key: str, user_id: str):
        self.api_key = api_key
        self.user_id = user_id


class MockProxyException(Exception):
    """Mock ProxyException for testing."""

    def __init__(self, message: str, type: str, param: str = None, code: int = 401):
        self.message = message
        self.type = type
        self.param = param
        self.code = code
        super().__init__(message)


class MockRequest:
    """Mock FastAPI Request for testing."""

    def __init__(self):
        self.headers = {}


mock_litellm_types.UserAPIKeyAuth = MockUserAPIKeyAuth
mock_litellm_proxy.ProxyException = MockProxyException

sys.modules["litellm.proxy._types"] = mock_litellm_types
sys.modules["litellm.proxy.proxy_server"] = mock_litellm_proxy

# Now import the module
from hooks.custom_auth import (
    _cleanup_cache,
    _get_cached_auth,
    _import_google_auth,
    _try_verify_token,
    _validate_expired_token_claims,
    _handle_expired_token,
    _extract_user_id,
    _cache_token,
    _handle_verification_error,
    _get_cached_idinfo,
    _try_import_google_auth_silent,
    _try_decode_expired_token_silent,
    _token_cache,
    _CACHE_DURATION_SECONDS,
    GOOGLE_CLIENT_IDS,
    user_api_key_auth,
    verify_google_token,
)


class TestCleanupCache:
    """Tests for _cleanup_cache function."""

    def setup_method(self):
        """Clear cache before each test."""
        _token_cache.clear()

    def test_cleanup_does_nothing_under_limit(self):
        """Test that cleanup doesn't run when cache is under limit."""
        _token_cache["token1"] = ("user1", time.time() + 100)
        _token_cache["token2"] = ("user2", time.time() + 100)

        _cleanup_cache()

        assert len(_token_cache) == 2

    def test_cleanup_removes_expired_entries(self):
        """Test that cleanup removes expired entries when over limit."""
        # Add many entries to trigger cleanup
        for i in range(10001):
            if i < 5000:
                # Half are expired
                _token_cache[f"token{i}"] = (f"user{i}", time.time() - 100)
            else:
                # Half are valid
                _token_cache[f"token{i}"] = (f"user{i}", time.time() + 100)

        _cleanup_cache()

        # Expired entries should be removed
        assert len(_token_cache) <= 5001


class TestGetCachedAuth:
    """Tests for _get_cached_auth function."""

    def setup_method(self):
        """Clear cache before each test."""
        _token_cache.clear()

    def test_returns_none_for_uncached_token(self):
        """Test that None is returned for tokens not in cache."""
        result = _get_cached_auth("uncached-token")
        assert result is None

    def test_returns_auth_for_valid_cached_token(self):
        """Test that valid cached tokens return auth object."""
        _token_cache["valid-token"] = ("user123", time.time() + 3600)

        result = _get_cached_auth("valid-token")

        assert result is not None
        assert result.api_key == "google:user123"
        assert result.user_id == "user123"

    def test_removes_and_returns_none_for_expired_token(self):
        """Test that expired cached tokens are removed and None returned."""
        _token_cache["expired-token"] = ("user456", time.time() - 100)

        result = _get_cached_auth("expired-token")

        assert result is None
        assert "expired-token" not in _token_cache


class TestImportGoogleAuth:
    """Tests for _import_google_auth function."""

    def test_raises_proxy_exception_when_import_fails(self):
        """Test that ProxyException is raised when google-auth not installed."""
        with patch.dict(sys.modules, {"google.oauth2": None, "google.auth.transport": None}):
            # Force ImportError by making the import fail
            original_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

            def mock_import(name, *args, **kwargs):
                if "google" in name:
                    raise ImportError("No module named 'google'")
                return original_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=mock_import):
                with pytest.raises(MockProxyException) as exc_info:
                    _import_google_auth()

                assert exc_info.value.code == 500
                assert "google-auth not installed" in exc_info.value.message


class TestTryVerifyToken:
    """Tests for _try_verify_token function."""

    def test_returns_idinfo_on_success(self):
        """Test successful token verification."""
        mock_id_token = MagicMock()
        mock_id_token.verify_oauth2_token.return_value = {"sub": "user123", "aud": "client1"}
        mock_requests = MagicMock()

        idinfo, error = _try_verify_token("valid-token", mock_id_token, mock_requests)

        assert idinfo == {"sub": "user123", "aud": "client1"}
        assert error is None

    def test_tries_all_client_ids_on_audience_error(self):
        """Test that all client IDs are tried on audience mismatch."""
        mock_id_token = MagicMock()
        mock_id_token.verify_oauth2_token.side_effect = [
            Exception("Wrong audience"),
            {"sub": "user123", "aud": "client2"},
        ]
        mock_requests = MagicMock()

        idinfo, error = _try_verify_token("valid-token", mock_id_token, mock_requests)

        assert idinfo == {"sub": "user123", "aud": "client2"}
        assert error is None
        assert mock_id_token.verify_oauth2_token.call_count == 2

    def test_stops_on_expired_error(self):
        """Test that verification stops on expired token error."""
        mock_id_token = MagicMock()
        mock_id_token.verify_oauth2_token.side_effect = Exception("Token has expired")
        mock_requests = MagicMock()

        idinfo, error = _try_verify_token("expired-token", mock_id_token, mock_requests)

        assert idinfo is None
        assert error is not None
        assert "expired" in str(error).lower()
        # Should stop after first expired error, not try all client IDs
        assert mock_id_token.verify_oauth2_token.call_count == 1

    def test_returns_last_error_on_failure(self):
        """Test that last error is returned when all verifications fail."""
        mock_id_token = MagicMock()
        mock_id_token.verify_oauth2_token.side_effect = Exception("Invalid signature")
        mock_requests = MagicMock()

        idinfo, error = _try_verify_token("invalid-token", mock_id_token, mock_requests)

        assert idinfo is None
        assert error is not None


class TestValidateExpiredTokenClaims:
    """Tests for _validate_expired_token_claims function."""

    def test_valid_claims_pass(self):
        """Test that valid claims don't raise exception."""
        idinfo = {
            "aud": GOOGLE_CLIENT_IDS[0],
            "iss": "accounts.google.com",
            "sub": "user123",
        }

        # Should not raise
        _validate_expired_token_claims(idinfo)

    def test_invalid_audience_raises(self):
        """Test that invalid audience raises ProxyException."""
        idinfo = {
            "aud": "wrong-client-id",
            "iss": "accounts.google.com",
            "sub": "user123",
        }

        with pytest.raises(MockProxyException) as exc_info:
            _validate_expired_token_claims(idinfo)

        assert exc_info.value.code == 401
        assert "audience" in exc_info.value.message.lower()

    def test_invalid_issuer_raises(self):
        """Test that invalid issuer raises ProxyException."""
        idinfo = {
            "aud": GOOGLE_CLIENT_IDS[0],
            "iss": "malicious-issuer.com",
            "sub": "user123",
        }

        with pytest.raises(MockProxyException) as exc_info:
            _validate_expired_token_claims(idinfo)

        assert exc_info.value.code == 401
        assert "issuer" in exc_info.value.message.lower()

    def test_https_issuer_accepted(self):
        """Test that https://accounts.google.com is also valid."""
        idinfo = {
            "aud": GOOGLE_CLIENT_IDS[0],
            "iss": "https://accounts.google.com",
            "sub": "user123",
        }

        # Should not raise
        _validate_expired_token_claims(idinfo)


class TestHandleExpiredToken:
    """Tests for _handle_expired_token function."""

    def test_decodes_expired_token(self):
        """Test that expired token is decoded without verification."""
        mock_jwt = MagicMock()
        mock_jwt.decode.return_value = {
            "aud": GOOGLE_CLIENT_IDS[0],
            "iss": "accounts.google.com",
            "sub": "user123",
        }

        result = _handle_expired_token(
            "expired-token",
            mock_jwt,
            Exception("Token has expired"),
        )

        assert result["sub"] == "user123"
        mock_jwt.decode.assert_called_once_with("expired-token", verify=False)

    def test_reraises_non_expiration_error(self):
        """Test that non-expiration errors are re-raised."""
        mock_jwt = MagicMock()

        with pytest.raises(Exception) as exc_info:
            _handle_expired_token(
                "token",
                mock_jwt,
                Exception("Invalid signature"),
            )

        assert "Invalid signature" in str(exc_info.value)
        mock_jwt.decode.assert_not_called()

    def test_raises_proxy_exception_on_decode_error(self):
        """Test that decode errors raise ProxyException."""
        mock_jwt = MagicMock()
        mock_jwt.decode.side_effect = Exception("Malformed token")

        with pytest.raises(MockProxyException) as exc_info:
            _handle_expired_token(
                "malformed-token",
                mock_jwt,
                Exception("Token has expired"),
            )

        assert exc_info.value.code == 401
        assert "decode" in exc_info.value.message.lower()


class TestExtractUserId:
    """Tests for _extract_user_id function."""

    def test_extracts_user_id(self):
        """Test successful user ID extraction."""
        idinfo = {"sub": "user123", "email": "user@example.com"}

        result = _extract_user_id(idinfo)

        assert result == "user123"

    def test_raises_on_missing_user_id(self):
        """Test that missing user ID raises ProxyException."""
        idinfo = {"email": "user@example.com"}  # No 'sub' claim

        with pytest.raises(MockProxyException) as exc_info:
            _extract_user_id(idinfo)

        assert exc_info.value.code == 401
        assert "missing user ID" in exc_info.value.message


class TestCacheToken:
    """Tests for _cache_token function."""

    def setup_method(self):
        """Clear cache before each test."""
        _token_cache.clear()

    def test_caches_token(self):
        """Test that token is cached correctly."""
        _cache_token("my-token", "user123")

        assert "my-token" in _token_cache
        user_id, cache_until = _token_cache["my-token"]
        assert user_id == "user123"
        assert cache_until > time.time()
        assert cache_until <= time.time() + _CACHE_DURATION_SECONDS + 1


class TestHandleVerificationError:
    """Tests for _handle_verification_error function."""

    def test_audience_error(self):
        """Test that audience errors produce appropriate message."""
        with pytest.raises(MockProxyException) as exc_info:
            _handle_verification_error(Exception("Wrong audience for token"))

        assert exc_info.value.code == 401
        assert "audience" in exc_info.value.message.lower()

    def test_expired_error(self):
        """Test that expired errors produce appropriate message."""
        with pytest.raises(MockProxyException) as exc_info:
            _handle_verification_error(Exception("Token has expired"))

        assert exc_info.value.code == 401
        assert "expired" in exc_info.value.message.lower()

    def test_generic_error(self):
        """Test that other errors produce generic message."""
        with pytest.raises(MockProxyException) as exc_info:
            _handle_verification_error(Exception("Some internal error with secret info"))

        assert exc_info.value.code == 401
        assert "Invalid authentication token" in exc_info.value.message
        # Should not leak internal error details
        assert "secret" not in exc_info.value.message


# =============================================================================
# Tests for verify_google_token helper functions
# =============================================================================


class TestGetCachedIdinfo:
    """Tests for _get_cached_idinfo function."""

    def setup_method(self):
        """Clear cache before each test."""
        _token_cache.clear()

    def test_returns_none_for_uncached_token(self):
        """Test returns None for tokens not in cache."""
        result = _get_cached_idinfo("uncached-token")
        assert result is None

    def test_returns_idinfo_for_valid_cached_token(self):
        """Test returns idinfo dict for valid cached tokens."""
        _token_cache["valid-token"] = ("user123", time.time() + 3600)

        result = _get_cached_idinfo("valid-token")

        assert result is not None
        assert result["sub"] == "user123"

    def test_removes_and_returns_none_for_expired_token(self):
        """Test removes expired tokens and returns None."""
        _token_cache["expired-token"] = ("user456", time.time() - 100)

        result = _get_cached_idinfo("expired-token")

        assert result is None
        assert "expired-token" not in _token_cache


class TestTryImportGoogleAuthSilent:
    """Tests for _try_import_google_auth_silent function."""

    def test_returns_none_when_import_fails(self):
        """Test returns None when google-auth not installed."""
        with patch.dict(sys.modules, {"google.oauth2": None, "google.auth.transport": None}):
            original_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

            def mock_import(name, *args, **kwargs):
                if "google" in name:
                    raise ImportError("No module named 'google'")
                return original_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=mock_import):
                result = _try_import_google_auth_silent()

            # Result should be None (not raise exception)
            assert result is None


class TestTryDecodeExpiredTokenSilent:
    """Tests for _try_decode_expired_token_silent function."""

    def test_returns_none_for_non_expired_error(self):
        """Test returns None when error is not expiration."""
        mock_jwt = MagicMock()
        result = _try_decode_expired_token_silent(
            "token",
            mock_jwt,
            Exception("Invalid signature")
        )
        assert result is None
        mock_jwt.decode.assert_not_called()

    def test_returns_none_when_no_error(self):
        """Test returns None when last_error is None."""
        mock_jwt = MagicMock()
        result = _try_decode_expired_token_silent("token", mock_jwt, None)
        assert result is None

    def test_decodes_expired_token_with_valid_claims(self):
        """Test decodes expired token when claims are valid."""
        mock_jwt = MagicMock()
        mock_jwt.decode.return_value = {
            "aud": GOOGLE_CLIENT_IDS[0],
            "iss": "accounts.google.com",
            "sub": "user123",
        }

        result = _try_decode_expired_token_silent(
            "expired-token",
            mock_jwt,
            Exception("Token has expired")
        )

        assert result is not None
        assert result["sub"] == "user123"

    def test_returns_none_for_invalid_audience(self):
        """Test returns None when audience is invalid."""
        mock_jwt = MagicMock()
        mock_jwt.decode.return_value = {
            "aud": "wrong-client-id",
            "iss": "accounts.google.com",
            "sub": "user123",
        }

        result = _try_decode_expired_token_silent(
            "expired-token",
            mock_jwt,
            Exception("Token has expired")
        )

        assert result is None

    def test_returns_none_on_decode_error(self):
        """Test returns None when decode fails."""
        mock_jwt = MagicMock()
        mock_jwt.decode.side_effect = Exception("Malformed token")

        result = _try_decode_expired_token_silent(
            "bad-token",
            mock_jwt,
            Exception("Token has expired")
        )

        assert result is None


# =============================================================================
# Tests for user_api_key_auth main function
# =============================================================================


class TestUserApiKeyAuth:
    """Tests for user_api_key_auth async function."""

    def setup_method(self):
        """Clear cache before each test."""
        _token_cache.clear()

    @pytest.mark.asyncio
    async def test_raises_on_missing_api_key(self):
        """Test raises ProxyException when api_key is empty."""
        request = MockRequest()

        with pytest.raises(MockProxyException) as exc_info:
            await user_api_key_auth(request, "")

        assert exc_info.value.code == 401
        assert "Missing" in exc_info.value.message

    @pytest.mark.asyncio
    async def test_returns_cached_auth(self):
        """Test returns cached auth without verification."""
        request = MockRequest()
        _token_cache["cached-token"] = ("user123", time.time() + 3600)

        result = await user_api_key_auth(request, "cached-token")

        assert result.api_key == "google:user123"
        assert result.user_id == "user123"

    @pytest.mark.asyncio
    async def test_successful_verification(self):
        """Test successful token verification flow."""
        request = MockRequest()

        mock_id_token = MagicMock()
        mock_id_token.verify_oauth2_token.return_value = {
            "sub": "user456",
            "aud": GOOGLE_CLIENT_IDS[0],
        }
        mock_requests = MagicMock()
        mock_jwt = MagicMock()

        with patch("hooks.custom_auth._import_google_auth") as mock_import:
            mock_import.return_value = (mock_id_token, mock_requests, mock_jwt)

            result = await user_api_key_auth(request, "valid-token")

        assert result.api_key == "google:user456"
        assert result.user_id == "user456"
        # Token should be cached
        assert "valid-token" in _token_cache

    @pytest.mark.asyncio
    async def test_handles_expired_token(self):
        """Test handles expired token by decoding without verification."""
        request = MockRequest()

        mock_id_token = MagicMock()
        mock_id_token.verify_oauth2_token.side_effect = Exception("Token has expired")
        mock_requests = MagicMock()
        mock_jwt = MagicMock()
        mock_jwt.decode.return_value = {
            "sub": "user789",
            "aud": GOOGLE_CLIENT_IDS[0],
            "iss": "accounts.google.com",
        }

        with patch("hooks.custom_auth._import_google_auth") as mock_import:
            mock_import.return_value = (mock_id_token, mock_requests, mock_jwt)

            result = await user_api_key_auth(request, "expired-token")

        assert result.api_key == "google:user789"
        assert result.user_id == "user789"

    @pytest.mark.asyncio
    async def test_raises_on_verification_failure(self):
        """Test raises ProxyException on verification failure."""
        request = MockRequest()

        mock_id_token = MagicMock()
        mock_id_token.verify_oauth2_token.side_effect = Exception("Invalid signature")
        mock_requests = MagicMock()
        mock_jwt = MagicMock()

        with patch("hooks.custom_auth._import_google_auth") as mock_import:
            mock_import.return_value = (mock_id_token, mock_requests, mock_jwt)

            with pytest.raises(MockProxyException) as exc_info:
                await user_api_key_auth(request, "invalid-token")

        assert exc_info.value.code == 401


# =============================================================================
# Tests for verify_google_token main function
# =============================================================================


class TestVerifyGoogleToken:
    """Tests for verify_google_token async function."""

    def setup_method(self):
        """Clear cache before each test."""
        _token_cache.clear()

    @pytest.mark.asyncio
    async def test_returns_none_for_empty_token(self):
        """Test returns None for empty token."""
        result = await verify_google_token("")
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_for_none_token(self):
        """Test returns None for None token."""
        result = await verify_google_token(None)
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_cached_idinfo(self):
        """Test returns cached idinfo."""
        _token_cache["cached-token"] = ("user123", time.time() + 3600)

        result = await verify_google_token("cached-token")

        assert result is not None
        assert result["sub"] == "user123"

    @pytest.mark.asyncio
    async def test_returns_none_when_google_auth_not_installed(self):
        """Test returns None when google-auth not available."""
        with patch("hooks.custom_auth._try_import_google_auth_silent", return_value=None):
            result = await verify_google_token("some-token")

        assert result is None

    @pytest.mark.asyncio
    async def test_successful_verification(self):
        """Test successful token verification."""
        mock_id_token = MagicMock()
        mock_id_token.verify_oauth2_token.return_value = {
            "sub": "user456",
            "aud": GOOGLE_CLIENT_IDS[0],
            "email": "user@example.com",
        }
        mock_requests = MagicMock()
        mock_jwt = MagicMock()

        with patch("hooks.custom_auth._try_import_google_auth_silent") as mock_import:
            mock_import.return_value = (mock_id_token, mock_requests, mock_jwt)

            result = await verify_google_token("valid-token")

        assert result is not None
        assert result["sub"] == "user456"
        assert result["email"] == "user@example.com"
        # Should be cached
        assert "valid-token" in _token_cache

    @pytest.mark.asyncio
    async def test_handles_expired_token(self):
        """Test handles expired token gracefully."""
        mock_id_token = MagicMock()
        mock_id_token.verify_oauth2_token.side_effect = Exception("Token has expired")
        mock_requests = MagicMock()
        mock_jwt = MagicMock()
        mock_jwt.decode.return_value = {
            "sub": "user789",
            "aud": GOOGLE_CLIENT_IDS[0],
            "iss": "accounts.google.com",
        }

        with patch("hooks.custom_auth._try_import_google_auth_silent") as mock_import:
            mock_import.return_value = (mock_id_token, mock_requests, mock_jwt)

            result = await verify_google_token("expired-token")

        assert result is not None
        assert result["sub"] == "user789"

    @pytest.mark.asyncio
    async def test_returns_none_on_verification_failure(self):
        """Test returns None on verification failure (no exception)."""
        mock_id_token = MagicMock()
        mock_id_token.verify_oauth2_token.side_effect = Exception("Invalid signature")
        mock_requests = MagicMock()
        mock_jwt = MagicMock()

        with patch("hooks.custom_auth._try_import_google_auth_silent") as mock_import:
            mock_import.return_value = (mock_id_token, mock_requests, mock_jwt)

            result = await verify_google_token("invalid-token")

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_no_user_id(self):
        """Test returns None when token has no sub claim."""
        mock_id_token = MagicMock()
        mock_id_token.verify_oauth2_token.return_value = {
            "aud": GOOGLE_CLIENT_IDS[0],
            # Missing "sub"
        }
        mock_requests = MagicMock()
        mock_jwt = MagicMock()

        with patch("hooks.custom_auth._try_import_google_auth_silent") as mock_import:
            mock_import.return_value = (mock_id_token, mock_requests, mock_jwt)

            result = await verify_google_token("no-sub-token")

        assert result is None
