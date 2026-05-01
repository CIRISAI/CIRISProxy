"""
OAuth ID Token verification for LiteLLM Proxy.

Supports both Google and Apple ID tokens:
- Google: Authorization: Bearer {google_id_token}
- Apple: Authorization: Bearer {apple_id_token}

Verifies: Token signature against provider's public keys
Extracts: User's ID (sub claim) for billing

This replaces the simple numeric ID validation with proper JWT verification.
The ID tokens are cryptographically signed by the provider and cannot be forged.

NOTE: Expiration checking is DISABLED. Mobile apps may send tokens that
expired hours ago. We still verify the signature (token was issued by provider)
and audience (token was issued for our app). The user ID is stable and the
billing service handles authorization.

TEST AUTH MODE: When CIRIS_TEST_AUTH_ENABLED=true, the proxy also accepts
opaque test tokens (hex strings). These are validated by calling CIRISBilling's
/v1/billing/credits/check endpoint directly. This enables integration testing
without OAuth infrastructure.

SECURITY NOTE: Auth failures are logged for debugging but MUST NOT include:
- Full tokens (only format/length hints)
- User IDs or PII
- Request body content
Only log: token format classification, error type, user agent, request path
"""

import hashlib
import logging
import os
import time

import httpx
from fastapi import Request
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.proxy_server import ProxyException

# Configure logger for auth events
logger = logging.getLogger(__name__)


def _classify_token_format(token: str) -> dict[str, any]:
    """
    Classify a token's format for debugging without exposing the token itself.

    Returns metadata about the token structure that helps identify:
    - Bots sending garbage
    - Malformed tokens from buggy clients
    - Expired but valid tokens
    - Wrong audience tokens

    SECURITY: Only returns structural info, never the token content.
    """
    if not token:
        return {"format": "empty", "length": 0}

    result = {
        "length": len(token),
        "format": "unknown",
    }

    # Check for JWT structure (three base64 segments separated by dots)
    parts = token.split(".")
    if len(parts) == 3:
        # Looks like a JWT
        result["format"] = "jwt"
        result["header_len"] = len(parts[0])
        result["payload_len"] = len(parts[1])
        result["sig_len"] = len(parts[2])

        # Check if parts look like valid base64
        import base64
        try:
            # Try to decode header to check if it's valid base64
            # Add padding if needed
            padded = parts[0] + "=" * (4 - len(parts[0]) % 4)
            base64.urlsafe_b64decode(padded)
            result["header_valid_b64"] = True
        except Exception:
            result["header_valid_b64"] = False

    elif len(parts) == 2:
        result["format"] = "jwt_incomplete"
    elif token.startswith("sk-") or token.startswith("Bearer "):
        result["format"] = "api_key_like"
    elif len(token) < 20:
        result["format"] = "too_short"
    elif not token.replace("-", "").replace("_", "").isalnum():
        result["format"] = "contains_special_chars"
    else:
        result["format"] = "opaque_string"

    # Hash prefix for correlation (allows matching without exposing token)
    result["prefix_hash"] = hashlib.sha256(token[:8].encode()).hexdigest()[:8] if len(token) >= 8 else "short"

    return result


def _classify_auth_error(error: Exception) -> dict[str, str]:
    """
    Classify an auth error for structured logging.

    Returns:
        Dict with error_type and error_category for debugging.
    """
    error_msg = str(error).lower()

    if "expired" in error_msg:
        return {"error_type": "token_expired", "error_category": "temporal"}
    elif "audience" in error_msg:
        return {"error_type": "wrong_audience", "error_category": "configuration"}
    elif "issuer" in error_msg:
        return {"error_type": "wrong_issuer", "error_category": "configuration"}
    elif "signature" in error_msg or "verification" in error_msg:
        return {"error_type": "invalid_signature", "error_category": "security"}
    elif "decode" in error_msg or "malformed" in error_msg or "invalid" in error_msg:
        return {"error_type": "malformed_token", "error_category": "format"}
    elif "network" in error_msg or "connection" in error_msg or "timeout" in error_msg:
        return {"error_type": "network_error", "error_category": "infrastructure"}
    else:
        return {"error_type": "unknown", "error_category": "unknown"}


def _extract_request_metadata(request: Request | None) -> dict[str, str]:
    """
    Extract safe request metadata for logging.

    SECURITY: Only extracts non-PII metadata.
    - User-Agent helps identify bots vs real apps
    - Path helps identify which endpoint was targeted
    """
    if not request:
        return {}

    metadata = {}

    # User-Agent helps identify bots (curl, python-requests) vs real apps
    try:
        headers = getattr(request, "headers", None)
        user_agent = headers.get("user-agent", "") if headers else ""
    except Exception:
        user_agent = ""

    if user_agent:
        # Truncate to avoid logging huge UA strings
        metadata["user_agent"] = user_agent[:100]

        # Classify UA for easier querying
        ua_lower = user_agent.lower()
        if "android" in ua_lower or "ciris" in ua_lower:
            metadata["client_type"] = "android_app"
        elif "python" in ua_lower or "requests" in ua_lower or "httpx" in ua_lower:
            metadata["client_type"] = "python_client"
        elif "curl" in ua_lower:
            metadata["client_type"] = "curl"
        elif "go-http" in ua_lower or "golang" in ua_lower:
            metadata["client_type"] = "go_client"
        elif "scanner" in ua_lower or "bot" in ua_lower:
            metadata["client_type"] = "scanner"
        else:
            metadata["client_type"] = "other"

    # Request path (already in logs but useful for filtering)
    try:
        url = getattr(request, "url", None)
        if url and hasattr(url, "path"):
            metadata["path"] = str(url.path)[:50]
    except Exception:
        pass  # Skip path if not available

    return metadata


def _log_auth_failure(
    token: str,
    error: Exception | None,
    reason: str,
    request: Request | None = None,
) -> None:
    """
    Log an authentication failure with structured debugging context.

    SECURITY: Never logs the token itself, only format classification.

    Args:
        token: The token that failed (for format analysis only)
        error: The exception that caused the failure
        reason: Human-readable failure reason
        request: Optional request object for metadata extraction
    """
    token_info = _classify_token_format(token)
    error_info = _classify_auth_error(error) if error else {"error_type": "none", "error_category": "none"}
    request_info = _extract_request_metadata(request)

    log_data = {
        "event": "auth_failure",
        "reason": reason,
        **token_info,
        **error_info,
        **request_info,
    }

    # Log at appropriate level based on error category
    if error_info.get("error_category") == "security":
        logger.warning("auth_failure reason=%s error_type=%s token_format=%s client=%s",
                       reason, error_info.get("error_type"), token_info.get("format"),
                       request_info.get("client_type", "unknown"))
    else:
        # Most auth failures are expected (expired tokens, bots, etc.)
        logger.info("auth_failure reason=%s error_type=%s token_format=%s client=%s",
                    reason, error_info.get("error_type"), token_info.get("format"),
                    request_info.get("client_type", "unknown"))

    # Also log full structured data at debug level for detailed investigation
    logger.debug("auth_failure_detail %s", log_data)

# Test auth mode configuration
# When enabled, accepts opaque test tokens validated via CIRISBilling
CIRIS_TEST_AUTH_ENABLED = os.environ.get("CIRIS_TEST_AUTH_ENABLED", "").lower() == "true"
CIRIS_TEST_USER_ID = os.environ.get("CIRIS_TEST_USER_ID", "ciris_synthetic_canary")
CIRIS_ENV = os.environ.get("CIRIS_ENV", "").lower()
BILLING_API_URL = os.environ.get("BILLING_API_URL", "")
BILLING_API_KEY = os.environ.get("BILLING_API_KEY", "")

# Fail-fast gate (AV-4 in docs/THREAT_MODEL.md): refuse to start with test
# auth enabled in production. Mirrors CIRISBilling's `environment=production`
# gate. Either unset CIRIS_TEST_AUTH_ENABLED or set CIRIS_ENV != "production".
if CIRIS_TEST_AUTH_ENABLED and CIRIS_ENV == "production":
    raise RuntimeError(
        "FATAL: CIRIS_TEST_AUTH_ENABLED=true is not allowed when CIRIS_ENV=production. "
        "Test tokens bypass OAuth verification and must never be reachable from a "
        "production deployment. Unset CIRIS_TEST_AUTH_ENABLED or change CIRIS_ENV."
    )

# Google OAuth Client IDs - both web and Android client IDs are valid audiences
# Web client ID (used as audience for most ID tokens)
GOOGLE_CLIENT_ID_WEB = os.environ.get(
    "GOOGLE_CLIENT_ID",
    "265882853697-l421ndojcs5nm7lkln53jj29kf7kck91.apps.googleusercontent.com"
)
# Android client ID (some tokens may use this as audience)
GOOGLE_CLIENT_ID_ANDROID = os.environ.get(
    "GOOGLE_CLIENT_ID_ANDROID",
    "265882853697-vqfv6ecjgc1ku7n6bm4hllg6csdiaild.apps.googleusercontent.com"
)
# All valid Google client IDs
GOOGLE_CLIENT_IDS = [GOOGLE_CLIENT_ID_WEB, GOOGLE_CLIENT_ID_ANDROID]

# Apple Sign-In configuration
# iOS bundle ID (used as audience for Apple ID tokens)
APPLE_CLIENT_ID = os.environ.get("APPLE_CLIENT_ID", "ai.ciris.mobile")
# Additional Apple client IDs (comma-separated)
APPLE_CLIENT_IDS_EXTRA = os.environ.get("APPLE_CLIENT_IDS", "")
# All valid Apple bundle IDs
APPLE_CLIENT_IDS = [APPLE_CLIENT_ID] + [x.strip() for x in APPLE_CLIENT_IDS_EXTRA.split(",") if x.strip()]

# Apple public keys cache
_apple_public_keys: dict[str, object] = {}
_apple_keys_fetched_at: float = 0
_APPLE_KEYS_CACHE_TTL = 3600  # Refresh keys every hour

if CIRIS_TEST_AUTH_ENABLED:
    logger.critical(
        "TEST AUTH MODE ENABLED - test tokens accepted as user_id=%s (CIRIS_ENV=%s). "
        "Never enable in production.",
        CIRIS_TEST_USER_ID,
        CIRIS_ENV or "unset",
    )

# Cache for verified tokens: token -> (user_id, auth_type, cache_until_timestamp)
# This avoids re-verifying the same token on every request
# We cache for 24 hours since we don't check expiration anyway
# auth_type is "google" or "test" to distinguish token sources
_token_cache: dict[str, tuple[str, str, float]] = {}

# Maximum cache size to prevent memory issues
_MAX_CACHE_SIZE = 10000

# Cache tokens for 24 hours (signature verification is expensive)
_CACHE_DURATION_SECONDS = 86400

# HTTP client for test token validation (reused across requests)
_http_client: "httpx.AsyncClient | None" = None


def _cleanup_cache() -> None:
    """Remove old entries from the cache."""
    if len(_token_cache) < _MAX_CACHE_SIZE:
        return

    now = time.time()
    expired = [k for k, (_, _, exp) in _token_cache.items() if exp < now]
    for k in expired:
        del _token_cache[k]


def _is_test_token(token: str) -> bool:
    """
    Check if a token looks like a test token (opaque hex string) vs a JWT.

    Test tokens are hex strings like "c6d7c30dd742f4424c5a214cf5a6bd23..."
    JWTs have 3 base64 segments separated by dots.
    """
    if not token:
        return False

    # JWTs have 3 segments separated by dots
    if token.count(".") == 2:
        return False

    # Test tokens are typically 64-char hex strings, but we accept any hex-like string
    # that's at least 32 chars (16 bytes) for security
    if len(token) < 32:
        return False

    # Check if it looks like a hex string (alphanumeric, no special chars except _-)
    return token.replace("-", "").replace("_", "").isalnum()


async def _get_http_client() -> httpx.AsyncClient:
    """Get or create the shared HTTP client for test token validation."""
    global _http_client
    if _http_client is None:
        _http_client = httpx.AsyncClient(timeout=10.0)
    return _http_client


async def _validate_test_token(token: str) -> tuple[str, bool]:
    """
    Validate a test token by calling CIRISBilling's credits/check endpoint.

    The billing service validates the token and returns user info if valid.

    Args:
        token: The test token to validate

    Returns:
        Tuple of (user_id, is_valid)
    """
    if not BILLING_API_URL:
        logger.error("test_auth_error: BILLING_API_URL not configured")
        return "", False

    try:
        client = await _get_http_client()
        response = await client.post(
            f"{BILLING_API_URL}/v1/billing/credits/check",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            json={
                "oauth_provider": "oauth:test",
                "external_id": CIRIS_TEST_USER_ID,
            },
        )

        if response.status_code == 200:
            data = response.json()
            # If billing accepts the token, it has credits
            if data.get("has_credit", False):
                logger.info("test_auth_success user_id=%s", CIRIS_TEST_USER_ID[:8])
                return CIRIS_TEST_USER_ID, True
            else:
                logger.warning("test_auth_denied reason=no_credits user_id=%s", CIRIS_TEST_USER_ID[:8])
                return CIRIS_TEST_USER_ID, False
        elif response.status_code == 401:
            logger.warning("test_auth_denied reason=invalid_token")
            return "", False
        else:
            logger.error("test_auth_error status=%d", response.status_code)
            return "", False

    except httpx.TimeoutException:
        logger.error("test_auth_error reason=timeout")
        return "", False
    except Exception as e:
        logger.error("test_auth_error reason=%s", str(e)[:50])
        return "", False


def _get_cached_auth(api_key: str) -> UserAPIKeyAuth | None:
    """
    Check cache for valid token and return auth if found.

    Args:
        api_key: The token to look up

    Returns:
        UserAPIKeyAuth if cached and valid, None otherwise
    """
    if api_key not in _token_cache:
        return None

    user_id, auth_type, cache_until = _token_cache[api_key]
    if time.time() < cache_until:
        return UserAPIKeyAuth(
            api_key=f"{auth_type}:{user_id}",
            user_id=user_id,
        )

    # Cache entry expired, remove it
    del _token_cache[api_key]
    return None


def _import_google_auth():
    """
    Import Google auth libraries.

    Returns:
        Tuple of (id_token, google_requests, jwt) modules

    Raises:
        ProxyException: If google-auth is not installed
    """
    try:
        from google.oauth2 import id_token
        from google.auth.transport import requests as google_requests
        from google.auth import jwt
        return id_token, google_requests, jwt
    except ImportError:
        raise ProxyException(
            message="Server misconfiguration: google-auth not installed",
            type="server_error",
            code=500,
        )


def _is_key_error(error: Exception) -> bool:
    """Check if an error indicates a key/signature verification issue."""
    error_msg = str(error).lower()
    key_error_indicators = [
        "could not verify",
        "signature",
        "invalid token",
        "verification failed",
        "key",
        "certificate",
    ]
    # Don't treat expiration or audience errors as key errors
    if "expired" in error_msg or "audience" in error_msg:
        return False
    return any(indicator in error_msg for indicator in key_error_indicators)


def _clear_google_certs_cache(id_token_module) -> None:
    """
    Clear Google's internal certificate cache to force refresh.

    When Google rotates signing keys, the cached certificates become stale.
    This forces the library to fetch fresh certificates on the next verification.
    """
    # Google's id_token module caches certs in _GOOGLE_OAUTH2_CERTS (TTLCache)
    # This is a private attribute but necessary to force refresh on key rotation
    if hasattr(id_token_module, "_GOOGLE_OAUTH2_CERTS"):
        try:
            id_token_module._GOOGLE_OAUTH2_CERTS.clear()
            logger.info("google_certs_cache_cleared")
        except Exception as e:
            logger.warning("google_certs_cache_clear_failed error=%s", str(e)[:50])


def _try_verify_token(api_key: str, id_token_module, google_requests) -> tuple[dict | None, Exception | None]:
    """
    Try to verify token against each valid client ID.

    If verification fails with what looks like a key/signature error,
    clears Google's certificate cache and retries once. This handles
    Google key rotation without requiring a proxy restart.

    Args:
        api_key: The token to verify
        id_token_module: google.oauth2.id_token module
        google_requests: google.auth.transport.requests module

    Returns:
        Tuple of (idinfo dict if successful, last_error if failed)
    """
    idinfo = None
    last_error = None
    retried = False

    for attempt in range(2):  # Max 2 attempts (original + 1 retry after cache clear)
        for client_id in GOOGLE_CLIENT_IDS:
            try:
                idinfo = id_token_module.verify_oauth2_token(
                    api_key,
                    google_requests.Request(),
                    client_id
                )
                return idinfo, None  # Success
            except Exception as e:
                last_error = e
                error_msg = str(e).lower()
                # If token expired, stop trying other client IDs
                if "expired" in error_msg:
                    return None, last_error
                # For other errors (including audience mismatch), try next client ID
                continue

        # If we've already retried or this doesn't look like a key error, stop
        if retried or not _is_key_error(last_error):
            break

        # Clear cache and retry - Google may have rotated keys
        logger.info("google_verification_failed_retrying reason=possible_key_rotation")
        _clear_google_certs_cache(id_token_module)
        retried = True

    return None, last_error


def _validate_expired_token_claims(idinfo: dict) -> None:
    """
    Validate audience and issuer claims from an expired token.

    Args:
        idinfo: Decoded token claims

    Raises:
        ProxyException: If audience or issuer is invalid
    """
    aud = idinfo.get("aud")
    if aud not in GOOGLE_CLIENT_IDS:
        raise ProxyException(
            message="Invalid token audience",
            type="auth_error",
            param="Authorization",
            code=401,
        )

    iss = idinfo.get("iss")
    if iss not in ("accounts.google.com", "https://accounts.google.com"):
        raise ProxyException(
            message="Invalid token issuer",
            type="auth_error",
            param="Authorization",
            code=401,
        )


def _handle_expired_token(api_key: str, jwt_module, last_error: Exception) -> dict:
    """
    Handle an expired token by decoding without verification.

    Args:
        api_key: The expired token
        jwt_module: google.auth.jwt module
        last_error: The expiration error from verification

    Returns:
        Decoded token claims if valid

    Raises:
        ProxyException: If token cannot be decoded or claims are invalid
        Exception: Re-raises last_error if not an expiration error
    """
    error_msg = str(last_error).lower()
    if "expired" not in error_msg:
        raise last_error

    try:
        idinfo = jwt_module.decode(api_key, verify=False)
        _validate_expired_token_claims(idinfo)
        return idinfo
    except ProxyException:
        raise
    except Exception as decode_error:
        raise ProxyException(
            message=f"Failed to decode expired token: {decode_error}",
            type="auth_error",
            param="Authorization",
            code=401,
        )


def _extract_user_id(idinfo: dict) -> str:
    """
    Extract user ID from token claims.

    Args:
        idinfo: Decoded token claims

    Returns:
        The user ID (sub claim)

    Raises:
        ProxyException: If user ID is missing
    """
    user_id = idinfo.get("sub")
    if not user_id:
        raise ProxyException(
            message="Invalid token: missing user ID",
            type="auth_error",
            param="Authorization",
            code=401,
        )
    return user_id


def _cache_token(api_key: str, user_id: str, auth_type: str = "google") -> None:
    """
    Cache a verified token.

    Args:
        api_key: The token to cache
        user_id: The extracted user ID
        auth_type: The auth type ("google" or "test")
    """
    cache_until = time.time() + _CACHE_DURATION_SECONDS
    _cleanup_cache()
    _token_cache[api_key] = (user_id, auth_type, cache_until)


def _handle_verification_error(error: Exception) -> None:
    """
    Convert a verification error to an appropriate ProxyException.

    Args:
        error: The verification error

    Raises:
        ProxyException: Always raises with appropriate message
    """
    error_msg = str(error).lower()

    if "audience" in error_msg:
        raise ProxyException(
            message="Invalid token audience. Please use the correct app.",
            type="auth_error",
            param="Authorization",
            code=401,
        )
    elif "expired" in error_msg:
        raise ProxyException(
            message="Token has expired. Please re-authenticate.",
            type="auth_error",
            param="Authorization",
            code=401,
        )
    else:
        raise ProxyException(
            message="Invalid authentication token",
            type="auth_error",
            param="Authorization",
            code=401,
        )


# Helper functions for verify_google_token (return None on error, never raise)


def _get_cached_idinfo(token: str) -> dict | None:
    """
    Check cache for valid token and return idinfo if found.

    Args:
        token: The token to look up

    Returns:
        dict with 'sub' key if cached and valid, None otherwise
    """
    if token not in _token_cache:
        return None

    user_id, auth_type, cache_until = _token_cache[token]
    if time.time() < cache_until:
        return {"sub": user_id, "_auth_type": auth_type}

    # Cache entry expired, remove it
    del _token_cache[token]
    return None


def _try_import_google_auth_silent():
    """
    Import Google auth libraries silently.

    Returns:
        Tuple of (id_token, google_requests, jwt) or None if not installed
    """
    try:
        from google.oauth2 import id_token
        from google.auth.transport import requests as google_requests
        from google.auth import jwt
        return id_token, google_requests, jwt
    except ImportError:
        return None


def _try_decode_expired_token_silent(token: str, jwt_module, last_error: Exception | None) -> dict | None:
    """
    Try to decode an expired token without verification.

    Args:
        token: The token to decode
        jwt_module: google.auth.jwt module
        last_error: The error from verification (to check if expired)

    Returns:
        dict with claims if valid expired token, None otherwise
    """
    if not last_error or "expired" not in str(last_error).lower():
        return None

    try:
        unverified = jwt_module.decode(token, verify=False)
        aud = unverified.get("aud")
        iss = unverified.get("iss")
        if aud in GOOGLE_CLIENT_IDS and iss in ("accounts.google.com", "https://accounts.google.com"):
            return unverified
    except Exception:
        pass
    return None


# ============================================================================
# Apple Sign-In Token Verification
# ============================================================================


async def _fetch_apple_public_keys() -> None:
    """Fetch and cache Apple's public keys for JWT verification."""
    global _apple_keys_fetched_at

    now = time.time()
    if _apple_public_keys and (now - _apple_keys_fetched_at) < _APPLE_KEYS_CACHE_TTL:
        return  # Keys are still fresh

    try:
        client = await _get_http_client()
        response = await client.get(
            "https://appleid.apple.com/auth/keys",
            timeout=10.0,
        )
        response.raise_for_status()
        keys_data = response.json()

        # Import jwt algorithms for JWK parsing
        from jwt import algorithms

        # Parse JWK keys into RSA public keys
        _apple_public_keys.clear()
        for key_dict in keys_data.get("keys", []):
            kid = key_dict.get("kid")
            if kid:
                # Convert JWK to RSA public key
                public_key = algorithms.RSAAlgorithm.from_jwk(key_dict)
                _apple_public_keys[kid] = public_key

        _apple_keys_fetched_at = now
        logger.info("apple_public_keys_fetched key_count=%d", len(_apple_public_keys))

    except Exception as e:
        logger.error("apple_public_keys_fetch_failed error=%s", str(e)[:50])
        # Don't clear existing keys on error - use stale keys if available
        if not _apple_public_keys:
            raise


def _is_apple_token(token: str) -> bool:
    """
    Check if a token looks like an Apple ID token by inspecting the issuer.

    Apple ID tokens have issuer "https://appleid.apple.com".
    """
    if not token or token.count(".") != 2:
        return False

    try:
        import jwt
        unverified = jwt.decode(token, options={"verify_signature": False})
        return unverified.get("iss") == "https://appleid.apple.com"
    except Exception:
        return False


async def _try_verify_apple_token(token: str) -> tuple[dict | None, Exception | None]:
    """
    Try to verify an Apple ID token.

    Args:
        token: The Apple ID token to verify

    Returns:
        Tuple of (idinfo dict if successful, last_error if failed)
    """
    import jwt

    # Fetch Apple's public keys
    try:
        await _fetch_apple_public_keys()
    except Exception as e:
        return None, e

    if not _apple_public_keys:
        return None, ValueError("No Apple public keys available")

    # Decode JWT header to get key ID (kid)
    try:
        unverified_header = jwt.get_unverified_header(token)
        kid = unverified_header.get("kid")
    except Exception as e:
        return None, e

    if not kid:
        return None, ValueError("Token missing key ID (kid)")

    # Find matching public key
    public_key = _apple_public_keys.get(kid)
    if not public_key:
        # Keys might have rotated, try refreshing
        global _apple_keys_fetched_at
        _apple_keys_fetched_at = 0  # Force refresh
        try:
            await _fetch_apple_public_keys()
            public_key = _apple_public_keys.get(kid)
        except Exception:
            pass

        if not public_key:
            return None, ValueError(f"Unknown key ID: {kid}")

    # Try each bundle ID until one works
    last_error = None
    for bundle_id in APPLE_CLIENT_IDS:
        try:
            payload = jwt.decode(
                token,
                public_key,
                algorithms=["RS256"],
                audience=bundle_id,
                issuer="https://appleid.apple.com",
            )
            return payload, None  # Success
        except jwt.exceptions.InvalidAudienceError:
            last_error = ValueError(f"Invalid audience for bundle ID {bundle_id}")
            continue
        except jwt.exceptions.ExpiredSignatureError as e:
            last_error = e
            break  # Token expired, try to decode without verification
        except Exception as e:
            last_error = e
            break

    return None, last_error


def _try_decode_expired_apple_token_silent(token: str, last_error: Exception | None) -> dict | None:
    """
    Try to decode an expired Apple token without verification.

    Args:
        token: The token to decode
        last_error: The error from verification (to check if expired)

    Returns:
        dict with claims if valid expired token, None otherwise
    """
    if not last_error or "expired" not in str(last_error).lower():
        return None

    try:
        import jwt
        unverified = jwt.decode(token, options={"verify_signature": False})
        aud = unverified.get("aud")
        iss = unverified.get("iss")
        if aud in APPLE_CLIENT_IDS and iss == "https://appleid.apple.com":
            return unverified
    except Exception:
        pass
    return None


async def verify_token(token: str) -> dict | None:
    """
    Verify an authentication token and return user info.

    Supports Google ID tokens, Apple ID tokens, and test tokens.
    This is a reusable function for endpoints that need auth but aren't
    using LiteLLM's auth middleware (e.g., /v1/web/search).

    Args:
        token: Google/Apple ID token (JWT) or test token (opaque string)

    Returns:
        dict with 'sub' (user ID) and '_auth_type' ("google", "apple", or "test"),
        or None if invalid
    """
    if not token:
        return None

    # Check cache first
    cached = _get_cached_idinfo(token)
    if cached:
        return cached

    # Test auth mode: validate opaque tokens via CIRISBilling
    if CIRIS_TEST_AUTH_ENABLED and _is_test_token(token):
        user_id, is_valid = await _validate_test_token(token)
        if is_valid:
            _cache_token(token, user_id, "test")
            return {"sub": user_id, "_auth_type": "test"}
        return None

    # Check if this is an Apple token
    if _is_apple_token(token):
        try:
            idinfo, last_error = await _try_verify_apple_token(token)

            # Handle expired tokens (still accept them)
            if idinfo is None:
                idinfo = _try_decode_expired_apple_token_silent(token, last_error)

            if idinfo is None:
                return None

            # Validate user ID exists
            user_id = idinfo.get("sub")
            if not user_id:
                return None

            # Cache and return
            _cache_token(token, user_id, "apple")
            idinfo["_auth_type"] = "apple"
            return idinfo

        except Exception:
            return None

    # Google OAuth: validate JWT tokens
    modules = _try_import_google_auth_silent()
    if not modules:
        return None
    id_token_module, google_requests, jwt_module = modules

    try:
        # Try verification with each client ID
        idinfo, last_error = _try_verify_token(token, id_token_module, google_requests)

        # Handle expired tokens (still accept them)
        if idinfo is None:
            idinfo = _try_decode_expired_token_silent(token, jwt_module, last_error)

        if idinfo is None:
            return None

        # Validate user ID exists
        user_id = idinfo.get("sub")
        if not user_id:
            return None

        # Cache and return
        _cache_token(token, user_id, "google")
        idinfo["_auth_type"] = "google"
        return idinfo

    except Exception:
        return None


# Alias for backwards compatibility
async def verify_google_token(token: str) -> dict | None:
    """Deprecated: Use verify_token instead."""
    return await verify_token(token)


async def user_api_key_auth(request: Request, api_key: str) -> UserAPIKeyAuth | str:
    """
    Verify authentication token and extract user identity.

    Supports three auth modes:
    1. Google OAuth: Validates Google ID tokens (JWTs)
    2. Apple Sign-In: Validates Apple ID tokens (JWTs)
    3. Test mode: Validates opaque test tokens via CIRISBilling

    Token type is auto-detected based on the JWT issuer claim.

    Args:
        request: The incoming FastAPI request object
        api_key: The token from the Authorization header (after "Bearer ")

    Returns:
        UserAPIKeyAuth object with api_key="{provider}:{user_id}"

    Raises:
        ProxyException: If token is missing, invalid, or verification fails
    """
    if not api_key:
        _log_auth_failure(
            token="",
            error=None,
            reason="missing_token",
            request=request,
        )
        raise ProxyException(
            message="Missing authorization token",
            type="auth_error",
            param="Authorization",
            code=401,
        )

    # Check cache first (avoids network call to Google/Apple or billing)
    cached_auth = _get_cached_auth(api_key)
    if cached_auth:
        return cached_auth

    # Test auth mode: validate opaque tokens via CIRISBilling
    if CIRIS_TEST_AUTH_ENABLED and _is_test_token(api_key):
        user_id, is_valid = await _validate_test_token(api_key)
        if is_valid:
            _cache_token(api_key, user_id, "test")
            return UserAPIKeyAuth(
                api_key=f"test:{user_id}",
                user_id=user_id,
            )
        else:
            _log_auth_failure(
                token=api_key,
                error=None,
                reason="test_token_invalid",
                request=request,
            )
            raise ProxyException(
                message="Invalid test authentication token",
                type="auth_error",
                param="Authorization",
                code=401,
            )

    # Check if this is an Apple token
    if _is_apple_token(api_key):
        try:
            idinfo, last_error = await _try_verify_apple_token(api_key)

            # If verification failed, try handling as expired token
            if idinfo is None and last_error:
                idinfo = _try_decode_expired_apple_token_silent(api_key, last_error)
                if idinfo is None:
                    _log_auth_failure(
                        token=api_key,
                        error=last_error,
                        reason="apple_verification_failed",
                        request=request,
                    )
                    _handle_verification_error(last_error)

            # Extract and validate user ID
            user_id = _extract_user_id(idinfo)

            # Cache the verified token
            _cache_token(api_key, user_id, "apple")

            # Return with apple|{user_id} format for billing callback compatibility
            # NOTE: Using pipe delimiter instead of colon because Apple user IDs
            # have format like "001234.abc.xyz" (3 dot-separated parts), which
            # triggers LiteLLM's naive JWT detection (is_jwt checks for 3 dots).
            # "apple:001234.abc.xyz" gets hashed as a JWT, breaking billing.
            return UserAPIKeyAuth(
                api_key=f"apple|{user_id}",
                user_id=user_id,
            )

        except ProxyException as e:
            _log_auth_failure(
                token=api_key,
                error=e,
                reason=e.message[:50] if hasattr(e, "message") else "proxy_exception",
                request=request,
            )
            raise
        except Exception as e:
            _log_auth_failure(
                token=api_key,
                error=e,
                reason="apple_verification_failed",
                request=request,
            )
            _handle_verification_error(e)

    # Google OAuth: validate JWT tokens
    id_token_module, google_requests, jwt_module = _import_google_auth()

    try:
        # Try verification with each valid client ID
        idinfo, last_error = _try_verify_token(api_key, id_token_module, google_requests)

        # If verification failed, try handling as expired token
        if idinfo is None and last_error:
            idinfo = _handle_expired_token(api_key, jwt_module, last_error)

        # Extract and validate user ID
        user_id = _extract_user_id(idinfo)

        # Cache the verified token
        _cache_token(api_key, user_id, "google")

        # Return with google:{user_id} format for billing callback compatibility
        return UserAPIKeyAuth(
            api_key=f"google:{user_id}",
            user_id=user_id,
        )

    except ProxyException as e:
        # Log all ProxyException auth failures for debugging
        _log_auth_failure(
            token=api_key,
            error=e,
            reason=e.message[:50] if hasattr(e, "message") else "proxy_exception",
            request=request,
        )
        raise
    except Exception as e:
        # Log unexpected errors before converting to ProxyException
        _log_auth_failure(
            token=api_key,
            error=e,
            reason="verification_failed",
            request=request,
        )
        _handle_verification_error(e)
