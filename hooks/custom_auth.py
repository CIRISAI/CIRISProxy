"""
Google ID Token verification for LiteLLM Proxy.

Accepts: Authorization: Bearer {google_id_token}
Verifies: Token signature against Google's public keys
Extracts: User's Google ID (sub claim) for billing

This replaces the simple numeric ID validation with proper JWT verification.
The Google ID token is cryptographically signed by Google and cannot be forged.

NOTE: Expiration checking is DISABLED. The Android app may send tokens that
expired hours ago. We still verify the signature (token was issued by Google)
and audience (token was issued for our app). The user ID is stable and the
billing service handles authorization.

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
# All valid client IDs
GOOGLE_CLIENT_IDS = [GOOGLE_CLIENT_ID_WEB, GOOGLE_CLIENT_ID_ANDROID]

# Cache for verified tokens: token -> (user_id, cache_until_timestamp)
# This avoids re-verifying the same token on every request
# We cache for 24 hours since we don't check expiration anyway
_token_cache: dict[str, tuple[str, float]] = {}

# Maximum cache size to prevent memory issues
_MAX_CACHE_SIZE = 10000

# Cache tokens for 24 hours (signature verification is expensive)
_CACHE_DURATION_SECONDS = 86400


def _cleanup_cache() -> None:
    """Remove old entries from the cache."""
    if len(_token_cache) < _MAX_CACHE_SIZE:
        return

    now = time.time()
    expired = [k for k, (_, exp) in _token_cache.items() if exp < now]
    for k in expired:
        del _token_cache[k]


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

    user_id, cache_until = _token_cache[api_key]
    if time.time() < cache_until:
        return UserAPIKeyAuth(
            api_key=f"google:{user_id}",
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


def _try_verify_token(api_key: str, id_token_module, google_requests) -> tuple[dict | None, Exception | None]:
    """
    Try to verify token against each valid client ID.

    Args:
        api_key: The token to verify
        id_token_module: google.oauth2.id_token module
        google_requests: google.auth.transport.requests module

    Returns:
        Tuple of (idinfo dict if successful, last_error if failed)
    """
    idinfo = None
    last_error = None

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
                break
            # For other errors (including audience mismatch), try next client ID
            continue

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


def _cache_token(api_key: str, user_id: str) -> None:
    """
    Cache a verified token.

    Args:
        api_key: The token to cache
        user_id: The extracted user ID
    """
    cache_until = time.time() + _CACHE_DURATION_SECONDS
    _cleanup_cache()
    _token_cache[api_key] = (user_id, cache_until)


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

    user_id, cache_until = _token_cache[token]
    if time.time() < cache_until:
        return {"sub": user_id}

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


async def verify_google_token(token: str) -> dict | None:
    """
    Verify a Google ID token and return user info.

    This is a reusable function for endpoints that need Google auth
    but aren't using LiteLLM's auth middleware (e.g., /v1/web/search).

    Args:
        token: Google ID token (JWT)

    Returns:
        dict with 'sub' (user ID) and other claims, or None if invalid
    """
    if not token:
        return None

    # Check cache first
    cached = _get_cached_idinfo(token)
    if cached:
        return cached

    # Import google auth libraries
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
        _cache_token(token, user_id)
        return idinfo

    except Exception:
        return None


async def user_api_key_auth(request: Request, api_key: str) -> UserAPIKeyAuth | str:
    """
    Verify Google ID token and extract user identity.

    The api_key parameter is actually a Google ID token (JWT).
    We verify it against Google's public keys and extract the user ID.

    Args:
        request: The incoming FastAPI request object
        api_key: The Google ID token from the Authorization header (after "Bearer ")

    Returns:
        UserAPIKeyAuth object with api_key="google:{user_id}"

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

    # Check cache first (avoids network call to Google)
    cached_auth = _get_cached_auth(api_key)
    if cached_auth:
        return cached_auth

    # Import Google auth libraries
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
        _cache_token(api_key, user_id)

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
