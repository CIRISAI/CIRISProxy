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
"""

import os
import time
from typing import Union

from fastapi import Request
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.proxy_server import ProxyException

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
    if token in _token_cache:
        user_id, cache_until = _token_cache[token]
        if time.time() < cache_until:
            return {"sub": user_id}
        else:
            del _token_cache[token]

    try:
        from google.oauth2 import id_token
        from google.auth.transport import requests as google_requests
        from google.auth import jwt
    except ImportError:
        return None

    try:
        idinfo = None
        last_error = None

        for client_id in GOOGLE_CLIENT_IDS:
            try:
                idinfo = id_token.verify_oauth2_token(
                    token,
                    google_requests.Request(),
                    client_id
                )
                break
            except Exception as e:
                last_error = e
                error_msg = str(e)
                if "expired" in error_msg.lower():
                    break
                continue

        # Handle expired tokens (still accept them)
        if idinfo is None and last_error and "expired" in str(last_error).lower():
            try:
                unverified = jwt.decode(token, verify=False)
                aud = unverified.get("aud")
                iss = unverified.get("iss")
                if aud in GOOGLE_CLIENT_IDS and iss in ("accounts.google.com", "https://accounts.google.com"):
                    idinfo = unverified
            except Exception:
                return None

        if idinfo is None:
            return None

        user_id = idinfo.get("sub")
        if not user_id:
            return None

        # Cache the result
        cache_until = time.time() + _CACHE_DURATION_SECONDS
        _cleanup_cache()
        _token_cache[token] = (user_id, cache_until)

        return idinfo

    except Exception:
        return None


async def user_api_key_auth(request: Request, api_key: str) -> Union[UserAPIKeyAuth, str]:
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
        raise ProxyException(
            message="Missing authorization token",
            type="auth_error",
            param="Authorization",
            code=401,
        )

    # Check cache first (avoids network call to Google)
    if api_key in _token_cache:
        user_id, cache_until = _token_cache[api_key]
        if time.time() < cache_until:
            return UserAPIKeyAuth(
                api_key=f"google:{user_id}",
                user_id=user_id,
            )
        else:
            # Cache entry expired, remove it
            del _token_cache[api_key]

    # Import here to avoid startup delay if google-auth isn't installed
    try:
        from google.oauth2 import id_token
        from google.auth.transport import requests as google_requests
        from google.auth import jwt
    except ImportError:
        raise ProxyException(
            message="Server misconfiguration: google-auth not installed",
            type="server_error",
            code=500,
        )

    try:
        # Try verification with each valid client ID
        idinfo = None
        last_error = None

        for client_id in GOOGLE_CLIENT_IDS:
            try:
                idinfo = id_token.verify_oauth2_token(
                    api_key,
                    google_requests.Request(),
                    client_id
                )
                break  # Success, stop trying
            except Exception as e:
                last_error = e
                error_msg = str(e)
                # If it's an audience mismatch, try next client ID
                if "audience" in error_msg.lower():
                    continue
                # If token expired, handle separately
                if "expired" in error_msg.lower():
                    break  # Will handle below
                # Other errors, try next client ID
                continue

        # If no valid verification succeeded, check for expired token
        if idinfo is None and last_error:
            error_msg = str(last_error)
            if "expired" in error_msg.lower():
                # Decode without verification to get claims
                try:
                    # Decode the token without verification to extract claims
                    unverified = jwt.decode(api_key, verify=False)

                    # Extract claims from unverified token
                    idinfo = unverified

                    # Manually verify audience against all valid client IDs
                    aud = idinfo.get("aud")
                    if aud not in GOOGLE_CLIENT_IDS:
                        raise ProxyException(
                            message=f"Invalid token audience",
                            type="auth_error",
                            param="Authorization",
                            code=401,
                        )

                    # Verify issuer
                    iss = idinfo.get("iss")
                    if iss not in ("accounts.google.com", "https://accounts.google.com"):
                        raise ProxyException(
                            message="Invalid token issuer",
                            type="auth_error",
                            param="Authorization",
                            code=401,
                        )

                except ProxyException:
                    raise
                except Exception as decode_error:
                    raise ProxyException(
                        message=f"Failed to decode expired token: {decode_error}",
                        type="auth_error",
                        param="Authorization",
                        code=401,
                    )
            else:
                # Re-raise last error if not expiration
                raise last_error

        # Extract user ID from the 'sub' (subject) claim
        # This is the stable Google user ID that never changes
        user_id = idinfo.get("sub")

        if not user_id:
            raise ProxyException(
                message="Invalid token: missing user ID",
                type="auth_error",
                param="Authorization",
                code=401,
            )

        # Cache the verified token for 24 hours (we don't check expiration)
        cache_until = time.time() + _CACHE_DURATION_SECONDS
        _cleanup_cache()
        _token_cache[api_key] = (user_id, cache_until)

        # Return with google:{user_id} format for billing callback compatibility
        return UserAPIKeyAuth(
            api_key=f"google:{user_id}",
            user_id=user_id,
        )

    except ProxyException:
        # Re-raise our own exceptions
        raise
    except Exception as e:
        # Token verification failed (invalid signature, wrong audience, etc.)
        error_msg = str(e)

        if "audience" in error_msg.lower():
            raise ProxyException(
                message="Invalid token audience. Please use the correct app.",
                type="auth_error",
                param="Authorization",
                code=401,
            )
        else:
            raise ProxyException(
                message=f"Invalid Google ID token: {error_msg}",
                type="auth_error",
                param="Authorization",
                code=401,
            )
