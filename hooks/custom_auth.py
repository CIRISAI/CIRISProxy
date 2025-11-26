"""
Custom authentication handler for LiteLLM Proxy.

This bypasses LiteLLM's database-based key validation and passes the raw API key
through to our CIRISBilling callback, which handles the actual authentication.

The API key format expected: "google:{user_id}" or just "{user_id}"

SECURITY: This handler performs basic validation but real auth happens in billing callback.
"""

import re
from typing import Union

from fastapi import Request
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.proxy_server import ProxyException

# Valid API key pattern: google:{numeric_id} or just numeric ID
# Google user IDs are numeric strings, typically 21 digits
_VALID_KEY_PATTERN = re.compile(r"^(google:)?[0-9]{10,30}$")

# Maximum API key length to prevent abuse
_MAX_KEY_LENGTH = 100


async def user_api_key_auth(request: Request, api_key: str) -> Union[UserAPIKeyAuth, str]:
    """
    Custom auth function that validates API key format and passes it through.

    Actual authorization (credit check) happens in the billing callback's
    async_pre_call_hook. This function validates format and prevents abuse.

    Args:
        request: The incoming FastAPI request object
        api_key: The API key from the Authorization header (after "Bearer ")

    Returns:
        UserAPIKeyAuth object with the raw API key preserved

    Raises:
        ProxyException: If API key format is invalid
    """
    # Reject empty keys
    if not api_key:
        raise ProxyException(
            message="Missing API key",
            type="auth_error",
            param="Authorization",
            code=401,
        )

    # Reject overly long keys (DoS protection)
    if len(api_key) > _MAX_KEY_LENGTH:
        raise ProxyException(
            message="Invalid API key format",
            type="auth_error",
            param="Authorization",
            code=401,
        )

    # Validate key format
    if not _VALID_KEY_PATTERN.match(api_key):
        raise ProxyException(
            message="Invalid API key format. Expected: google:{user_id}",
            type="auth_error",
            param="Authorization",
            code=401,
        )

    # Return UserAPIKeyAuth with the raw key - billing callback will verify credits
    return UserAPIKeyAuth(
        api_key=api_key,
        user_id="user",  # Generic - don't expose actual ID in logs
    )
