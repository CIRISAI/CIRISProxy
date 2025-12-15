"""
Status Endpoint for CIRISProxy

Exposes /v1/status with health checks for all providers:
- LLM Providers: OpenRouter, Groq, Together AI
- Billing: CIRISBilling API
- Search: Brave Search API

Returns standardized status response for CIRISLens aggregation.
"""

import asyncio
import os
import time
from datetime import datetime, timezone
from typing import Any

import httpx

# Cache for rate limiting - prevents hammering upstream providers
_status_cache: dict[str, Any] = {}
_cache_timestamp: float = 0
CACHE_TTL_SECONDS = 10  # Cache status for 10 seconds

# Status levels
STATUS_OPERATIONAL = "operational"
STATUS_DEGRADED = "degraded"
STATUS_OUTAGE = "outage"

# Latency thresholds (ms)
LATENCY_GOOD = 1000
LATENCY_DEGRADED = 3000

# Auth prefix constants
AUTH_PREFIX_BEARER = "Bearer "

# Provider configurations
PROVIDERS = {
    "openrouter": {
        "name": "OpenRouter",
        "type": "llm",
        "check_url": "https://openrouter.ai/api/v1/models",
        "env_key": "OPENROUTER_API_KEY",
        "auth_header": "Authorization",
        "auth_prefix": AUTH_PREFIX_BEARER,
    },
    "groq": {
        "name": "Groq",
        "type": "llm",
        "check_url": "https://api.groq.com/openai/v1/models",
        "env_key": "GROQ_API_KEY",
        "auth_header": "Authorization",
        "auth_prefix": AUTH_PREFIX_BEARER,
    },
    "together": {
        "name": "Together AI",
        "type": "llm",
        "check_url": "https://api.together.xyz/v1/models",
        "env_key": "TOGETHER_API_KEY",
        "auth_header": "Authorization",
        "auth_prefix": AUTH_PREFIX_BEARER,
    },
    "billing": {
        "name": "CIRISBilling",
        "type": "internal",
        "check_url": None,  # Dynamic from env
        "env_key": "BILLING_API_KEY",
        "env_url": "BILLING_API_URL",
        "check_path": "/health",
    },
    "brave": {
        "name": "Brave Search",
        "type": "search",
        "check_url": "https://api.search.brave.com/res/v1/web/search",
        "check_params": {"q": "test", "count": 1},
        "env_key": "BRAVE_API_KEY",
        "auth_header": "X-Subscription-Token",
        "auth_prefix": "",
    },
}


def _init_provider_result(provider_id: str, config: dict) -> dict[str, Any]:
    """Initialize a provider status result with default values."""
    return {
        "provider": provider_id,
        "name": config["name"],
        "type": config["type"],
        "status": STATUS_OUTAGE,
        "latency_ms": None,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "error": None,
    }


def _get_api_key(config: dict) -> str | None:
    """Get API key from environment, or None if not configured."""
    api_key = os.environ.get(config["env_key"], "")
    return api_key if api_key else None


def _build_check_url(config: dict) -> str | None:
    """Build the check URL from config. Returns None if URL cannot be built."""
    if config.get("env_url"):
        base_url = os.environ.get(config["env_url"], "")
        if not base_url:
            return None
        return f"{base_url.rstrip('/')}{config.get('check_path', '/health')}"
    return config["check_url"]


def _build_auth_headers(config: dict, api_key: str) -> dict[str, str]:
    """Build authentication headers from config."""
    headers = {}
    if config.get("auth_header"):
        headers[config["auth_header"]] = f"{config.get('auth_prefix', '')}{api_key}"
    return headers


def _evaluate_response(result: dict, latency_ms: int, status_code: int) -> None:
    """Evaluate response and update result status."""
    result["latency_ms"] = latency_ms
    if status_code < 400:
        if latency_ms < LATENCY_GOOD:
            result["status"] = STATUS_OPERATIONAL
        elif latency_ms < LATENCY_DEGRADED:
            result["status"] = STATUS_DEGRADED
        else:
            result["status"] = STATUS_DEGRADED
            result["error"] = f"High latency: {latency_ms}ms"
    else:
        result["status"] = STATUS_OUTAGE
        result["error"] = f"HTTP {status_code}"


async def check_provider(
    client: httpx.AsyncClient,
    provider_id: str,
    config: dict,
) -> dict[str, Any]:
    """
    Check a single provider's health.

    Returns:
        {
            "provider": "openrouter",
            "name": "OpenRouter",
            "type": "llm",
            "status": "operational|degraded|outage",
            "latency_ms": 123,
            "checked_at": "2025-01-01T00:00:00Z",
            "error": null
        }
    """
    result = _init_provider_result(provider_id, config)

    # Validate configuration
    api_key = _get_api_key(config)
    if not api_key:
        result["error"] = "API key not configured"
        return result

    check_url = _build_check_url(config)
    if not check_url:
        result["error"] = "URL not configured"
        return result

    headers = _build_auth_headers(config, api_key)

    # Make request and evaluate response
    start_time = time.monotonic()
    try:
        resp = await client.get(
            check_url,
            params=config.get("check_params"),
            headers=headers,
            timeout=10.0,
        )
        latency_ms = int((time.monotonic() - start_time) * 1000)
        _evaluate_response(result, latency_ms, resp.status_code)

    except httpx.TimeoutException:
        result["latency_ms"] = int((time.monotonic() - start_time) * 1000)
        result["error"] = "Timeout"
    except httpx.RequestError as e:
        result["latency_ms"] = int((time.monotonic() - start_time) * 1000)
        result["error"] = f"Connection error: {type(e).__name__}"
    except Exception:
        result["latency_ms"] = int((time.monotonic() - start_time) * 1000)
        result["error"] = "Internal error"

    return result


async def get_status() -> dict[str, Any]:
    """
    Get status for all providers.

    Returns cached response if within TTL to prevent upstream API abuse.

    Returns:
        {
            "service": "cirisproxy",
            "status": "operational|degraded|outage",
            "checked_at": "2025-01-01T00:00:00Z",
            "providers": [...]
        }
    """
    global _status_cache, _cache_timestamp

    # Return cached response if still valid (rate limiting)
    now = time.monotonic()
    if _status_cache and (now - _cache_timestamp) < CACHE_TTL_SECONDS:
        return _status_cache

    async with httpx.AsyncClient() as client:
        # Check all providers concurrently
        tasks = [
            check_provider(client, provider_id, config)
            for provider_id, config in PROVIDERS.items()
        ]
        results = await asyncio.gather(*tasks)

    # Determine overall status
    statuses = [r["status"] for r in results]
    if all(s == STATUS_OPERATIONAL for s in statuses):
        overall_status = STATUS_OPERATIONAL
    elif any(s == STATUS_OUTAGE for s in statuses):
        # Check if critical providers are down
        critical = ["openrouter", "groq", "together", "billing"]
        critical_down = any(
            r["status"] == STATUS_OUTAGE and r["provider"] in critical
            for r in results
        )
        overall_status = STATUS_OUTAGE if critical_down else STATUS_DEGRADED
    else:
        overall_status = STATUS_DEGRADED

    # Cache the result
    _status_cache = {
        "service": "cirisproxy",
        "version": os.environ.get("CIRISPROXY_VERSION", "unknown"),
        "status": overall_status,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "providers": results,
    }
    _cache_timestamp = now

    return _status_cache


# Sync wrapper for non-async contexts
def get_status_sync() -> dict[str, Any]:
    """Synchronous wrapper for get_status."""
    return asyncio.run(get_status())
