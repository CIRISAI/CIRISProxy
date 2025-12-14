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

# Status levels
STATUS_OPERATIONAL = "operational"
STATUS_DEGRADED = "degraded"
STATUS_OUTAGE = "outage"

# Latency thresholds (ms)
LATENCY_GOOD = 1000
LATENCY_DEGRADED = 3000

# Provider configurations
PROVIDERS = {
    "openrouter": {
        "name": "OpenRouter",
        "type": "llm",
        "check_url": "https://openrouter.ai/api/v1/models",
        "env_key": "OPENROUTER_API_KEY",
        "auth_header": "Authorization",
        "auth_prefix": "Bearer ",
    },
    "groq": {
        "name": "Groq",
        "type": "llm",
        "check_url": "https://api.groq.com/openai/v1/models",
        "env_key": "GROQ_API_KEY",
        "auth_header": "Authorization",
        "auth_prefix": "Bearer ",
    },
    "together": {
        "name": "Together AI",
        "type": "llm",
        "check_url": "https://api.together.xyz/v1/models",
        "env_key": "TOGETHER_API_KEY",
        "auth_header": "Authorization",
        "auth_prefix": "Bearer ",
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
    result = {
        "provider": provider_id,
        "name": config["name"],
        "type": config["type"],
        "status": STATUS_OUTAGE,
        "latency_ms": None,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "error": None,
    }

    # Get API key
    api_key = os.environ.get(config["env_key"], "")
    if not api_key:
        result["error"] = "API key not configured"
        return result

    # Build check URL
    if config.get("env_url"):
        base_url = os.environ.get(config["env_url"], "")
        if not base_url:
            result["error"] = "URL not configured"
            return result
        check_url = f"{base_url.rstrip('/')}{config.get('check_path', '/health')}"
    else:
        check_url = config["check_url"]

    # Build headers
    headers = {}
    if config.get("auth_header"):
        headers[config["auth_header"]] = f"{config.get('auth_prefix', '')}{api_key}"

    # Make request
    start_time = time.monotonic()
    try:
        if config.get("check_params"):
            resp = await client.get(
                check_url,
                params=config["check_params"],
                headers=headers,
                timeout=10.0,
            )
        else:
            resp = await client.get(check_url, headers=headers, timeout=10.0)

        latency_ms = int((time.monotonic() - start_time) * 1000)
        result["latency_ms"] = latency_ms

        if resp.status_code < 400:
            if latency_ms < LATENCY_GOOD:
                result["status"] = STATUS_OPERATIONAL
            elif latency_ms < LATENCY_DEGRADED:
                result["status"] = STATUS_DEGRADED
            else:
                result["status"] = STATUS_DEGRADED
                result["error"] = f"High latency: {latency_ms}ms"
        else:
            result["status"] = STATUS_OUTAGE
            result["error"] = f"HTTP {resp.status_code}"

    except httpx.TimeoutException:
        result["latency_ms"] = int((time.monotonic() - start_time) * 1000)
        result["error"] = "Timeout"
    except httpx.RequestError as e:
        result["latency_ms"] = int((time.monotonic() - start_time) * 1000)
        result["error"] = f"Connection error: {type(e).__name__}"
    except Exception as e:
        result["latency_ms"] = int((time.monotonic() - start_time) * 1000)
        result["error"] = str(e)[:100]

    return result


async def get_status() -> dict[str, Any]:
    """
    Get status for all providers.

    Returns:
        {
            "service": "cirisproxy",
            "status": "operational|degraded|outage",
            "checked_at": "2025-01-01T00:00:00Z",
            "providers": [...]
        }
    """
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

    return {
        "service": "cirisproxy",
        "version": os.environ.get("CIRISPROXY_VERSION", "unknown"),
        "status": overall_status,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "providers": results,
    }


# Sync wrapper for non-async contexts
def get_status_sync() -> dict[str, Any]:
    """Synchronous wrapper for get_status."""
    return asyncio.run(get_status())
