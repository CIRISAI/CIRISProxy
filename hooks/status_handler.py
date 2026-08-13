"""
Status Endpoint for CIRISProxy

Exposes /v1/status with health checks for all providers:
- LLM Providers: DeepInfra (primary), OpenRouter, Groq, Together AI
- Billing: CIRISBilling API

Search providers are deliberately NOT checked here — see the note above
PROVIDERS.

Returns standardized status response for the ciris-status aggregator.

Pooled vs non-pooled
--------------------
The LLM providers are a *pool*: the router has alternatives, so one member
being slow or down is routed around and is not service impairment. Only
dependencies nothing else can serve — billing — degrade this service's own
`status`. A pool contributes to `status` only when it drops below
`min_available`, i.e. when there is genuinely nothing left to route to.

Every provider's own verdict is still reported verbatim in `providers[]`;
the pool rollup is reported separately in `pools`. See CIRISStatus
FSD/CAPABILITY_MONITORING.md §2.2 for the shared semantics.
"""

import asyncio
import os
import time
from datetime import datetime, timezone
from typing import Any

import httpx

# The image flattens hooks/ into /app; a source checkout keeps the package.
try:  # pragma: no cover - import shape differs between container and checkout
    from model_audit import (
        audit_models,
        describe_missing,
        extract_available_models,
        load_configured_models,
    )
except ImportError:  # pragma: no cover
    from hooks.model_audit import (
        audit_models,
        describe_missing,
        extract_available_models,
        load_configured_models,
    )

# Cache for rate limiting - prevents hammering upstream providers
_status_cache: dict[str, Any] = {}
_cache_timestamp: float = 0
CACHE_TTL_SECONDS = 10  # Cache status for 10 seconds

# Status levels
STATUS_OPERATIONAL = "operational"
STATUS_DEGRADED = "degraded"
STATUS_OUTAGE = "outage"
STATUS_UNKNOWN = "unknown"  # declared but never measured — never green by omission

# Rollup severity. `unknown` ranks with `degraded`: we cannot claim health we
# have not measured, but neither can we claim an outage we have not observed.
_SEVERITY = {
    STATUS_OPERATIONAL: 0,
    STATUS_DEGRADED: 1,
    STATUS_UNKNOWN: 1,
    STATUS_OUTAGE: 2,
}

# Provider roles within a pool (CIRISStatus FSD §2.3). Serving on a fallback is
# not an outage, but it is worth reporting — it precedes cost, latency and
# quality changes nothing else on the board would explain.
ROLE_PRIMARY = "primary"
ROLE_FALLBACK = "fallback"

# Declared pools. A pool is declared here rather than inferred from whatever
# happens to be in PROVIDERS, so removing the last member of a pool renders
# `unknown` instead of silently disappearing.
POOLS = {
    "llm": {
        "label": "LLM providers",
        "min_available": 1,  # redundancy: serving while any member is up
    },
}

# Latency thresholds (ms)
LATENCY_GOOD = 1000
LATENCY_DEGRADED = 3000

# Auth prefix constants
AUTH_PREFIX_BEARER = "Bearer "

# Provider configurations
#
# DO NOT add a metered search API (Brave, Exa) here. The check this module
# performs is a real, billable request — the Brave entry issued a live
# `/res/v1/web/search?q=test` with the subscription token on every uncached
# /v1/status call, and Brave bills per request since it dropped its free tier.
# Disabling the key to stop that spend did not stop the probe: an unconfigured
# provider reports `outage`, which made this service permanently `degraded`,
# which made both regions degraded on ciris.ai's public status page and cost
# ~27 points of published uptime for an outage that never happened.
#
# Search health is monitored PASSIVELY instead — derived from the real search
# traffic in `hooks/search_handler.py`, which is already paid for and is a
# truer signal (it reflects whether the key and quota actually work). See
# CIRISStatus README, "Monitoring billable providers — the right way".
#
# Keep this set aligned with the default routing chain in litellm_config.yaml.
# Monitoring the fallbacks while the primary goes unchecked renders a green
# board during a primary outage — which is exactly what happened when DeepInfra
# was added to the routing chain and not to this dict.
PROVIDERS = {
    "deepinfra": {
        "name": "DeepInfra",
        "type": "llm",
        "role": ROLE_PRIMARY,
        "check_url": "https://api.deepinfra.com/v1/openai/models",
        "env_key": "DEEPINFRA_API_KEY",
        "auth_header": "Authorization",
        "auth_prefix": AUTH_PREFIX_BEARER,
    },
    "openrouter": {
        "name": "OpenRouter",
        "type": "llm",
        "role": ROLE_FALLBACK,
        "check_url": "https://openrouter.ai/api/v1/models",
        "env_key": "OPENROUTER_API_KEY",
        "auth_header": "Authorization",
        "auth_prefix": AUTH_PREFIX_BEARER,
    },
    "groq": {
        "name": "Groq",
        "type": "llm",
        "role": ROLE_FALLBACK,
        "check_url": "https://api.groq.com/openai/v1/models",
        "env_key": "GROQ_API_KEY",
        "auth_header": "Authorization",
        "auth_prefix": AUTH_PREFIX_BEARER,
    },
    "together": {
        "name": "Together AI",
        "type": "llm",
        "role": ROLE_FALLBACK,
        "check_url": "https://api.together.xyz/v1/models",
        "env_key": "TOGETHER_API_KEY",
        "auth_header": "Authorization",
        "auth_prefix": AUTH_PREFIX_BEARER,
    },
    "billing": {
        "name": "CIRISBilling",
        "type": "internal",  # NOT pooled — nothing else serves billing
        "check_url": None,  # Dynamic from env
        "env_key": "BILLING_API_KEY",
        "env_url": "BILLING_API_URL",
        "check_path": "/health",
    },
}


def _init_provider_result(provider_id: str, config: dict) -> dict[str, Any]:
    """Initialize a provider status result with default values."""
    return {
        "provider": provider_id,
        "name": config["name"],
        "type": config["type"],
        "role": config.get("role"),  # None for non-pooled dependencies
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


def _attach_model_audit(result: dict, provider_id: str, response: httpx.Response) -> None:
    """
    Audit the configured models against the provider's own /models response.

    The health check already fetched this body, so the audit is free — no extra
    request, and nothing metered is touched. A provider that is reachable but
    has dropped a model we route to cannot serve that route, so it counts as
    degraded: the router will have to fall through it. We only ever lower a
    verdict here, never raise one.
    """
    configured = load_configured_models().get(provider_id)
    if not configured:
        return

    try:
        payload = response.json()
    except ValueError:
        return

    audit = audit_models(configured, extract_available_models(payload))
    if audit is None:
        return

    result["models"] = audit
    if audit["missing"] and result["status"] == STATUS_OPERATIONAL:
        result["status"] = STATUS_DEGRADED
        result["error"] = describe_missing(audit)


async def check_provider(
    client: httpx.AsyncClient,
    provider_id: str,
    config: dict,
) -> dict[str, Any]:
    """
    Check a single provider's health.

    For LLM providers the check URL is the provider's `/models` endpoint, so the
    same response also tells us whether the models we route there still exist —
    see `_attach_model_audit`.

    Returns:
        {
            "provider": "openrouter",
            "name": "OpenRouter",
            "type": "llm",
            "role": "fallback",
            "status": "operational|degraded|outage",
            "latency_ms": 123,
            "checked_at": "2025-01-01T00:00:00Z",
            "error": null,
            "models": {"configured": 4, "available": 4, "missing": []}
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
        if config["type"] == "llm" and result["status"] != STATUS_OUTAGE:
            _attach_model_audit(result, provider_id, resp)

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


def _worst(statuses: list[str]) -> str:
    """Return the most severe status in the list, or operational if empty."""
    if not statuses:
        return STATUS_OPERATIONAL
    return max(statuses, key=lambda s: _SEVERITY.get(s, _SEVERITY[STATUS_OUTAGE]))


def _summarize_pool(pool_id: str, config: dict, results: list[dict]) -> dict[str, Any]:
    """
    Roll a pool's members up into a single capability verdict.

    A `degraded` member counts as unavailable for the threshold — the router
    will route around it — but does not itself make the pool degraded.
    """
    members = [r for r in results if r["type"] == pool_id]
    available = [r for r in members if r["status"] == STATUS_OPERATIONAL]
    min_available = config["min_available"]

    if not members:
        status = STATUS_UNKNOWN
    elif len(available) >= min_available:
        status = STATUS_OPERATIONAL
    elif available:
        status = STATUS_DEGRADED  # still serving, but the margin is gone
    else:
        status = STATUS_OUTAGE

    primaries = [r for r in members if r["role"] == ROLE_PRIMARY]

    return {
        "label": config["label"],
        "status": status,
        "available": len(available),
        "min_available": min_available,
        "members": [r["provider"] for r in members],
        # False while healthy means "we are serving on a fallback" — a fact
        # worth surfacing, but not an impairment.
        "primary_available": (
            any(r["status"] == STATUS_OPERATIONAL for r in primaries)
            if primaries
            else None
        ),
    }


async def get_status() -> dict[str, Any]:
    """
    Get status for all providers.

    Returns cached response if within TTL to prevent upstream API abuse.

    `status` covers only what this service alone can serve: non-pooled
    dependencies, plus any pool that has fallen below `min_available`. A single
    unhealthy member of a redundant pool is reported in `providers[]` and
    `pools`, but does not degrade the service.

    Returns:
        {
            "service": "cirisproxy",
            "status": "operational|degraded|outage",
            "checked_at": "2025-01-01T00:00:00Z",
            "providers": [...],
            "pools": {"llm": {"status": ..., "available": 3, ...}}
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

    # Roll pooled providers up per capability, separately from our own status
    pools = {
        pool_id: _summarize_pool(pool_id, config, results)
        for pool_id, config in POOLS.items()
    }

    # Our own status: non-pooled dependencies, plus any pool with nothing left
    # to route to. A pool that is merely missing members is not impairment.
    contributing = [r["status"] for r in results if r["type"] not in POOLS]
    contributing += [
        p["status"] for p in pools.values() if p["status"] != STATUS_OPERATIONAL
    ]
    overall_status = _worst(contributing)

    # Cache the result
    _status_cache = {
        "service": "cirisproxy",
        "version": os.environ.get("CIRISPROXY_VERSION", "unknown"),
        "status": overall_status,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "providers": results,
        "pools": pools,
    }
    _cache_timestamp = now

    return _status_cache


# Sync wrapper for non-async contexts
def get_status_sync() -> dict[str, Any]:
    """Synchronous wrapper for get_status."""
    return asyncio.run(get_status())
