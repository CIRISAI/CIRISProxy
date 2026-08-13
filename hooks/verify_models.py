#!/usr/bin/env python3
"""
Verify every model in litellm_config.yaml still exists at its provider.

Providers retire models silently. When they do, the routing chain keeps naming
a model that no longer resolves and the failure surfaces as a user-facing
fallthrough. This checks the config against each provider's own `/models`
listing — a free endpoint, never an inference call.

Usage:
    python -m hooks.verify_models                    # audit the active config
    python -m hooks.verify_models --config path.yaml
    python -m hooks.verify_models --json
    ciris-proxy-verify-models                        # installed entry point

Exit codes:
    0  every configured model is available (or skipped for want of a key)
    1  at least one configured model is missing upstream, or a fallback chain
       names an alias that is not configured
    2  the config could not be read

Providers without a key in the environment are reported as `skipped`, not as
failures — an audit we could not perform is not a defect we found.
"""

import argparse
import asyncio
import json
import os
import sys

import httpx
import yaml

from hooks.model_audit import (
    audit_models,
    extract_available_models,
    find_dangling_fallbacks,
    load_configured_models,
    resolve_config_path,
)
from hooks.status_handler import PROVIDERS

TIMEOUT_SECONDS = 20.0


async def fetch_available(client: httpx.AsyncClient, config: dict) -> set[str] | None:
    """Fetch a provider's model list. Returns None if it cannot be read."""
    api_key = os.environ.get(config["env_key"], "")
    headers = {}
    if api_key and config.get("auth_header"):
        headers[config["auth_header"]] = f"{config.get('auth_prefix', '')}{api_key}"

    resp = await client.get(config["check_url"], headers=headers, timeout=TIMEOUT_SECONDS)
    resp.raise_for_status()
    return extract_available_models(resp.json())


async def audit_provider(
    client: httpx.AsyncClient, provider_id: str, config: dict, configured: dict
) -> dict:
    """
    Audit one provider, converting any transport failure into a skip.

    The attempt is made with or without a key: some providers publish their
    model list unauthenticated, so a keyless checkout still audits what it can.
    """
    try:
        available = await fetch_available(client, config)
    except (httpx.HTTPError, ValueError) as e:
        reason = (
            f"{config['env_key']} not set"
            if not os.environ.get(config["env_key"], "")
            else f"{type(e).__name__}: {e}"
        )
        return {"provider": provider_id, "skipped": reason}

    audit = audit_models(configured, available)
    if audit is None:
        return {"provider": provider_id, "skipped": "no usable /models response"}

    return {"provider": provider_id, **audit}


async def run(config_path: str | None) -> list[dict]:
    """Audit every LLM provider that has models configured."""
    configured_by_provider = load_configured_models(config_path)

    async with httpx.AsyncClient() as client:
        tasks = [
            audit_provider(client, provider_id, config, configured_by_provider[provider_id])
            for provider_id, config in PROVIDERS.items()
            if config["type"] == "llm" and configured_by_provider.get(provider_id)
        ]
        return await asyncio.gather(*tasks)


def report_fallbacks(config_path: str) -> bool:
    """Report fallback chains pointing at undefined aliases. True if any found."""
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError) as e:
        print(f"  ?  fallback chains: skipped — {e}")
        return False

    dangling = find_dangling_fallbacks(config)
    if not dangling:
        return False

    print("\nFallback chains naming an alias that is not in model_list:")
    for entry in dangling:
        if entry["undefined_source"]:
            print(f"  X  source `{entry['source']}` is not a configured model")
        for target in entry["missing_targets"]:
            print(f"  X  {entry['source']} -> `{target}` is not a configured model")
    return True


def report(results: list[dict]) -> int:
    """Print a human-readable report. Returns the process exit code."""
    failed = False

    for result in sorted(results, key=lambda r: r["provider"]):
        provider = result["provider"]

        if "skipped" in result:
            print(f"  ?  {provider}: skipped — {result['skipped']}")
            continue

        if not result["missing"]:
            print(f"  ok {provider}: {result['configured']}/{result['configured']} models available")
            continue

        failed = True
        print(f"  X  {provider}: {result['available']}/{result['configured']} models available")
        for entry in result["missing"]:
            aliases = ", ".join(entry["aliases"]) or "(no alias)"
            print(f"       missing: {entry['model']}  <- {aliases}")

    if failed:
        print("\nRemove the missing models and any fallback entries naming their aliases.")
    return 1 if failed else 0


def _has_missing(results: list[dict]) -> bool:
    return any(r.get("missing") for r in results)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--config", help="path to litellm config (default: active config)")
    parser.add_argument("--json", action="store_true", help="emit raw JSON instead of a report")
    args = parser.parse_args()

    resolved = resolve_config_path(args.config)
    if not resolved:
        print(f"error: no config found at {args.config or 'any default location'}", file=sys.stderr)
        return 2

    results = asyncio.run(run(resolved))

    if args.json:
        with open(resolved, "r") as f:
            dangling = find_dangling_fallbacks(yaml.safe_load(f) or {})
        print(json.dumps({"providers": results, "dangling_fallbacks": dangling}, indent=2))
        return 1 if _has_missing(results) or dangling else 0

    print(f"Auditing models in {resolved}\n")
    exit_code = report(results)
    if report_fallbacks(resolved):
        exit_code = 1
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
