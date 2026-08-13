"""
Model availability audit for CIRISProxy.

Providers deprecate models without warning. When they do, the routing chain in
litellm_config.yaml keeps pointing at a model that no longer exists, and the
first thing that notices is a user request failing over — or exhausting the
chain. `llama-4-scout` is the worked example: retired upstream while still
named in every Scout fallback we had.

This module answers "does each model we route to still exist at its provider?"
by comparing the configured model list against the provider's own `/models`
response — the **same response** the health check in `status_handler.py`
already fetches. Auditing therefore costs zero additional upstream requests and
never touches a metered inference endpoint (see the PROVIDERS note in
`status_handler.py` for why that matters).

Two consumers:
- `status_handler.check_provider` — attaches a per-provider `models` block to
  /v1/status and marks a provider degraded when it can no longer serve what we
  route to it.
- `scripts/verify_models.py` — standalone CLI for CI and local runs, exits
  non-zero when the config names a model no provider will serve.
"""

import os
from typing import Any

import yaml

# Where the running container's config lives. entrypoint.sh exports
# LITELLM_CONFIG_PATH; the fallbacks cover the image layout and a source
# checkout so the CLI works without arguments in both.
_CONFIG_PATH_ENV = "LITELLM_CONFIG_PATH"
_CONFIG_FALLBACKS = (
    "/app/config.processed.yaml",
    "/app/config.yaml",
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "litellm_config.yaml"),
)

# LiteLLM model-string prefix -> provider id in status_handler.PROVIDERS.
# A model string is "<prefix>/<upstream model id>"; the upstream id is what the
# provider's /models endpoint reports, so the comparison is exact after the
# prefix is stripped.
PROVIDER_PREFIXES = {
    "deepinfra": "deepinfra",
    "openrouter": "openrouter",
    "groq": "groq",
    "together_ai": "together",
}

# Models with no prefix (e.g. "gpt-4o-mini") are OpenAI's. OpenAI is an
# optional fallback and is not health-checked, so it is audited by the CLI only
# when OPENAI_API_KEY is set — never by /v1/status.
DEFAULT_PROVIDER = "openai"

_config_cache: dict[str, dict[str, dict[str, list[str]]]] = {}


# A config path can arrive from a CLI argument or an environment variable, and
# it ends up as an argument to open(). Only ever accept a real, existing YAML
# file: symlinks are resolved so the extension check cannot be bypassed by
# pointing a .yaml symlink at something else, and directories, devices and
# sockets are rejected outright. This module reads LiteLLM configs — there is
# no legitimate call that needs anything else.
_CONFIG_SUFFIXES = (".yaml", ".yml")


def _validated_config_path(candidate: str | None) -> str | None:
    """Canonicalize a candidate path, or None if it is not a readable YAML file."""
    if not candidate:
        return None

    resolved = os.path.realpath(candidate)
    if not resolved.endswith(_CONFIG_SUFFIXES):
        return None
    if not os.path.isfile(resolved):
        return None
    return resolved


def resolve_config_path(path: str | None = None) -> str | None:
    """
    Find the active LiteLLM config, validated and canonicalized.

    Returns None when no candidate resolves to an existing YAML file — callers
    treat that as "not audited", never as a failure.
    """
    if path:
        return _validated_config_path(path)

    env_path = os.environ.get(_CONFIG_PATH_ENV)
    candidates = (env_path, *_CONFIG_FALLBACKS) if env_path else _CONFIG_FALLBACKS
    for candidate in candidates:
        validated = _validated_config_path(candidate)
        if validated:
            return validated
    return None


def read_config(path: str | None = None) -> dict | None:
    """
    Read and parse a LiteLLM config.

    The single place in this codebase that opens a config file: every caller
    goes through here, so path validation cannot be skipped by adding another
    reader. Returns None when the file cannot be read or parsed.
    """
    resolved = resolve_config_path(path)
    if not resolved:
        return None

    try:
        with open(resolved) as f:
            return yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError):
        return None


def split_model_string(model: str) -> tuple[str, str]:
    """
    Split a LiteLLM model string into (provider_id, upstream model id).

    >>> split_model_string("deepinfra/Qwen/Qwen3.6-35B-A3B")
    ('deepinfra', 'Qwen/Qwen3.6-35B-A3B')
    >>> split_model_string("gpt-4o-mini")
    ('openai', 'gpt-4o-mini')
    """
    prefix, sep, remainder = model.partition("/")
    if sep and prefix in PROVIDER_PREFIXES:
        return PROVIDER_PREFIXES[prefix], remainder
    return DEFAULT_PROVIDER, model


def parse_configured_models(config: dict) -> dict[str, dict[str, list[str]]]:
    """
    Map each provider to the upstream models we route to it.

    Returns {provider_id: {upstream_model_id: [aliases that use it]}}. The
    aliases make failures actionable: "Qwen3.6 is gone" matters far more when
    the report names `default` as one of the aliases pointing at it.
    """
    configured: dict[str, dict[str, list[str]]] = {}

    for entry in config.get("model_list") or []:
        params = entry.get("litellm_params") or {}
        model = params.get("model")
        if not isinstance(model, str) or not model:
            continue

        provider_id, upstream_model = split_model_string(model)
        aliases = configured.setdefault(provider_id, {}).setdefault(upstream_model, [])
        alias = entry.get("model_name")
        if alias and alias not in aliases:
            aliases.append(alias)

    return configured


def load_configured_models(path: str | None = None) -> dict[str, dict[str, list[str]]]:
    """
    Load and cache the configured models per provider.

    Returns an empty mapping when the config cannot be read or parsed — an
    audit we cannot perform must not be reported as a failure (that is the
    mistake the Brave probe made, in a different costume).
    """
    resolved = resolve_config_path(path)
    if not resolved:
        return {}

    if resolved in _config_cache:
        return _config_cache[resolved]

    config = read_config(resolved)
    if config is None:
        return {}

    configured = parse_configured_models(config)
    _config_cache[resolved] = configured
    return configured


def extract_available_models(payload: Any) -> set[str] | None:
    """
    Pull model ids out of a `/models` response.

    Handles both shapes in use: OpenAI-style `{"data": [{"id": ...}]}` (Groq,
    OpenRouter, DeepInfra) and Together AI's bare top-level array.

    Returns None when the payload is not a recognizable model list, which the
    callers treat as "not audited" rather than "nothing available".
    """
    if isinstance(payload, list):
        data = payload
    elif isinstance(payload, dict):
        data = payload.get("data")
    else:
        return None

    if not isinstance(data, list) or not data:
        return None

    ids = {
        entry["id"]
        for entry in data
        if isinstance(entry, dict) and isinstance(entry.get("id"), str)
    }
    return ids or None


def audit_models(
    configured: dict[str, list[str]],
    available: set[str] | None,
) -> dict[str, Any] | None:
    """
    Compare one provider's configured models against what it reports serving.

    Returns None when there is nothing to audit (no configured models, or an
    unusable /models response).
    """
    if not configured or available is None:
        return None

    missing = sorted(model for model in configured if model not in available)

    return {
        "configured": len(configured),
        "available": len(configured) - len(missing),
        "missing": [
            {"model": model, "aliases": configured[model]} for model in missing
        ],
    }


def find_dangling_fallbacks(config: dict) -> list[dict[str, Any]]:
    """
    Find fallback entries naming an alias that no longer exists in model_list.

    Removing a retired model is only half the repair: the fallback chains that
    named it are left pointing at nothing, and LiteLLM will not tell us until a
    request walks the chain during an outage — the worst possible moment to
    discover it. This is a pure config check, no network required.

    Returns [{"source": alias, "missing_targets": [...]}, ...].
    """
    defined = {
        entry.get("model_name")
        for entry in config.get("model_list") or []
        if entry.get("model_name")
    }

    dangling = []
    for rule in config.get("router_settings", {}).get("fallbacks") or []:
        if not isinstance(rule, dict):
            continue
        for source, targets in rule.items():
            missing_targets = [t for t in targets or [] if t not in defined]
            if source not in defined:
                missing_targets = missing_targets or []
                dangling.append(
                    {"source": source, "undefined_source": True, "missing_targets": missing_targets}
                )
            elif missing_targets:
                dangling.append(
                    {"source": source, "undefined_source": False, "missing_targets": missing_targets}
                )

    return dangling


def describe_missing(audit: dict[str, Any]) -> str:
    """One-line summary of a provider's missing models, for a status `error`."""
    names = ", ".join(entry["model"] for entry in audit["missing"])
    count = len(audit["missing"])
    noun = "model" if count == 1 else "models"
    return f"{count} configured {noun} unavailable: {names}"
