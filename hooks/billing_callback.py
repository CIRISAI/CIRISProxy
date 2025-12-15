"""
CIRISBilling Integration Callback for LiteLLM

Key design: 1 credit = 1 interaction (not 1 LLM call)
Uses interaction_id + idempotency to ensure single charge per interaction.

The on-device CIRIS Agent makes 12-70 LLM calls per user interaction.
All calls share the same interaction_id. CIRISBilling's idempotency ensures
only the first /charge call deducts credits; subsequent calls are no-ops.

Interaction limits (triggers new charge):
- MAX_INTERACTION_AGE_SECONDS (300): If same interaction_id reused after 5 minutes
- MAX_LLM_CALLS_PER_INTERACTION (80): If interaction exceeds 80 LLM calls

SECURITY NOTE: This callback MUST NOT log any PII including:
- User IDs (external_id, Google IDs)
- API keys
- Message content
- Full request/response bodies
Only log: interaction_id, model names, token counts, status codes, timing
"""

from __future__ import annotations

import hashlib
import logging
import os
import sys
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

import httpx
from litellm.integrations.custom_logger import CustomLogger


# Custom exceptions for billing errors
class BillingError(Exception):
    """Base exception for billing-related errors."""


class InvalidAPIKeyError(BillingError):
    """Raised when API key format is invalid."""


class MissingInteractionIdError(BillingError):
    """Raised when interaction_id is missing from request."""


class InsufficientCreditsError(BillingError):
    """Raised when user has no credits available."""


class BillingServiceError(BillingError):
    """Raised when billing service returns an error."""


class BillingServiceUnavailableError(BillingError):
    """Raised when billing service is unreachable."""

# Add sdk to path for logshipper import
# In production: /app/billing_callback.py needs /app in path to find /app/sdk/
# In local: hooks/billing_callback.py needs parent dir to find sdk/
_this_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_this_dir)
# If file is directly in /app, add /app. If in hooks/, add parent.
if os.path.basename(_this_dir) == "hooks":
    sys.path.insert(0, _parent_dir)
else:
    sys.path.insert(0, _this_dir)
from sdk.logshipper import LogShipper

# Configure logging to both stdout and file
LOG_FILE = os.environ.get("CIRIS_LOG_FILE", "/app/logs/cirisproxy.log")
LOG_LEVEL = os.environ.get("CIRIS_LOG_LEVEL", "INFO")

# Ensure log directory exists (gracefully handle permission errors in CI/test)
try:
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
except PermissionError:
    # Fall back to temp directory if /app not writable (e.g., in CI)
    import tempfile
    LOG_FILE = os.path.join(tempfile.gettempdir(), "cirisproxy.log")

# Set up root logger for our module
logger = logging.getLogger(__name__)
logger.setLevel(getattr(logging, LOG_LEVEL))

# File handler - append mode, with rotation-friendly format
_file_handler = logging.FileHandler(LOG_FILE, mode='a')
_file_handler.setLevel(logging.DEBUG)
_file_handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s [%(name)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
))
logger.addHandler(_file_handler)

# Also log to stdout
_stream_handler = logging.StreamHandler(sys.stdout)
_stream_handler.setLevel(getattr(logging, LOG_LEVEL))
_stream_handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s [%(name)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
))
logger.addHandler(_stream_handler)

# Initialize CIRISLens log shipper (if token configured)
_lens_shipper: LogShipper | None = None
_LENS_TOKEN = os.environ.get("CIRISLENS_TOKEN", "")
if _LENS_TOKEN:
    _lens_shipper = LogShipper(
        service_name="cirisproxy",
        token=_LENS_TOKEN,
        endpoint=os.environ.get(
            "CIRISLENS_ENDPOINT",
            "https://agents.ciris.ai/lens/api/v1/logs/ingest"
        ),
        batch_size=50,
        flush_interval=10.0,
    )
    logger.info("CIRISLens log shipper initialized")
else:
    logger.info("CIRISLens log shipper disabled (CIRISLENS_TOKEN not set)")

# Interaction limits - exceeding these triggers a new charge
MAX_INTERACTION_AGE_SECONDS = 300  # 5 minutes - interaction reused after this is a new task
MAX_LLM_CALLS_PER_INTERACTION = 80  # Exceeding this triggers a new charge

# OAuth provider constant
DEFAULT_OAUTH_PROVIDER = "oauth:google"

# Vision/multimodal routing configuration
# Groq doesn't support system messages + images, so route multimodal to these providers
VISION_COMPATIBLE_PROVIDERS = ["together_ai", "openrouter"]
VISION_FALLBACK_MODEL = "together_ai/meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"


def _is_multimodal_request(data: dict) -> bool:
    """
    Detect if request contains multimodal (image) content.

    Returns True if any message has content as an array with image_url type.
    """
    messages = data.get("messages", [])
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    return True
    return False


def _get_vision_compatible_model(current_model: str) -> str | None:
    """
    If current model routes to a non-vision-compatible provider (Groq),
    return a vision-compatible alternative. Returns None if already compatible.
    """
    if not current_model:
        return VISION_FALLBACK_MODEL

    model_lower = current_model.lower()

    # Check if already using a vision-compatible provider
    for provider in VISION_COMPATIBLE_PROVIDERS:
        if provider.lower() in model_lower:
            return None  # Already compatible

    # Groq models need to be rerouted for vision
    if "groq" in model_lower:
        # Map to Together AI equivalent
        if "maverick" in model_lower or "llama-4" in model_lower:
            return "together_ai/meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"
        elif "llama-3.3" in model_lower or "llama-3.1-70b" in model_lower:
            return "together_ai/meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
        elif "llama-3.1-8b" in model_lower:
            return "together_ai/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
        else:
            return VISION_FALLBACK_MODEL

    # Default/fast models - check if they route to Groq
    if current_model in ("default", "fast"):
        return VISION_FALLBACK_MODEL

    return None  # Assume compatible

# Global aggregate usage counters (persist across all interactions)
_global_usage = {
    "total_llm_calls": 0,
    "total_prompt_tokens": 0,
    "total_completion_tokens": 0,
    "total_cost_cents": 0.0,
    "total_interactions": 0,
    "models_used": defaultdict(int),  # model -> call count
    "start_time": None,
}


def _hash_id(user_id: str) -> str:
    """Hash user ID for safe logging. Only first 8 chars of hash shown."""
    if not user_id:
        return "unknown"
    return hashlib.sha256(user_id.encode()).hexdigest()[:8]


def _ship_log(
    level: str,
    message: str,
    event: str,
    interaction_id: str | None = None,
    user_hash: str | None = None,
    **extra,
) -> None:
    """Ship a log entry to CIRISLens (if configured)."""
    if _lens_shipper is None:
        return

    kwargs = {
        "event": event,
        "interaction_id": interaction_id,
    }
    if user_hash:
        kwargs["user_hash"] = user_hash
    kwargs.update(extra)

    if level == "DEBUG":
        _lens_shipper.debug(message, **kwargs)
    elif level == "INFO":
        _lens_shipper.info(message, **kwargs)
    elif level == "WARNING":
        _lens_shipper.warning(message, **kwargs)
    elif level == "ERROR":
        _lens_shipper.error(message, **kwargs)
    elif level == "CRITICAL":
        _lens_shipper.critical(message, **kwargs)


def get_global_usage() -> dict[str, Any]:
    """Get current global aggregate usage stats."""
    return {
        "total_llm_calls": _global_usage["total_llm_calls"],
        "total_prompt_tokens": _global_usage["total_prompt_tokens"],
        "total_completion_tokens": _global_usage["total_completion_tokens"],
        "total_tokens": _global_usage["total_prompt_tokens"] + _global_usage["total_completion_tokens"],
        "total_cost_cents": _global_usage["total_cost_cents"],
        "total_cost_dollars": _global_usage["total_cost_cents"] / 100,
        "total_interactions": _global_usage["total_interactions"],
        "models_used": dict(_global_usage["models_used"]),
        "start_time": _global_usage["start_time"].isoformat() if _global_usage["start_time"] else None,
    }


def _generate_continuation_id(original_id: str, reason: str) -> str:
    """
    Generate a new unique interaction_id for continuation billing.

    When an interaction exceeds limits (time or call count), we generate
    a new ID to trigger a fresh charge while maintaining traceability.
    """
    return f"{original_id}__cont_{reason}_{uuid.uuid4().hex[:8]}"


class CIRISBillingCallback(CustomLogger):
    """
    LiteLLM callback that integrates with CIRISBilling.

    Auth Flow:
    1. Extract user identity from API key (format: "google:{user_id}")
    2. Extract interaction_id from request metadata
    3. On first call per interaction: verify credits via /credits/check
    4. On every call: charge via /charges (idempotent with interaction_id)
    5. On every call: stream usage via /litellm/usage (fire-and-forget)
    """

    def __init__(self) -> None:
        super().__init__()
        self.billing_url = os.environ.get("BILLING_API_URL", "https://billing.ciris.ai")
        self.billing_key = os.environ.get("BILLING_API_KEY", "")

        # Cache authorized interactions to avoid repeated auth checks
        # Key: interaction_id, Value: True
        self._authorized_interactions: dict[str, bool] = {}

        # Aggregate usage per interaction for final logging
        # Key: interaction_id, Value: usage dict
        self._interaction_usage: dict[str, dict[str, Any]] = defaultdict(
            lambda: {
                "external_id": "",
                "llm_calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "models": set(),
                "cost_cents": 0.0,
                "duration_ms": 0,
                "errors": 0,
                "fallbacks": 0,
                "start_time": None,
            }
        )

        # HTTP client for billing API calls
        self._client: httpx.AsyncClient | None = None

        logger.info(
            "CIRISBillingCallback initialized, billing_url=%s",
            self.billing_url,
        )

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(10.0, connect=5.0),
                headers={"X-API-Key": self.billing_key},
            )
        return self._client

    def _parse_user_key(self, api_key: str) -> tuple[str, str]:
        """
        Parse the API key to extract OAuth provider and external ID.

        Expected format: "google:{user_id}" or just "{user_id}"
        Returns: (oauth_provider, external_id)
        """
        if not api_key:
            return DEFAULT_OAUTH_PROVIDER, ""

        if api_key.startswith("google:"):
            return DEFAULT_OAUTH_PROVIDER, api_key[7:]  # Skip "google:" prefix
        elif ":" in api_key:
            provider, user_id = api_key.split(":", 1)
            return f"oauth:{provider}", user_id
        else:
            # Assume it's just the user ID with Google OAuth
            return DEFAULT_OAUTH_PROVIDER, api_key

    async def _log_interaction_usage(self, interaction_id: str) -> None:
        """
        Log aggregated usage for a completed interaction to the billing service.

        Called when an interaction is finalized (stale timeout or call limit exceeded).
        """
        usage_data = self._interaction_usage.get(interaction_id)
        if not usage_data:
            return

        external_id = usage_data.get("external_id", "")
        if not external_id:
            return

        # Calculate total duration from start to now
        start_time = usage_data.get("start_time")
        if start_time:
            total_duration_ms = int(
                (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            )
        else:
            total_duration_ms = usage_data.get("duration_ms", 0)

        try:
            client = await self._get_client()
            resp = await client.post(
                f"{self.billing_url}/v1/billing/litellm/usage",
                json={
                    "oauth_provider": DEFAULT_OAUTH_PROVIDER,
                    "external_id": external_id,
                    "interaction_id": interaction_id,
                    "total_llm_calls": usage_data.get("llm_calls", 0),
                    "total_prompt_tokens": usage_data.get("prompt_tokens", 0),
                    "total_completion_tokens": usage_data.get("completion_tokens", 0),
                    "models_used": list(usage_data.get("models", set())),
                    "actual_cost_cents": int(usage_data.get("cost_cents", 0)),
                    "duration_ms": total_duration_ms,
                    "error_count": usage_data.get("errors", 0),
                    "fallback_count": usage_data.get("fallbacks", 0),
                },
            )

            if resp.status_code == 200:
                print(
                    f"[CIRIS USAGE] LOGGED interaction={interaction_id} "
                    f"calls={usage_data.get('llm_calls', 0)} "
                    f"tokens={usage_data.get('prompt_tokens', 0)}+{usage_data.get('completion_tokens', 0)} "
                    f"cost_cents={usage_data.get('cost_cents', 0):.2f}",
                    flush=True
                )
            else:
                print(
                    f"[CIRIS USAGE] FAILED to log interaction={interaction_id} status={resp.status_code}",
                    flush=True
                )

        except httpx.RequestError as e:
            print(f"[CIRIS USAGE] ERROR logging interaction={interaction_id} error={type(e).__name__}", flush=True)

        # Clean up in-memory tracking for this interaction
        self._interaction_usage.pop(interaction_id, None)
        self._authorized_interactions.pop(interaction_id, None)

    async def _stream_usage_to_billing(
        self,
        oauth_provider: str,
        external_id: str,
        interaction_id: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        cost_cents: float,
        duration_ms: int,
    ) -> None:
        """
        Stream per-call usage data to billing service (fire-and-forget).

        Called after every successful LLM call. This ensures usage is logged
        even if the interaction never explicitly completes (no finalization signal).

        The billing service aggregates these per-call records by interaction_id.
        """
        try:
            client = await self._get_client()
            resp = await client.post(
                f"{self.billing_url}/v1/billing/litellm/usage",
                json={
                    "oauth_provider": oauth_provider,
                    "external_id": external_id,
                    "interaction_id": interaction_id,
                    "total_llm_calls": 1,  # Single call
                    "total_prompt_tokens": prompt_tokens,
                    "total_completion_tokens": completion_tokens,
                    "models_used": [model],
                    # Keep as float - per-call costs are often fractions of a cent
                    "actual_cost_cents": round(cost_cents, 4),
                    "duration_ms": duration_ms,
                    "error_count": 0,
                    "fallback_count": 0,
                },
            )

            if resp.status_code == 200:
                logger.debug(
                    "interaction=%s usage_streamed=true model=%s tokens=%d/%d",
                    interaction_id,
                    model,
                    prompt_tokens,
                    completion_tokens,
                )
            else:
                # Log at debug level to avoid spamming - usage logging is best-effort
                logger.debug(
                    "interaction=%s usage_stream=failed status=%d",
                    interaction_id,
                    resp.status_code,
                )

        except httpx.RequestError as e:
            # Best effort - don't fail the request if usage logging fails
            logger.debug(
                "interaction=%s usage_stream=error error_type=%s",
                interaction_id,
                type(e).__name__,
            )

    def _handle_multimodal_routing(self, data: dict[str, Any]) -> None:
        """Handle vision request routing to compatible providers."""
        if not _is_multimodal_request(data):
            return

        current_model = data.get("model", "")
        vision_model = _get_vision_compatible_model(current_model)
        if vision_model:
            print(
                f"[CIRIS VISION] Multimodal detected, rerouting from {current_model} to {vision_model}",
                flush=True
            )
            data["model"] = vision_model
            _ship_log(
                "INFO",
                f"Vision request rerouted to {vision_model}",
                event="vision_reroute",
                original_model=current_model,
                vision_model=vision_model,
            )

    def _extract_api_key(self, user_api_key_dict: Any) -> str:
        """Extract API key from various input formats."""
        if hasattr(user_api_key_dict, "api_key"):
            return user_api_key_dict.api_key or ""
        elif isinstance(user_api_key_dict, dict):
            return user_api_key_dict.get("api_key", "")
        return ""

    async def _check_interaction_limits(
        self,
        interaction_id: str,
        original_interaction_id: str,
    ) -> str:
        """
        Check if interaction has exceeded limits and needs continuation.

        Returns:
            The interaction_id to use (may be a continuation ID)
        """
        if interaction_id not in self._interaction_usage:
            return interaction_id

        usage_data = self._interaction_usage[interaction_id]
        start_time = usage_data.get("start_time")
        llm_calls = usage_data.get("llm_calls", 0)
        now = datetime.now(timezone.utc)

        # Check for stale interaction (>5 minutes old)
        if start_time:
            age_seconds = (now - start_time).total_seconds()
            if age_seconds > MAX_INTERACTION_AGE_SECONDS:
                await self._log_interaction_usage(original_interaction_id)
                new_id = _generate_continuation_id(original_interaction_id, "stale")
                print(f"[CIRIS DEBUG] interaction={original_interaction_id} STALE after {age_seconds:.0f}s, new_id={new_id}", flush=True)
                self._authorized_interactions.pop(original_interaction_id, None)
                return new_id

        # Check for excessive LLM calls (>80 calls)
        if llm_calls >= MAX_LLM_CALLS_PER_INTERACTION:
            await self._log_interaction_usage(original_interaction_id)
            new_id = _generate_continuation_id(original_interaction_id, f"calls{llm_calls}")
            print(f"[CIRIS DEBUG] interaction={original_interaction_id} EXCEEDED {llm_calls} calls, new_id={new_id}", flush=True)
            self._authorized_interactions.pop(original_interaction_id, None)
            return new_id

        return interaction_id

    def _initialize_interaction_tracking(
        self,
        interaction_id: str,
        external_id: str,
        original_interaction_id: str,
    ) -> None:
        """Initialize usage tracking for a new interaction."""
        if interaction_id in self._interaction_usage:
            return

        now = datetime.now(timezone.utc)
        self._interaction_usage[interaction_id]["external_id"] = external_id
        self._interaction_usage[interaction_id]["start_time"] = now
        if interaction_id != original_interaction_id:
            self._interaction_usage[interaction_id]["original_interaction_id"] = original_interaction_id

    async def _check_billing_auth(
        self,
        oauth_provider: str,
        external_id: str,
        interaction_id: str,
        model: str,
    ) -> None:
        """
        Check billing authorization for an interaction.

        Raises:
            InsufficientCreditsError: If user has no credits
            BillingServiceError: If billing service returns error
            BillingServiceUnavailableError: If billing service unreachable
        """
        id_preview = f"{external_id[:4]}...{external_id[-4:]}" if len(external_id) > 8 else external_id
        print(f"[CIRIS DEBUG] interaction={interaction_id} auth=checking oauth={oauth_provider} external_id_preview={id_preview} external_id_hash={_hash_id(external_id)} model={model}", flush=True)

        try:
            client = await self._get_client()
            resp = await client.post(
                f"{self.billing_url}/v1/billing/credits/check",
                json={
                    "oauth_provider": oauth_provider,
                    "external_id": external_id,
                },
            )

            if resp.status_code != 200:
                logger.error(
                    "interaction=%s billing_auth=error status=%d body=%s",
                    interaction_id,
                    resp.status_code,
                    resp.text[:200],
                )
                raise BillingServiceError(f"Billing service error: {resp.status_code}")

            result = resp.json()
            has_credit = result.get("has_credit", False)
            free_uses = result.get("free_uses_remaining") or 0
            paid_credits = result.get("credits_remaining") or 0
            daily_free = result.get("daily_free_uses_remaining") or 0
            total_credits = free_uses + paid_credits + daily_free

            print(f"[CIRIS DEBUG] interaction={interaction_id} billing_response has_credit={has_credit} free={free_uses} paid={paid_credits} daily={daily_free} total={total_credits}", flush=True)

            if not has_credit:
                logger.info("interaction=%s auth=denied reason=no_credits total=%d", interaction_id, total_credits)
                _ship_log(
                    "WARNING",
                    "Auth denied: no credits available",
                    event="auth_denied",
                    interaction_id=interaction_id,
                    user_hash=_hash_id(external_id),
                    credits_remaining=total_credits,
                )
                raise InsufficientCreditsError(f"Insufficient credits: {total_credits} available")

            # Success - cache authorization
            self._authorized_interactions[interaction_id] = True
            logger.info("interaction=%s auth=granted credits=%d", interaction_id, total_credits)
            _ship_log(
                "INFO",
                f"Auth granted with {total_credits} credits",
                event="auth_granted",
                interaction_id=interaction_id,
                user_hash=_hash_id(external_id),
                credits_remaining=total_credits,
                model=model,
            )

            # Track new interaction in global stats
            _global_usage["total_interactions"] += 1
            if _global_usage["start_time"] is None:
                _global_usage["start_time"] = datetime.now(timezone.utc)

        except httpx.RequestError as e:
            logger.error("interaction=%s billing=unreachable error_type=%s", interaction_id, type(e).__name__)
            _ship_log(
                "ERROR",
                f"Billing service unreachable: {type(e).__name__}",
                event="billing_error",
                interaction_id=interaction_id,
                user_hash=_hash_id(external_id),
                error_type=type(e).__name__,
            )
            raise BillingServiceUnavailableError("Billing service unavailable. Please try again later.")

    async def async_pre_call_hook(
        self,
        user_api_key_dict: Any,  # Can be dict or UserAPIKeyAuth Pydantic model
        cache: Any,
        data: dict[str, Any],
        call_type: str,
    ) -> None:
        """
        Pre-request: Verify user has credits (cached per interaction).

        Called before every LLM request. We cache the auth result per
        interaction_id to avoid hammering the billing API.

        Also handles multimodal routing - Groq doesn't support system messages
        with images, so we route vision requests to Together AI instead.
        """
        # Route vision requests to compatible providers
        self._handle_multimodal_routing(data)

        # Extract and validate user identity
        api_key = self._extract_api_key(user_api_key_dict)
        oauth_provider, external_id = self._parse_user_key(api_key)

        if not external_id:
            logger.warning("Auth denied: missing user identifier in API key")
            raise InvalidAPIKeyError("Invalid API key format. Expected: google:{user_id}")

        # Get and validate interaction_id
        metadata = data.get("metadata", {})
        interaction_id = metadata.get("interaction_id")

        if not interaction_id:
            logger.warning("Auth denied: missing interaction_id")
            raise MissingInteractionIdError(
                "Missing interaction_id in request metadata. "
                "The on-device agent must set metadata.interaction_id for each interaction."
            )

        # Store for post-call hooks
        if "metadata" not in data:
            data["metadata"] = {}
        data["metadata"]["_ciris_oauth_provider"] = oauth_provider
        data["metadata"]["_ciris_external_id"] = external_id

        # Check limits and get (possibly updated) interaction_id
        original_interaction_id = interaction_id
        interaction_id = await self._check_interaction_limits(interaction_id, original_interaction_id)
        data["metadata"]["_ciris_interaction_id"] = interaction_id

        # Initialize usage tracking
        self._initialize_interaction_tracking(interaction_id, external_id, original_interaction_id)

        # Check if already authorized (cache hit)
        if interaction_id in self._authorized_interactions:
            logger.debug("interaction=%s auth=cached", interaction_id)
            return

        # First call for this interaction - check billing auth
        await self._check_billing_auth(oauth_provider, external_id, interaction_id, data.get("model", "unknown"))

    def _extract_success_metadata(
        self, kwargs: dict[str, Any]
    ) -> tuple[str | None, str | None, str | None, dict[str, Any], dict[str, Any]]:
        """
        Extract metadata from kwargs for success event processing.

        Returns:
            Tuple of (oauth_provider, external_id, interaction_id, metadata, litellm_params)
        """
        litellm_params = kwargs.get("litellm_params", {})
        metadata = litellm_params.get("metadata", {})

        oauth_provider = metadata.get("_ciris_oauth_provider")
        external_id = metadata.get("_ciris_external_id")
        interaction_id = metadata.get("_ciris_interaction_id")

        return oauth_provider, external_id, interaction_id, metadata, litellm_params

    def _extract_usage_data(
        self,
        response_obj: Any,
        kwargs: dict[str, Any],
        litellm_params: dict[str, Any],
        start_time: datetime,
        end_time: datetime,
    ) -> dict[str, Any]:
        """
        Extract usage data from response object.

        Returns:
            Dict with usage metrics (tokens, cost, duration, model info)
        """
        usage = getattr(response_obj, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0 if usage else 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0 if usage else 0
        model = kwargs.get("model", "unknown")

        actual_model = litellm_params.get("model", model)
        api_base = litellm_params.get("api_base", "")

        hidden_params = getattr(response_obj, "_hidden_params", {})
        cost_dollars = 0.0
        if isinstance(hidden_params, dict):
            cost_dollars = hidden_params.get("response_cost", 0) or 0

        duration_ms = 0
        if start_time and end_time:
            duration_ms = int((end_time - start_time).total_seconds() * 1000)

        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "model": model,
            "actual_model": actual_model,
            "api_base": api_base,
            "cost_dollars": cost_dollars,
            "duration_ms": duration_ms,
        }

    def _log_llm_request(
        self,
        interaction_id: str,
        external_id: str,
        usage_data: dict[str, Any],
        metadata: dict[str, Any],
    ) -> None:
        """Log LLM request to logger and CIRISLens."""
        retry_count = metadata.get("retry_count", 0)
        previous_error = metadata.get("previous_error", "")
        original_request_id = metadata.get("original_request_id", "")

        log_data = {
            "event": "llm_request",
            "interaction_id": interaction_id,
            "user_hash": _hash_id(external_id),
            "model": usage_data["actual_model"],
            "api_base": usage_data["api_base"][:50] if usage_data["api_base"] else "default",
            "prompt_tokens": usage_data["prompt_tokens"],
            "completion_tokens": usage_data["completion_tokens"],
        }

        if retry_count > 0:
            log_data["retry_count"] = retry_count
            log_data["previous_error"] = previous_error
            log_data["original_request_id"] = original_request_id[:8] if original_request_id else None
            logger.info(
                "llm_request interaction=%s model=%s api_base=%s tokens=%d/%d RETRY=%d prev_error=%s orig_req=%s",
                interaction_id[:8] if interaction_id else "none",
                usage_data["actual_model"],
                usage_data["api_base"][:30] if usage_data["api_base"] else "default",
                usage_data["prompt_tokens"],
                usage_data["completion_tokens"],
                retry_count,
                previous_error,
                original_request_id[:8] if original_request_id else "none",
            )
        else:
            logger.info(
                "llm_request interaction=%s model=%s api_base=%s tokens=%d/%d",
                interaction_id[:8] if interaction_id else "none",
                usage_data["actual_model"],
                usage_data["api_base"][:30] if usage_data["api_base"] else "default",
                usage_data["prompt_tokens"],
                usage_data["completion_tokens"],
            )

        _ship_log("INFO", "LLM request completed", **log_data)

    async def _charge_interaction(
        self,
        oauth_provider: str,
        external_id: str,
        interaction_id: str,
        model: str,
    ) -> None:
        """
        Charge the interaction (idempotent - first call charges, rest are no-ops).
        """
        try:
            client = await self._get_client()
            resp = await client.post(
                f"{self.billing_url}/v1/billing/charges",
                json={
                    "oauth_provider": oauth_provider,
                    "external_id": external_id,
                    "amount_minor": 1,
                    "currency": "USD",
                    "description": f"LiteLLM interaction: {interaction_id}",
                    "idempotency_key": f"litellm:{interaction_id}",
                },
            )

            if resp.status_code == 201:
                result = resp.json()
                logger.debug(
                    "interaction=%s charge=new balance_after=%s",
                    interaction_id,
                    result.get("balance_after"),
                )
                _ship_log(
                    "INFO",
                    "Credit charged for interaction",
                    event="charge_created",
                    interaction_id=interaction_id,
                    user_hash=_hash_id(external_id),
                    balance_after=result.get("balance_after"),
                    model=model,
                )
            elif resp.status_code in (200, 409):
                logger.debug("interaction=%s charge=idempotent", interaction_id)
            else:
                logger.warning(
                    "interaction=%s charge=failed status=%d body=%s",
                    interaction_id,
                    resp.status_code,
                    resp.text[:100] if resp.text else "",
                )
                _ship_log(
                    "WARNING",
                    f"Charge failed with status {resp.status_code}",
                    event="charge_failed",
                    interaction_id=interaction_id,
                    user_hash=_hash_id(external_id),
                    status_code=resp.status_code,
                )

        except httpx.RequestError as e:
            logger.error("interaction=%s charge=error error_type=%s", interaction_id, type(e).__name__)
            _ship_log(
                "ERROR",
                f"Charge request failed: {type(e).__name__}",
                event="charge_error",
                interaction_id=interaction_id,
                user_hash=_hash_id(external_id),
                error_type=type(e).__name__,
            )

    def _aggregate_usage(
        self,
        interaction_id: str,
        usage_data: dict[str, Any],
    ) -> None:
        """Aggregate usage data in memory and update global stats."""
        model = usage_data["model"]
        prompt_tokens = usage_data["prompt_tokens"]
        completion_tokens = usage_data["completion_tokens"]
        cost_dollars = usage_data["cost_dollars"]
        duration_ms = usage_data["duration_ms"]

        # Update interaction-level stats
        interaction_usage = self._interaction_usage[interaction_id]
        interaction_usage["llm_calls"] += 1
        interaction_usage["prompt_tokens"] += prompt_tokens
        interaction_usage["completion_tokens"] += completion_tokens
        interaction_usage["models"].add(model)
        interaction_usage["cost_cents"] += cost_dollars * 100
        interaction_usage["duration_ms"] += duration_ms

        # Update global aggregate stats
        _global_usage["total_llm_calls"] += 1
        _global_usage["total_prompt_tokens"] += prompt_tokens
        _global_usage["total_completion_tokens"] += completion_tokens
        _global_usage["total_cost_cents"] += cost_dollars * 100
        _global_usage["models_used"][model] += 1

        # Print aggregate stats every 10 calls
        if _global_usage["total_llm_calls"] % 10 == 0:
            print(
                f"[CIRIS USAGE] AGGREGATE: calls={_global_usage['total_llm_calls']} "
                f"tokens={_global_usage['total_prompt_tokens']}+{_global_usage['total_completion_tokens']}="
                f"{_global_usage['total_prompt_tokens'] + _global_usage['total_completion_tokens']} "
                f"cost=${_global_usage['total_cost_cents']/100:.4f} "
                f"interactions={_global_usage['total_interactions']}",
                flush=True
            )

    async def async_log_success_event(
        self,
        kwargs: dict[str, Any],
        response_obj: Any,
        start_time: datetime,
        end_time: datetime,
    ) -> None:
        """
        Post-success: Charge and log usage.

        Called after every successful LLM response. We call /charge on every
        request (idempotent) and log usage for analytics.
        """
        # Extract metadata
        oauth_provider, external_id, interaction_id, metadata, litellm_params = (
            self._extract_success_metadata(kwargs)
        )

        if not external_id or not interaction_id:
            return

        # Extract usage data
        usage_data = self._extract_usage_data(
            response_obj, kwargs, litellm_params, start_time, end_time
        )

        # Log the request
        self._log_llm_request(interaction_id, external_id, usage_data, metadata)

        # Charge the interaction (idempotent)
        await self._charge_interaction(
            oauth_provider, external_id, interaction_id, usage_data["model"]
        )

        # Aggregate usage stats
        self._aggregate_usage(interaction_id, usage_data)

        # Stream usage to billing service
        await self._stream_usage_to_billing(
            oauth_provider=oauth_provider,
            external_id=external_id,
            interaction_id=interaction_id,
            model=usage_data["model"],
            prompt_tokens=usage_data["prompt_tokens"],
            completion_tokens=usage_data["completion_tokens"],
            cost_cents=usage_data["cost_dollars"] * 100,
            duration_ms=usage_data["duration_ms"],
        )

        logger.debug(
            "interaction=%s calls=%d model=%s tokens=%d/%d",
            interaction_id,
            self._interaction_usage[interaction_id]["llm_calls"],
            usage_data["model"],
            self._interaction_usage[interaction_id]["prompt_tokens"],
            self._interaction_usage[interaction_id]["completion_tokens"],
        )

    async def async_log_failure_event(
        self,
        kwargs: dict[str, Any],
        response_obj: Any,
        start_time: datetime,
        end_time: datetime,
    ) -> None:
        """Track errors for usage logging."""
        litellm_params = kwargs.get("litellm_params", {})
        metadata = litellm_params.get("metadata", {})
        interaction_id = metadata.get("_ciris_interaction_id")

        if interaction_id and interaction_id in self._interaction_usage:
            self._interaction_usage[interaction_id]["errors"] += 1
            logger.debug("interaction=%s event=error", interaction_id)

            external_id = metadata.get("_ciris_external_id", "")
            # Extract exception details for debugging
            exception = kwargs.get("exception", None)
            error_msg = str(exception)[:200] if exception else "Unknown error"
            _ship_log(
                "ERROR",
                f"LLM call failed: {error_msg}",
                event="llm_error",
                interaction_id=interaction_id,
                user_hash=_hash_id(external_id) if external_id else None,
                model=kwargs.get("model", "unknown"),
                error=error_msg,
            )

    async def finalize_interaction(
        self,
        external_id: str,
        interaction_id: str,
        oauth_provider: str = "oauth:google",
    ) -> None:
        """
        Call this when interaction completes to log aggregated usage.

        Should be triggered by the on-device agent signaling completion.
        This could be via a dedicated endpoint or a special metadata flag.
        """
        usage_data = self._interaction_usage.pop(interaction_id, None)
        if not usage_data:
            return

        # Calculate total duration from start to now
        start_time = usage_data.get("start_time")
        if start_time:
            total_duration_ms = int(
                (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            )
        else:
            total_duration_ms = usage_data["duration_ms"]

        try:
            client = await self._get_client()
            resp = await client.post(
                f"{self.billing_url}/v1/billing/litellm/usage",
                json={
                    "oauth_provider": oauth_provider,
                    "external_id": external_id,
                    "interaction_id": interaction_id,
                    "total_llm_calls": usage_data["llm_calls"],
                    "total_prompt_tokens": usage_data["prompt_tokens"],
                    "total_completion_tokens": usage_data["completion_tokens"],
                    "models_used": list(usage_data["models"]),
                    "actual_cost_cents": int(usage_data["cost_cents"]),
                    "duration_ms": total_duration_ms,
                    "error_count": usage_data["errors"],
                    "fallback_count": usage_data["fallbacks"],
                },
            )

            if resp.status_code == 200:
                logger.info(
                    "interaction=%s usage_logged=true calls=%d tokens=%d cost_cents=%.2f",
                    interaction_id,
                    usage_data["llm_calls"],
                    usage_data["prompt_tokens"] + usage_data["completion_tokens"],
                    usage_data["cost_cents"],
                )
            else:
                logger.warning(
                    "interaction=%s usage_logged=false status=%d",
                    interaction_id,
                    resp.status_code,
                )

        except httpx.RequestError as e:
            logger.error("interaction=%s usage_log=error error_type=%s", interaction_id, type(e).__name__)

        # Clean up auth cache
        self._authorized_interactions.pop(interaction_id, None)

    async def cleanup_stale_interactions(self, max_age_seconds: int = 3600) -> None:
        """
        Clean up interactions that have been open for too long.

        Call periodically to prevent memory leaks from abandoned interactions.
        """
        now = datetime.now(timezone.utc)
        stale_ids = []

        for interaction_id, usage_data in self._interaction_usage.items():
            start_time = usage_data.get("start_time")
            if start_time:
                age_seconds = (now - start_time).total_seconds()
                if age_seconds > max_age_seconds:
                    stale_ids.append(interaction_id)

        for interaction_id in stale_ids:
            usage_data = self._interaction_usage.get(interaction_id, {})
            external_id = usage_data.get("external_id", "")

            logger.info("interaction=%s cleanup=stale age_limit=%ds", interaction_id, max_age_seconds)

            if external_id:
                await self.finalize_interaction(external_id, interaction_id)
            else:
                self._interaction_usage.pop(interaction_id, None)
                self._authorized_interactions.pop(interaction_id, None)

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None


# Instance for LiteLLM to load via config
# Reference in litellm_config.yaml as: "billing_callback.billing_callback_instance"
billing_callback_instance = CIRISBillingCallback()

# ═══════════════════════════════════════════════════════════════════════════════
# CALLBACK STATUS - For health check verification
# ═══════════════════════════════════════════════════════════════════════════════
import json

CALLBACK_STATUS_FILE = "/app/callback_status.json"

def _write_callback_status():
    """Write callback status file for health check verification."""
    status = {
        "billing_callback_loaded": True,
        "lens_shipper_loaded": _lens_shipper is not None,
        "initialized_at": datetime.now(timezone.utc).isoformat(),
        "billing_url": billing_callback_instance.billing_url,
    }
    try:
        with open(CALLBACK_STATUS_FILE, "w") as f:
            json.dump(status, f)
        logger.info("Callback status written to %s", CALLBACK_STATUS_FILE)
    except Exception as e:
        logger.warning("Failed to write callback status: %s", e)

def get_callback_status() -> dict:
    """Get current callback status for health checks."""
    return {
        "billing_callback_loaded": True,
        "lens_shipper_loaded": _lens_shipper is not None,
        "requests_processed": _global_usage["total_llm_calls"],
        "interactions_processed": _global_usage["total_interactions"],
        "billing_url": billing_callback_instance.billing_url,
        "uptime_seconds": (
            (datetime.now(timezone.utc) - _global_usage["start_time"]).total_seconds()
            if _global_usage["start_time"] else 0
        ),
    }

# Write status file on module load (proves callback was imported)
_write_callback_status()
