# CIRIS Proxy - Claude Context

## Project Overview

**CIRISProxy** is a LiteLLM-based proxy service that provides secure, credit-gated LLM access for CIRIS Agent mobile clients.

- **Domain**: llm.ciris.ai
- **Tech Stack**: LiteLLM, FastAPI, Docker
- **Purpose**: Secure API key isolation + credit-based access control

## Current Android Architecture (Important Context)

The Android app runs **100% on-device** except for LLM inference:

```
┌──────────────────────────────────────────────────────────────────────┐
│                        Android Device                                 │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │              CIRIS Agent (100% On-Device)                      │  │
│  │                                                                │  │
│  │  - Python 3.10+ runtime (Chaquopy)                             │  │
│  │  - Complete CIRIS codebase (22 services, 6 buses)              │  │
│  │  - FastAPI server @ localhost:8000                             │  │
│  │  - Web UI in WebView                                           │  │
│  │  - SQLite database                                             │  │
│  │  - All business logic, tools, memory                           │  │
│  │                                                                │  │
│  │  LLM calls via httpx to:                                       │  │
│  │    OPENAI_API_BASE (user-configured)                           │  │
│  │    OPENAI_API_KEY (user-configured)                            │  │
│  │                                                                │  │
│  └─────────────────────────┬──────────────────────────────────────┘  │
│                            │                                          │
│                            │ LLM Requests                             │
│                            ▼                                          │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │              Google Play Billing                               │  │
│  │  - BillingManager.kt (purchase flow)                           │  │
│  │  - BillingApiClient.kt → CIRISBilling /google-play/verify      │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────────┐
│                        Remote Services                                │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  CURRENT: User-configured LLM endpoint                               │
│  - User enters their own API key in Settings                          │
│  - No credit gating on LLM usage                                      │
│  - Credits only used for Google Play purchases                        │
│                                                                       │
│  WITH CIRISProxy:                                                     │
│  - User sets OPENAI_API_BASE = https://llm.ciris.ai                   │
│  - User sets OPENAI_API_KEY = their Google OAuth token                │
│  - CIRISProxy verifies credits before proxying                        │
│  - CIRISProxy charges credits after successful response               │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
```

## Why CIRISProxy Exists

**Current Problem**:
1. Users must enter their own API keys in Settings
2. Keys are stored in SharedPreferences (encrypted but still on-device)
3. No credit gating on actual LLM usage - credits are decorative
4. ToS violation risk if users share/lose their devices

**Solution**: CIRISProxy replaces user-managed API keys:
1. User configures `llm.ciris.ai` as their LLM endpoint
2. User authenticates with Google OAuth token
3. CIRISProxy holds Groq/Together keys server-side
4. Credits are actually enforced per LLM request

## Credit Model: 1 Credit = 1 Interaction

**CIRISBilling already supports this!** The key is `idempotency_key` on the charge endpoint.

The on-device agent makes **multiple LLM calls per user interaction**:

```
User sends "Research climate change" →
  On-device agent processes →
    LLM call 1: Think about approach        ─┐
    LLM call 2: Use search tool              │
    LLM call 3: Process search results       │ ALL share same interaction_id
    LLM call 4: Use another tool             │ Only FIRST charge succeeds
    ... (12-70 calls typical)                │ Rest are idempotent no-ops
    LLM call N: Final response              ─┘
→ User sees response
→ User is charged exactly 1 credit
```

### How It Works

1. **Android app generates `interaction_id`** when user sends a message
2. **Every LLM call includes the same `interaction_id`** in metadata
3. **CIRISProxy calls `/litellm/charge`** with `idempotency_key = "litellm:{interaction_id}"`
4. **First call succeeds**, deducts 1 credit (100 minor units)
5. **Subsequent calls with same interaction_id** return success but don't charge again

From `CIRISBilling/app/api/routes.py:1224`:
```python
# Default idempotency key ensures same interaction = one charge
idempotency_key = request.idempotency_key or f"litellm:{request.interaction_id}"
```

From `routes.py:1260-1267` - IdempotencyConflictError returns success:
```python
except IdempotencyConflictError as exc:
    # Already charged - return success (idempotent)
    return LiteLLMChargeResponse(
        charged=True,
        credits_deducted=1,
        credits_remaining=0,  # Unknown, but charge was successful
        charge_id=exc.existing_id,
    )
```

### Implementation in CIRISProxy

```python
# Pre-call: Auth check (can be cached per interaction)
# Only call once per interaction, not per LLM request
if interaction_id not in self._authorized_interactions:
    auth_result = await self._check_auth(external_id, interaction_id)
    if auth_result.authorized:
        self._authorized_interactions[interaction_id] = True
    else:
        raise Exception("Insufficient credits")

# Post-call: Charge (idempotent - safe to call multiple times)
await self._charge(external_id, interaction_id)
# First call charges 1 credit, subsequent calls are no-ops
```

### Android App Integration

The Android app needs to:
1. Generate `interaction_id` (UUID) when user sends a message
2. Include it in metadata for all LLM calls in that interaction
3. Clear it when interaction completes

```kotlin
// In mobile_main.py or the LLM client layer
class LLMClient:
    def __init__(self):
        self._current_interaction_id: str | None = None

    def start_interaction(self) -> str:
        """Called when user sends a message."""
        self._current_interaction_id = str(uuid4())
        return self._current_interaction_id

    def get_interaction_id(self) -> str:
        """Include in every LLM call metadata."""
        if not self._current_interaction_id:
            self._current_interaction_id = str(uuid4())
        return self._current_interaction_id

    def end_interaction(self):
        """Called when agent response is complete."""
        self._current_interaction_id = None
```

## Integration with CIRISBilling

CIRISProxy calls these endpoints with **idempotent per-interaction charging**:

### 1. Pre-Request Authorization (Once per interaction)
```
POST https://billing.ciris.ai/v1/billing/litellm/auth
{
    "oauth_provider": "oauth:google",
    "external_id": "user-google-id",
    "model": "groq/llama-3.1-70b",
    "interaction_id": "int-uuid-456"         // Same for all calls in interaction
}

Response:
{
    "authorized": true,
    "credits_remaining": 10,
    "interaction_id": "int-uuid-456"
}
```

Cache this result in CIRISProxy for the duration of the interaction.
Only check once when `interaction_id` is first seen.

### 2. Post-Request Charge (Idempotent per interaction)
```
POST https://billing.ciris.ai/v1/billing/litellm/charge
{
    "oauth_provider": "oauth:google",
    "external_id": "user-google-id",
    "interaction_id": "int-uuid-456"         // SAME for all calls!
    // idempotency_key defaults to "litellm:{interaction_id}"
}

First call response:
{
    "charged": true,
    "credits_deducted": 1,
    "credits_remaining": 9,
    "charge_id": "uuid"
}

Subsequent calls (same interaction_id) response:
{
    "charged": true,                         // Still returns true!
    "credits_deducted": 1,
    "credits_remaining": 0,                  // Unknown but charge exists
    "charge_id": "existing-uuid"             // Returns existing charge
}
```

**Key insight:** Call charge on EVERY LLM request. The idempotency_key ensures
only the first call actually deducts credits. This is simpler than tracking
"first call" vs "subsequent calls" in CIRISProxy.

### 3. Usage Analytics (Aggregate per interaction)
```
POST https://billing.ciris.ai/v1/billing/litellm/usage
{
    "oauth_provider": "oauth:google",
    "external_id": "user-google-id",
    "interaction_id": "int-uuid-456",
    "total_llm_calls": 45,                   // Total calls in this interaction
    "total_prompt_tokens": 80000,
    "total_completion_tokens": 5000,
    "models_used": ["groq/llama-3.1-70b", "together/mixtral"],
    "actual_cost_cents": 12,
    "duration_ms": 8500,
    "error_count": 2,
    "fallback_count": 3
}
```

Call this ONCE at the end of the interaction with aggregated totals.
This gives you margin analytics (user paid 1 credit, you paid 12 cents).

## Implementation Requirements

### LiteLLM Custom Callback

CIRISProxy uses LiteLLM's `CustomLogger` for billing integration:

```python
# hooks/billing_callback.py
from litellm.integrations.custom_logger import CustomLogger
import httpx
import os
from collections import defaultdict

class CIRISBillingCallback(CustomLogger):
    """
    LiteLLM callback that integrates with CIRISBilling.

    Key design: 1 credit = 1 interaction (not 1 LLM call)
    Uses interaction_id + idempotency to ensure single charge per interaction.
    """

    def __init__(self):
        self.billing_url = os.environ.get("BILLING_API_URL", "https://billing.ciris.ai")
        self.billing_key = os.environ.get("BILLING_API_KEY")

        # Cache authorized interactions to avoid repeated auth checks
        self._authorized_interactions: dict[str, bool] = {}

        # Aggregate usage per interaction for final logging
        self._interaction_usage: dict[str, dict] = defaultdict(lambda: {
            "llm_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "models": set(),
            "cost_cents": 0,
            "duration_ms": 0,
            "errors": 0,
            "fallbacks": 0,
        })

    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        """Pre-request: Verify user has credits (cached per interaction)."""

        # Extract user identity
        user_key = user_api_key_dict.get("api_key", "")
        if user_key.startswith("google:"):
            external_id = user_key.split(":", 1)[1]
        else:
            external_id = user_key

        # Get interaction_id from request metadata (set by on-device agent)
        metadata = data.get("metadata", {})
        interaction_id = metadata.get("interaction_id")

        if not interaction_id:
            raise Exception("Missing interaction_id in request metadata")

        # Store for post-call
        data["_ciris_external_id"] = external_id
        data["_ciris_interaction_id"] = interaction_id

        # Check if already authorized for this interaction (cache hit)
        if interaction_id in self._authorized_interactions:
            return  # Already checked, proceed

        # First LLM call for this interaction - check auth
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self.billing_url}/v1/billing/litellm/auth",
                json={
                    "oauth_provider": "oauth:google",
                    "external_id": external_id,
                    "model": data.get("model"),
                    "interaction_id": interaction_id,
                },
                headers={"Authorization": f"Bearer {self.billing_key}"},
                timeout=5.0
            )
            result = resp.json()

        if not result.get("authorized"):
            raise Exception(f"Insufficient credits: {result.get('reason', 'No credits')}")

        # Cache authorization for this interaction
        self._authorized_interactions[interaction_id] = True

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        """Post-success: Charge and aggregate usage."""

        metadata = kwargs.get("litellm_params", {}).get("metadata", {})
        external_id = metadata.get("_ciris_external_id")
        interaction_id = metadata.get("_ciris_interaction_id")

        if not external_id or not interaction_id:
            return

        # Charge (idempotent - first call charges, rest are no-ops)
        async with httpx.AsyncClient() as client:
            await client.post(
                f"{self.billing_url}/v1/billing/litellm/charge",
                json={
                    "oauth_provider": "oauth:google",
                    "external_id": external_id,
                    "interaction_id": interaction_id,
                    # idempotency_key defaults to "litellm:{interaction_id}"
                },
                headers={"Authorization": f"Bearer {self.billing_key}"},
                timeout=5.0
            )

        # Aggregate usage for this interaction
        usage_data = self._interaction_usage[interaction_id]
        usage_data["llm_calls"] += 1

        usage = getattr(response_obj, "usage", None)
        if usage:
            usage_data["prompt_tokens"] += getattr(usage, "prompt_tokens", 0)
            usage_data["completion_tokens"] += getattr(usage, "completion_tokens", 0)

        usage_data["models"].add(kwargs.get("model", "unknown"))
        usage_data["cost_cents"] += int(
            response_obj._hidden_params.get("response_cost", 0) * 100
        )
        usage_data["duration_ms"] += int(
            (end_time - start_time).total_seconds() * 1000
        )

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        """Track errors for usage logging."""
        metadata = kwargs.get("litellm_params", {}).get("metadata", {})
        interaction_id = metadata.get("_ciris_interaction_id")

        if interaction_id:
            self._interaction_usage[interaction_id]["errors"] += 1

    async def finalize_interaction(self, external_id: str, interaction_id: str):
        """
        Call this when interaction completes to log aggregated usage.
        Should be triggered by the on-device agent signaling completion.
        """
        usage_data = self._interaction_usage.pop(interaction_id, None)
        if not usage_data:
            return

        async with httpx.AsyncClient() as client:
            await client.post(
                f"{self.billing_url}/v1/billing/litellm/usage",
                json={
                    "oauth_provider": "oauth:google",
                    "external_id": external_id,
                    "interaction_id": interaction_id,
                    "total_llm_calls": usage_data["llm_calls"],
                    "total_prompt_tokens": usage_data["prompt_tokens"],
                    "total_completion_tokens": usage_data["completion_tokens"],
                    "models_used": list(usage_data["models"]),
                    "actual_cost_cents": usage_data["cost_cents"],
                    "duration_ms": usage_data["duration_ms"],
                    "error_count": usage_data["errors"],
                    "fallback_count": usage_data["fallbacks"],
                },
                headers={"Authorization": f"Bearer {self.billing_key}"},
                timeout=5.0
            )

        # Clean up auth cache
        self._authorized_interactions.pop(interaction_id, None)
```

### Environment Variables

```bash
# LLM Provider Keys (server-side only, NEVER exposed to clients)
GROQ_API_KEY=gsk_...
TOGETHER_API_KEY=...
OPENAI_API_KEY=sk-...  # Optional fallback

# Billing Integration
BILLING_API_URL=https://billing.ciris.ai
BILLING_API_KEY=...  # Service-to-service auth

# LiteLLM Config
LITELLM_MASTER_KEY=...
LITELLM_MODEL_LIST=...
```

### Security Requirements

1. **No API keys exposed to clients** - All LLM keys are server-side only
2. **OAuth verification** - Validate user identity through billing service
3. **Rate limiting** - Per-user rate limits in addition to credit limits
4. **Audit logging** - Track all requests for security analysis

## Android App Changes Required

To use CIRISProxy, the Android app needs minor updates:

### SettingsActivity.kt Changes
```kotlin
// Current: User enters any OpenAI-compatible endpoint + their key
// New: Pre-fill with llm.ciris.ai and use Google user ID as key

companion object {
    private const val DEFAULT_API_BASE = "https://llm.ciris.ai/v1"
}

private fun loadSettings() {
    val prefs = getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

    // Default to CIRISProxy
    apiBaseInput.setText(
        prefs.getString(KEY_API_BASE, DEFAULT_API_BASE)
    )

    // Use Google user ID as the API key
    val googleUserId = prefs.getString(KEY_GOOGLE_USER_ID, "")
    apiKeyInput.setText("google:$googleUserId")
}
```

### Required: Google Sign-In Integration
The app needs to:
1. Implement Google Sign-In to get user ID
2. Store the Google user ID in SharedPreferences
3. Pass it as the API key to CIRISProxy

This is already partially implemented in `BillingApiClient.kt`:
- `getGoogleUserId()` / `setGoogleUserId()` exist
- Just need to wire up actual Google Sign-In flow

## CIRISLens Observability Integration

CIRISProxy ships logs to CIRISLens for centralized observability across all CIRIS services.

### Setup

1. Generate a service token at https://agents.ciris.ai/lens/admin/
2. Set the environment variable: `CIRISLENS_TOKEN=svc_xxx`
3. Optionally configure endpoint: `CIRISLENS_ENDPOINT=https://agents.ciris.ai/lens/api/v1/logs/ingest`

### Events Logged

| Event | Level | Description |
|-------|-------|-------------|
| `auth_granted` | INFO | User authorized with available credits |
| `auth_denied` | WARNING | User denied due to insufficient credits |
| `charge_created` | INFO | Credit deducted for new interaction |
| `charge_failed` | WARNING | Charge request failed (non-network) |
| `charge_error` | ERROR | Charge request network error |
| `billing_error` | ERROR | Billing service unreachable |
| `llm_error` | ERROR | LLM call failed |

### Log Format

All logs include:
- `interaction_id`: Links all calls in a user interaction
- `user_hash`: SHA-256 hash of user ID (first 8 chars only)
- `event`: Structured event type for filtering

### LogShipper SDK

The `sdk/logshipper.py` module provides batched log shipping with:
- Background thread for async flushing
- Retry with exponential backoff
- Thread-safe buffer management
- Graceful shutdown

```python
from sdk.logshipper import LogShipper

shipper = LogShipper(
    service_name="cirisproxy",
    token=os.environ["CIRISLENS_TOKEN"],
)

shipper.info("Request processed", event="request_completed", model="groq/llama-3.1-70b")
```

## Related Repositories

| Repository | Purpose | URL |
|------------|---------|-----|
| CIRISAgent | Core AI platform + Android app | agents.ciris.ai |
| CIRISBilling | Credit gating + Google Play | billing.ciris.ai |
| CIRISProxy | LLM proxy (this repo) | llm.ciris.ai |
| CIRISLens | Observability + dashboards | agents.ciris.ai/lens/ |

## Project Structure

```
CIRISProxy/
├── CLAUDE.md               # This file
├── docker-compose.yml      # LiteLLM proxy config
├── litellm_config.yaml     # Model routing (Groq, Together, etc.)
├── hooks/
│   ├── __init__.py
│   └── billing_callback.py # CIRISBilling + CIRISLens integration
├── sdk/
│   ├── __init__.py
│   └── logshipper.py       # CIRISLens log shipping SDK
├── Dockerfile              # Custom build if needed
├── .env.example            # Environment template
└── .github/
    └── workflows/
        └── deploy.yml      # CD pipeline
```

## Testing Locally

```bash
# Start locally (requires billing service running)
docker-compose up -d

# Test with a Google user ID
curl -X POST http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer google:test-user-id" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "groq/llama-3.1-8b",
    "messages": [{"role": "user", "content": "Hello"}]
  }'

# Expected: 402 if no credits, or proxied response if credits exist
```

## Quality Standards

- **Error Handling**: Return 402 Payment Required when no credits
- **Graceful Degradation**: If billing service is down, deny by default (fail closed)
- **Observability**: Log all auth failures for debugging
- **Security**: No LLM provider keys exposed to clients ever
