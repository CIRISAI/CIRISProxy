# CIRIS Proxy - Claude Context

## Project Overview

**CIRISProxy** is a LiteLLM-based proxy service that provides secure, credit-gated LLM access for CIRIS Agent mobile clients.

- **Domain**: llm.ciris.ai (planned)
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

## Credit Model - CRITICAL DECISION NEEDED

The on-device agent makes **multiple LLM calls per user interaction**:

```
User sends "Research climate change" →
  On-device agent processes →
    LLM call 1: Think about approach
    LLM call 2: Use search tool
    LLM call 3: Process search results
    LLM call 4: Use another tool
    ... (12-70 calls typical)
    LLM call N: Final response
→ User sees response
```

### Option A: 1 Credit = 1 LLM Request (Simple)
```
Each /chat/completions call = 1 credit
- Simple to implement (pre-check, post-charge per request)
- User pays for actual compute used
- Problem: Complex interaction = 50+ credits
- Problem: Users can't predict costs
```

### Option B: 1 Credit = 1 Interaction (User-Friendly)
```
All LLM calls for one interaction = 1 credit
- Requires interaction_id tracking
- Requires aggregation layer in CIRISProxy
- User sees predictable "1 credit per question"
- Problem: How to detect interaction boundaries?
```

### Option C: Credit Bundles (Hybrid)
```
1 credit = N LLM calls (e.g., 10 calls)
- Partial credits tracked
- Simpler than full aggregation
- Still somewhat predictable
```

### Recommended: Option A with Rate Limiting

For MVP, simplest is **1 credit = 1 LLM request**, but with:
1. Low-cost models only (Groq Llama-3.1-8B-instant is nearly free)
2. Generous credit bundles (100 credits = $0.99)
3. Rate limiting (max 100 requests/minute per user)
4. Clear pricing: "~50 credits per complex question"

The CIRISBilling `/litellm/auth`, `/litellm/charge`, `/litellm/usage` endpoints
already support this model - they just need to be called per-request.

## Integration with CIRISBilling

CIRISProxy calls these CIRISBilling endpoints **per LLM request**:

### 1. Pre-Request Authorization (Before proxying)
```
POST https://billing.ciris.ai/v1/billing/litellm/auth
{
    "oauth_provider": "oauth:google",
    "external_id": "user-google-id",       // From Authorization header
    "model": "groq/llama-3.1-70b"           // From request body
}

Response:
{
    "authorized": true,
    "credits_remaining": 10
}

If authorized=false, return 402 Payment Required to client.
```

### 2. Post-Request Charge (After successful response)
```
POST https://billing.ciris.ai/v1/billing/litellm/charge
{
    "oauth_provider": "oauth:google",
    "external_id": "user-google-id",
    "interaction_id": "<unique-request-id>",  // Generated per request
    "idempotency_key": "<litellm-call-id>"    // Prevent double-charging on retry
}

Response:
{
    "charged": true,
    "credits_deducted": 1,
    "credits_remaining": 9,
    "charge_id": "uuid"
}
```

### 3. Usage Analytics (Optional, for margin tracking)
```
POST https://billing.ciris.ai/v1/billing/litellm/usage
{
    "oauth_provider": "oauth:google",
    "external_id": "user-google-id",
    "interaction_id": "<request-id>",
    "total_llm_calls": 1,
    "total_prompt_tokens": 1500,
    "total_completion_tokens": 200,
    "models_used": ["groq/llama-3.1-70b"],
    "actual_cost_cents": 0,                    // Groq is free tier
    "duration_ms": 850
}
```

## Implementation Requirements

### LiteLLM Custom Callback

CIRISProxy uses LiteLLM's `CustomLogger` for billing integration:

```python
# hooks/billing_callback.py
from litellm.integrations.custom_logger import CustomLogger
import httpx
import os
from uuid import uuid4

class CIRISBillingCallback(CustomLogger):
    def __init__(self):
        self.billing_url = os.environ.get("BILLING_API_URL", "https://billing.ciris.ai")
        self.billing_key = os.environ.get("BILLING_API_KEY")

    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        """Pre-request: Verify user has credits."""
        # Extract Google user ID from Authorization header
        # Format: "Bearer google:<user-id>" or just the user ID
        user_key = user_api_key_dict.get("api_key", "")
        if user_key.startswith("google:"):
            external_id = user_key.split(":", 1)[1]
        else:
            external_id = user_key

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self.billing_url}/v1/billing/litellm/auth",
                json={
                    "oauth_provider": "oauth:google",
                    "external_id": external_id,
                    "model": data.get("model")
                },
                headers={"Authorization": f"Bearer {self.billing_key}"},
                timeout=5.0
            )
            result = resp.json()

        if not result.get("authorized"):
            raise Exception(f"Insufficient credits: {result.get('reason', 'No credits')}")

        # Store external_id for post-call
        data["_ciris_external_id"] = external_id
        data["_ciris_request_id"] = str(uuid4())

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        """Post-success: Charge 1 credit."""
        external_id = kwargs.get("litellm_params", {}).get("metadata", {}).get("_ciris_external_id")
        request_id = kwargs.get("litellm_params", {}).get("metadata", {}).get("_ciris_request_id")

        if not external_id:
            return  # Can't charge without user ID

        async with httpx.AsyncClient() as client:
            # Charge the credit
            await client.post(
                f"{self.billing_url}/v1/billing/litellm/charge",
                json={
                    "oauth_provider": "oauth:google",
                    "external_id": external_id,
                    "interaction_id": request_id,
                    "idempotency_key": kwargs.get("litellm_call_id")
                },
                headers={"Authorization": f"Bearer {self.billing_key}"},
                timeout=5.0
            )

            # Log usage for analytics
            usage = getattr(response_obj, "usage", None)
            if usage:
                duration_ms = int((end_time - start_time).total_seconds() * 1000)
                await client.post(
                    f"{self.billing_url}/v1/billing/litellm/usage",
                    json={
                        "oauth_provider": "oauth:google",
                        "external_id": external_id,
                        "interaction_id": request_id,
                        "total_llm_calls": 1,
                        "total_prompt_tokens": getattr(usage, "prompt_tokens", 0),
                        "total_completion_tokens": getattr(usage, "completion_tokens", 0),
                        "models_used": [kwargs.get("model", "unknown")],
                        "actual_cost_cents": int(response_obj._hidden_params.get("response_cost", 0) * 100),
                        "duration_ms": duration_ms
                    },
                    headers={"Authorization": f"Bearer {self.billing_key}"},
                    timeout=5.0
                )
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

## Related Repositories

| Repository | Purpose | URL |
|------------|---------|-----|
| CIRISAgent | Core AI platform + Android app | agents.ciris.ai |
| CIRISBilling | Credit gating + Google Play | billing.ciris.ai |
| CIRISProxy | LLM proxy (this repo) | llm.ciris.ai (planned) |

## Project Structure

```
CIRISProxy/
├── CLAUDE.md               # This file
├── docker-compose.yml      # LiteLLM proxy config
├── litellm_config.yaml     # Model routing (Groq, Together, etc.)
├── hooks/
│   ├── __init__.py
│   └── billing_callback.py # CIRISBilling integration
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
