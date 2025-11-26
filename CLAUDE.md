# CIRIS Proxy - Claude Context

## Project Overview

**CIRISProxy** is a LiteLLM-based proxy service that provides secure LLM access for CIRIS Agent mobile clients.

- **Domain**: llm.ciris.ai (planned)
- **Tech Stack**: LiteLLM, FastAPI, Docker
- **Purpose**: Secure API key isolation from mobile clients

## Architecture Context

CIRISProxy is part of the CIRIS ecosystem:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CIRIS Ecosystem                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────────┐                                                       │
│  │   CIRISAgent     │  Core AI platform - Discord + API at agents.ciris.ai │
│  │   (Main Repo)    │  22 services, 6 buses, multi-occurrence support      │
│  └────────┬─────────┘                                                       │
│           │                                                                 │
│           │ OAuth identity (oauth:google)                                   │
│           ▼                                                                 │
│  ┌──────────────────┐                                                       │
│  │  CIRISBilling    │  Credit gating service at billing.ciris.ai           │
│  │  (This context)  │  Credit management, usage tracking, analytics        │
│  └────────┬─────────┘                                                       │
│           │                                                                 │
│           │ Pre-auth check, post-charge, usage logging                      │
│           ▼                                                                 │
│  ┌──────────────────┐                                                       │
│  │   CIRISProxy     │  LiteLLM proxy at llm.ciris.ai                       │
│  │   (THIS REPO)    │  Holds Groq/Together API keys securely               │
│  └────────┬─────────┘                                                       │
│           │                                                                 │
│           │ Proxied LLM requests                                            │
│           ▼                                                                 │
│  ┌──────────────────┐                                                       │
│  │  LLM Providers   │  Groq, Together, OpenAI, Anthropic, etc.             │
│  └──────────────────┘                                                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Why CIRISProxy Exists

**Problem**: Mobile apps cannot securely store API keys. Embedding Groq/Together/OpenAI keys in APKs is:
1. Insecure - Can be extracted via APK decompilation
2. ToS violation - Most LLM providers prohibit client-side key exposure
3. Uncontrollable - No way to revoke or rate-limit compromised keys

**Solution**: CIRISProxy holds all LLM provider keys server-side. Mobile clients authenticate via OAuth through CIRISBilling, which authorizes LLM usage via credits.

## Credit Model

```
1 Credit = 1 Interaction (user message → agent response)

An interaction may involve:
- 12-70 LLM calls (tool use, reasoning, etc.)
- Multiple model fallbacks
- Various providers (Groq, Together, etc.)

User sees: "1 credit used"
Backend sees: Detailed analytics for margin monitoring
```

## Integration with CIRISBilling

CIRISProxy must integrate with these CIRISBilling endpoints:

### Pre-Request Authorization
```
POST https://billing.ciris.ai/v1/billing/litellm/auth
{
    "oauth_provider": "oauth:google",
    "external_id": "user-google-id",
    "model": "groq/llama-3.1-70b",
    "interaction_id": "int-uuid"
}

Response:
{
    "authorized": true,
    "credits_remaining": 10,
    "interaction_id": "int-uuid"
}
```

### Post-Interaction Charge
```
POST https://billing.ciris.ai/v1/billing/litellm/charge
{
    "oauth_provider": "oauth:google",
    "external_id": "user-google-id",
    "interaction_id": "int-uuid",
    "idempotency_key": "charge-uuid"
}

Response:
{
    "charged": true,
    "credits_deducted": 1,
    "credits_remaining": 9,
    "charge_id": "uuid"
}
```

### Usage Analytics Logging
```
POST https://billing.ciris.ai/v1/billing/litellm/usage
{
    "oauth_provider": "oauth:google",
    "external_id": "user-google-id",
    "interaction_id": "int-uuid",
    "total_llm_calls": 45,
    "total_prompt_tokens": 80000,
    "total_completion_tokens": 5000,
    "models_used": ["groq/llama-3.1-70b", "together/mixtral"],
    "actual_cost_cents": 12,
    "duration_ms": 8500,
    "error_count": 0,
    "fallback_count": 2
}

Response:
{
    "logged": true,
    "usage_log_id": "uuid"
}
```

## Implementation Requirements

### LiteLLM Configuration

CIRISProxy should:
1. Use LiteLLM as the core proxy
2. Configure pre-request hooks for billing auth
3. Configure post-completion hooks for charging
4. Aggregate usage metrics per interaction
5. Support model fallback chains

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

## Related Repositories

| Repository | Purpose | URL |
|------------|---------|-----|
| CIRISAgent | Core AI platform | agents.ciris.ai |
| CIRISBilling | Credit gating service | billing.ciris.ai |
| CIRISProxy | LLM proxy (this repo) | llm.ciris.ai (planned) |

## Development Notes

### Testing Locally

```bash
# Start proxy (requires billing service)
docker-compose up -d

# Test auth flow
curl -X POST http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer user-oauth-token" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "groq/llama-3.1-70b",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

### Key Files (to be created)

```
CIRISProxy/
├── docker-compose.yml       # LiteLLM + proxy config
├── litellm_config.yaml     # Model routing configuration
├── hooks/
│   ├── pre_request.py      # Billing auth check
│   └── post_completion.py  # Charge and usage logging
├── Dockerfile
└── .env.example
```

## Quality Standards

- **Type Safety**: Use Pydantic models for all data structures
- **Error Handling**: Graceful degradation when billing service is unavailable
- **Observability**: Structured logging, metrics for monitoring
- **Security**: No secrets in code, proper key rotation support
