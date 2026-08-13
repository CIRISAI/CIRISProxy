# CIRISProxy

Secure, credit-gated LLM proxy for CIRIS Agent mobile clients.

## What is CIRISProxy?

CIRISProxy is **temporary bridging infrastructure** that enables CIRIS ethical agents to access LLM providers while the [Veilid](https://veilid.com/) decentralized network matures. It is designed to be retired.

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  CIRIS Agent    │────▶│   CIRISProxy    │────▶│  LLM Providers  │
│  (Mobile App)   │     │  (This Service) │     │  Groq/Together  │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        │                       │
        │                       ▼
        │               ┌─────────────────┐
        └──────────────▶│  CIRISBilling   │
                        │  (Credit Mgmt)  │
                        └─────────────────┘
```

### Key Features

| Feature | Description |
|---------|-------------|
| **API Key Isolation** | Provider keys (Groq, Together, OpenRouter) stored server-side only |
| **Credit Gating** | Users authenticate via Google OAuth; credits enforced per-interaction |
| **Per-Interaction Billing** | 1 credit = 1 user interaction (12-70 LLM calls typical) |
| **Provider Routing** | Multi-provider fallback with configurable provider exclusions |
| **ZDR Web Search** | Exa AI primary (Zero Data Retention), Brave fallback |
| **Test Auth Mode** | Integration testing without Google OAuth infrastructure |
| **Observability** | Structured logging to CIRISLens for audit trails |

### Safety & Privacy

- **Zero conversation retention** - No message content stored
- **Fail-closed billing** - Deny access if billing service unavailable
- **Anonymized logging** - Only interaction IDs and event types, no PII
- **Tamper-evident audit** - Structured events for accountability

See [ciris.ai/safety](https://ciris.ai/safety) for the full safety framework.

## Quick Start

```bash
# Clone and configure
git clone https://github.com/CIRISAI/CIRISProxy.git
cd CIRISProxy
cp .env.example .env
# Edit .env with your API keys

# Start the proxy
docker-compose up -d

# Test health
curl http://localhost:4000/health/liveliness

# Test with a request
curl -X POST http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer google:your-user-id" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "groq/llama-3.1-8b",
    "messages": [{"role": "user", "content": "Hello"}],
    "metadata": {"interaction_id": "test-123"}
  }'
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `DEEPINFRA_API_KEY` | Yes | DeepInfra API key (primary provider) |
| `OPENROUTER_API_KEY` | Yes | OpenRouter API key (first fallback) |
| `GROQ_API_KEY` | Yes | Groq API key (second fallback, `fast` alias) |
| `TOGETHER_API_KEY` | No | Together AI API key |
| `EXA_API_KEY` | Yes* | Exa AI API key (ZDR-compliant search) |
| `BRAVE_API_KEY` | No | Brave Search API key (fallback) |
| `SEARCH_PROVIDER` | No | `auto` (default), `exa`, or `brave` |
| `BILLING_API_URL` | Yes | CIRISBilling endpoint |
| `BILLING_API_KEY` | Yes | Service-to-service auth key |
| `GOOGLE_CLIENT_ID` | Yes | Google OAuth client ID for token verification |
| `CIRISLENS_TOKEN` | No | Token for log shipping to CIRISLens |
| `OPENROUTER_IGNORE_PROVIDERS` | No | Comma-separated providers to exclude (e.g., "Friendli,Google") |
| `CIRIS_TEST_AUTH_ENABLED` | No | Enable test auth mode (`true` to enable) |
| `CIRIS_TEST_USER_ID` | No | Test user ID (default: `ciris_synthetic_canary`) |

*Either `EXA_API_KEY` or `BRAVE_API_KEY` required for web search functionality.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | OpenAI-compatible chat endpoint |
| `/v1/status` | GET | Provider health status |
| `/v1/status/simple` | GET | Simple liveness check |
| `/v1/web/search` | POST | Web search (requires credits) |
| `/health/liveliness` | GET | Container health check |

### Reading `/v1/status`

`status` is the **service's own** health: it covers the dependencies nothing
else can serve (billing), plus any provider pool that has dropped below
`min_available`. LLM providers are pooled — the router has alternatives — so a
single provider being slow or down shows up in `providers[]` and `pools`
without degrading the service. It is not impairment while there is somewhere
to route.

```jsonc
{
  "status": "operational",          // the service, not the pool members
  "providers": [
    { "provider": "deepinfra", "role": "primary",  "status": "operational",
      "models": { "configured": 2, "available": 2, "missing": [] } },
    { "provider": "groq",      "role": "fallback", "status": "degraded", "...": "..." }
  ],
  "pools": {
    "llm": { "status": "operational", "available": 3, "min_available": 1,
             "primary_available": true }
  }
}
```

`primary_available: false` with an operational pool means **we are serving on a
fallback** — worth knowing, since it precedes cost, latency and quality
changes, but not an outage.

Each provider's `models` block compares the models configured in
`litellm_config.yaml` against that provider's own `/models` listing, using the
response the health check already fetched. A provider that is reachable but no
longer serves a model we route to it reports `degraded` with the missing model
named. See [Model availability](#model-availability).

## Test Auth Mode

For integration testing without Google OAuth infrastructure, CIRISProxy supports a test auth mode that validates opaque tokens via CIRISBilling.

### Setup

**1. Generate a test token:**
```bash
openssl rand -hex 32
# Example: c6d7c30dd742f4424c5a214cf5a6bd23838ad40bac177634b5667c1811f1814b
```

**2. Configure CIRISProxy:**
```bash
CIRIS_TEST_AUTH_ENABLED=true
CIRIS_TEST_USER_ID=ciris_synthetic_canary
BILLING_API_URL=http://billing:8000
```

**3. Configure CIRISBilling** (must also have test auth enabled):
```bash
CIRIS_TEST_AUTH_ENABLED=True
CIRIS_TEST_AUTH_TOKEN=c6d7c30dd742f4424c5a214cf5a6bd23838ad40bac177634b5667c1811f1814b
CIRIS_TEST_USER_ID=ciris_synthetic_canary
```

### Usage

```bash
# Make authenticated requests with test token
curl -X POST http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer c6d7c30dd742f4424c5a214cf5a6bd23838ad40bac177634b5667c1811f1814b" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "Hello"}],
    "metadata": {"interaction_id": "test-123"}
  }'
```

### How It Works

1. Proxy detects non-JWT tokens (opaque hex strings ≥32 chars)
2. Calls CIRISBilling `/v1/billing/credits/check` to validate
3. If billing returns `has_credit: true`, request is authorized
4. User identity becomes `test:{user_id}` for billing tracking

### Security

- **Never enable in production** - bypasses Google OAuth entirely
- Test users get standard credit limits (10 free, 2 daily)
- All test token usage logged at WARNING level
- CIRISBilling refuses to start if enabled with `ENVIRONMENT=production`

## Development

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run with coverage
pytest --cov=hooks --cov=sdk --cov=server --cov-report=term-missing

# Lint
ruff check .

# Audit configured models against what providers actually serve
python -m hooks.verify_models
```

### Model availability

Providers retire models without notice, and a config that was correct when
written keeps naming them. Groq dropped `llama-4-scout` and `llama-4-maverick`
while both were still in the routing chain — including the last hop of the
default chain, which would only have failed during a primary outage, when the
fallback was actually needed.

`python -m hooks.verify_models` compares every model in `litellm_config.yaml`
against its provider's `/models` listing, and checks that no fallback chain
names an alias that is not configured. It queries only free listing endpoints,
never inference. Providers with no key in the environment report `skipped`
rather than failing, so it is useful in a keyless checkout.

It runs in three places:

| Where | When | Purpose |
|-------|------|---------|
| `config-audit` job in CI | every PR | catches a bad edit before merge |
| `model-availability.yml` | daily 07:00 UTC | catches a retirement upstream; files an issue |
| `/v1/status` | every health check | reports it live, per provider, at no extra cost |

**Any change to the routing chain in `litellm_config.yaml` must be mirrored in
`PROVIDERS` in `hooks/status_handler.py`** — monitoring the fallbacks while the
primary goes unchecked renders a green board during a primary outage.

### Project Structure

```
CIRISProxy/
├── hooks/
│   ├── billing_callback.py   # LiteLLM callback for billing integration
│   ├── custom_auth.py        # Google OAuth + test token verification
│   ├── model_audit.py        # Configured models vs. provider /models listings
│   ├── search_handler.py     # Web search (Exa/Brave)
│   ├── status_handler.py     # Provider health monitoring + pool rollup
│   └── verify_models.py      # CLI: python -m hooks.verify_models
├── libs/
│   └── cirislens/sdk/        # CIRISLens log shipping (git submodule)
├── scripts/
│   ├── entrypoint.sh         # Container entrypoint
│   └── preprocess_config.py  # Config env var processing
├── server.py                 # Custom FastAPI endpoints
├── litellm_config.yaml       # Model routing configuration
├── docker-compose.yml        # Container orchestration
└── tests/                    # Test suite (306 tests)
```

## Ecosystem

CIRISProxy is part of the [CIRISBridge](https://github.com/CIRISAI/CIRISBridge) infrastructure:

| Service | Purpose | Veilid Replacement |
|---------|---------|-------------------|
| CIRISDNS | Service discovery | DHT peer discovery |
| **CIRISProxy** | LLM routing | Private routes |
| CIRISBilling | Credit management | TBD |
| CIRISLens | Observability | Decentralized logging |

### Sunset Plan

This infrastructure is temporary. Per the [CIRIS Covenant](https://ciris.ai/covenant):

> *"We vow not to freeze the music into marble, nor surrender the melody to chaos, but to keep the song singable for every voice yet unheard."*

**Target retirement:** 18-24 months after Veilid production readiness.

See [CIRISBridge FSD](https://github.com/CIRISAI/CIRISBridge/blob/main/FSD.md) for the full transition plan.

## Mission

CIRISProxy serves **Meta-Goal M-1** from the CIRIS Covenant:

> *Promote sustainable adaptive coherence — the living conditions under which diverse sentient beings may pursue their own flourishing in justice and wonder.*

Agents cannot serve this mission if users cannot reach them. CIRISProxy bridges that gap until decentralized alternatives mature.

## License

[Apache License 2.0](LICENSE)

## Links

- [CIRIS Agent](https://github.com/CIRISAI/CIRISAgent) - The ethical AI agent framework
- [CIRISBilling](https://github.com/CIRISAI/CIRISBilling) - Credit and payment management
- [CIRISBridge](https://github.com/CIRISAI/CIRISBridge) - Infrastructure orchestration
- [CIRIS Covenant](https://ciris.ai/covenant) - Ethical framework
- [Safety Policy](https://ciris.ai/safety-policy) - Operational safety guidelines
