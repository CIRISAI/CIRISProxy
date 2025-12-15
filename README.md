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
| **Vision Routing** | Automatic rerouting of multimodal requests to compatible providers |
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
| `GROQ_API_KEY` | Yes | Groq API key |
| `TOGETHER_API_KEY` | Yes | Together AI API key |
| `OPENROUTER_API_KEY` | No | OpenRouter API key (fallback) |
| `BILLING_API_URL` | Yes | CIRISBilling endpoint |
| `BILLING_API_KEY` | Yes | Service-to-service auth key |
| `GOOGLE_CLIENT_ID` | Yes | Google OAuth client ID for token verification |
| `CIRISLENS_TOKEN` | No | Token for log shipping to CIRISLens |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | OpenAI-compatible chat endpoint |
| `/v1/status` | GET | Provider health status |
| `/v1/status/simple` | GET | Simple liveness check |
| `/v1/web/search` | POST | Web search (requires credits) |
| `/health/liveliness` | GET | Container health check |

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
```

### Project Structure

```
CIRISProxy/
├── hooks/
│   ├── billing_callback.py   # LiteLLM callback for billing integration
│   ├── custom_auth.py        # Google OAuth token verification
│   ├── search_handler.py     # Web search with Brave API
│   └── status_handler.py     # Provider health monitoring
├── sdk/
│   └── logshipper.py         # CIRISLens log shipping SDK
├── server.py                 # Custom FastAPI endpoints
├── litellm_config.yaml       # Model routing configuration
├── docker-compose.yml        # Container orchestration
└── tests/                    # Test suite (185 tests, 86% coverage)
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
