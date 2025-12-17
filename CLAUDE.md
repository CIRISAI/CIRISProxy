# CIRIS Proxy - Claude Context

## Current Status (v0.2.0 - 2025-12-16)

**Image:** `ghcr.io/cirisai/cirisproxy:latest`
**Tests:** 200 passing, 86% coverage
**CI/CD:** GitHub Actions → GHCR

### Recent Changes
- Migrated LogShipper to CIRISLens git submodule (libs/cirislens/sdk)
- Added Exa AI as primary ZDR-compliant search provider (Brave fallback)
- Enhanced error logging with provider identification for debugging
- Refactored high cognitive complexity functions (SonarCloud compliant)
- Comprehensive test suite with Hypothesis property-based testing

### Known Issues
- **ActionSelectionPDMA malformed JSON**: One LLM provider occasionally returns garbage like `{'type': 'type: ', 'type: ': 'type: '}` for complex schemas. Error logs now include `provider` field to identify culprit. Query CIRISLens:
  ```sql
  SELECT provider, COUNT(*) as errors FROM cirislens.service_logs
  WHERE event = 'llm_error' GROUP BY provider ORDER BY errors DESC;
  ```

## Project Overview

**CIRISProxy** is a LiteLLM-based proxy service that provides secure, credit-gated LLM access for CIRIS Agent mobile clients.

- **Domain**: llm.ciris.ai (via CIRISBridge proxy1.ciris-services-1.ai)
- **Tech Stack**: LiteLLM, FastAPI, Docker
- **Purpose**: Secure API key isolation + credit-based access control + ZDR web search

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  CIRIS Agent    │────▶│   CIRISProxy    │────▶│  LLM Providers  │
│  (Mobile App)   │     │  (This Service) │     │  Groq/Together/ │
└─────────────────┘     └─────────────────┘     │  OpenRouter     │
        │                       │               └─────────────────┘
        │                       │
        │                       ▼
        │               ┌─────────────────┐
        └──────────────▶│  CIRISBilling   │
                        │  (Credit Mgmt)  │
                        └─────────────────┘
```

## Project Structure

```
CIRISProxy/
├── CLAUDE.md               # This file
├── README.md               # Public documentation
├── RELEASE_NOTES.md        # Version changelog
├── pyproject.toml          # Python package config
├── docker-compose.yml      # Local dev container
├── litellm_config.yaml     # Model routing config
├── server.py               # Custom FastAPI endpoints
├── hooks/
│   ├── billing_callback.py # LiteLLM callback for billing + logging
│   ├── custom_auth.py      # Google OAuth token verification
│   ├── search_handler.py   # Web search (Exa primary, Brave fallback)
│   └── status_handler.py   # Provider health monitoring
├── libs/                   # Git submodules for sister repos
│   └── cirislens/          # → github.com/CIRISAI/CIRISLens
│       └── sdk/            # LogShipper, resilience patterns
├── tests/                  # 200 tests, 86% coverage
│   ├── test_billing_callback.py
│   ├── test_custom_auth.py
│   ├── test_search_handler.py
│   ├── test_server.py
│   ├── test_status_handler.py
│   └── test_logshipper.py
└── .github/workflows/      # CI/CD pipelines
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | Yes | Groq API key |
| `TOGETHER_API_KEY` | Yes | Together AI API key |
| `OPENROUTER_API_KEY` | Yes | OpenRouter API key (default provider) |
| `EXA_API_KEY` | Yes* | Exa AI search (ZDR-compliant) |
| `BRAVE_API_KEY` | No | Brave Search (fallback) |
| `SEARCH_PROVIDER` | No | `auto` (default), `exa`, or `brave` |
| `BILLING_API_URL` | Yes | CIRISBilling endpoint |
| `BILLING_API_KEY` | Yes | Service-to-service auth |
| `GOOGLE_CLIENT_ID` | Yes | Google OAuth client ID |
| `CIRISLENS_TOKEN` | No | Log shipping token |
| `LITELLM_MASTER_KEY` | Yes | Admin operations key |

## Model Routing

Primary model: **Llama 4 Maverick** across 3 providers for redundancy.

```yaml
# Default routing (cheapest first)
default → openrouter/meta-llama/llama-4-maverick  # $0.11/$0.34 per 1M
  ↓ fallback
groq/llama-4-maverick                              # $0.50/$0.77 per 1M
  ↓ fallback
together/llama-4-maverick                          # $0.50/$0.80 per 1M
```

Configured in `litellm_config.yaml` with:
- Error-specific retry policies (timeout, rate limit, server error)
- 60s cooldown for failing providers
- Never degrades to older models (Llama 4 only)

## Credit Model

**1 Credit = 1 User Interaction** (not 1 LLM call)

```
User sends message →
  Agent processes (12-70 LLM calls) →
    All calls share same interaction_id →
    Only FIRST call charges via idempotency_key →
  User charged exactly 1 credit
```

## CIRISLens Events

| Event | Level | Fields |
|-------|-------|--------|
| `auth_granted` | INFO | interaction_id, user_hash, credits_remaining |
| `auth_denied` | WARNING | interaction_id, user_hash, reason |
| `charge_created` | INFO | interaction_id, user_hash |
| `llm_request` | INFO | interaction_id, model, actual_model, api_base, tokens |
| `llm_error` | ERROR | interaction_id, **provider**, actual_model, api_base, error |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | OpenAI-compatible chat |
| `/v1/status` | GET | Provider health status |
| `/v1/status/simple` | GET | Liveness check |
| `/v1/web/search` | POST | Web search (Exa/Brave) |
| `/health/liveliness` | GET | Container health |

## Development

```bash
# Setup
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Run tests
pytest                                    # All 200 tests
pytest --cov=hooks --cov-report=term-missing  # With coverage

# Lint
ruff check .
```

## Deployment

Via CIRISBridge Ansible:
```bash
cd ~/CIRISBridge/ansible
ansible-playbook -i inventory/production.yml playbooks/site.yml --tags proxy
```

## Related Repositories

| Repository | Purpose |
|------------|---------|
| [CIRISAgent](https://github.com/CIRISAI/CIRISAgent) | Core AI agent + Android app |
| [CIRISBilling](https://github.com/CIRISAI/CIRISBilling) | Credit management + Google Play |
| [CIRISBridge](https://github.com/CIRISAI/CIRISBridge) | Infrastructure orchestration |
| [CIRISLens](https://github.com/CIRISAI/CIRISLens) | Observability + dashboards |

## Mission Alignment

CIRISProxy is **temporary bridging infrastructure** designed to be retired when Veilid matures:
- DNS → Veilid DHT peer discovery
- Proxy → Veilid private routes
- Billing → TBD

Target retirement: 18-24 months after Veilid production readiness.

See [CIRIS Covenant](https://ciris.ai/covenant) and [Safety Policy](https://ciris.ai/safety-policy).
