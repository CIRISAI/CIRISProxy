# CIRIS Proxy - Claude Context

## Current Status (v0.2.0 - 2025-12-16)

**Image:** `ghcr.io/cirisai/cirisproxy:latest`
**Tests:** 200 passing, 86% coverage
**CI/CD:** GitHub Actions → GHCR

### Status Endpoint Semantics

`/v1/status` reports two different things, and conflating them cost four days
of public amber (issue #7):

- `status` — **this service's** health. Non-pooled dependencies (billing) plus
  any pool below `min_available`. One dead LLM provider does not appear here.
- `pools` — capability rollup over redundant providers. `min_available: 1`, so
  the pool is operational while any member is up. `primary_available: false`
  means we are serving on a fallback: worth knowing, not an outage.

Each LLM provider also carries a `models` block auditing the configured models
against that provider's `/models` listing — free, since the health check
already fetched that response. See `hooks/model_audit.py`.

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
| `DEEPINFRA_API_KEY` | Yes | DeepInfra API key (**primary** — serves the default chain) |
| `OPENROUTER_API_KEY` | Yes | OpenRouter API key (first fallback) |
| `GROQ_API_KEY` | Yes | Groq API key (second fallback, `fast` alias) |
| `TOGETHER_API_KEY` | No | Together AI API key (not in the default chain) |
| `EXA_API_KEY` | Yes* | Exa AI search (ZDR-compliant) |
| `BRAVE_API_KEY` | No | Brave Search (fallback) |
| `SEARCH_PROVIDER` | No | `auto` (default), `exa`, or `brave` |
| `BILLING_API_URL` | Yes | CIRISBilling endpoint |
| `BILLING_API_KEY` | Yes | Service-to-service auth |
| `GOOGLE_CLIENT_ID` | Yes | Google OAuth client ID |
| `CIRISLENS_TOKEN` | No | Log shipping token |
| `LITELLM_MASTER_KEY` | Yes | Admin operations key |
| `OPENROUTER_IGNORE_PROVIDERS` | No | Comma-separated providers to exclude (e.g., "Friendli,Google") |
| `CIRIS_TEST_AUTH_ENABLED` | No | Enable test auth mode ("true" to enable) |
| `CIRIS_TEST_USER_ID` | No | Test user ID (default: "ciris_synthetic_canary") |

## Model Routing

Primary model: **Qwen 3.6-35B-A3B** (MoE, 262K context).

```yaml
# Default routing
default → deepinfra/Qwen/Qwen3.6-35B-A3B      # PRIMARY — the one that serves
  ↓ fallback
openrouter/qwen/qwen3.6-35b-a3b
  ↓ fallback
groq/qwen/qwen3.6-27b                          # cross-model, high speed
```

Configured in `litellm_config.yaml` with:
- Error-specific retry policies (timeout, rate limit, server error)
- 60s cooldown for failing providers
- Same model family throughout the chain, so tool-calling behaviour survives failover

**Two invariants when editing the chain:**

1. Mirror it in `PROVIDERS` in `hooks/status_handler.py`. Monitoring the
   fallbacks while the primary goes unchecked renders a green board during a
   primary outage — that was issue #7.
2. Run `python -m hooks.verify_models` before merging. Providers retire models
   silently; Groq dropped `llama-4-scout` and `llama-4-maverick` while both
   were still named in the chain.

`litellm_config.yaml` in this repo is the source of truth; CIRISBridge's
Ansible template is synced from it. When the two drift, the monitored set stops
matching the serving set.

## Credit Model

**1 Credit = 1 User Interaction** (not 1 LLM call)

```
User sends message →
  Agent processes (12-70 LLM calls) →
    All calls share same interaction_id →
    Only FIRST call charges via idempotency_key →
  User charged exactly 1 credit
```

## Test Auth Mode

For integration testing without Google OAuth infrastructure:

```bash
# Enable test auth mode
CIRIS_TEST_AUTH_ENABLED=true
CIRIS_TEST_USER_ID=ciris_synthetic_canary

# CIRISBilling must also have test auth enabled with matching token
BILLING_API_URL=http://billing:8000
```

**Flow:**
1. Client sends opaque test token (64-char hex string) instead of Google JWT
2. Proxy detects non-JWT token and validates via CIRISBilling `/v1/billing/credits/check`
3. Billing validates token and returns credit status
4. Proxy returns `test:{user_id}` format for billing callback

**Security:** Never enable in production. Test tokens bypass Google OAuth entirely.

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
| `/v1/status` | GET | Provider health + pool rollup + model audit |
| `/v1/status/simple` | GET | Liveness check |
| `/v1/web/search` | POST | Web search (Exa/Brave) |
| `/health/liveliness` | GET | Container health |

## Development

```bash
# Setup
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Run tests
pytest                                        # All 306 tests
pytest --cov=hooks --cov-report=term-missing  # With coverage

# Lint (enforced in CI — keep it clean)
ruff check .

# Audit routing config against what providers actually serve
python -m hooks.verify_models
```

## CI

| Workflow | Trigger | Jobs |
|----------|---------|------|
| `ci.yml` | PRs; called by `deploy.yml` | ruff, pytest + coverage gate (75%), config audit, image runtime smoke |
| `deploy.yml` | push to main | calls `ci.yml`, then builds/pushes to GHCR |
| `model-availability.yml` | daily 07:00 UTC | live model audit; opens/closes an issue labelled `model-availability` |
| `sonarcloud.yml` | push/PR | static analysis |

`ci.yml` is a reusable workflow — `deploy.yml` calls it rather than keeping a
second copy, so the publish gate and the PR gate cannot drift. The image
runtime smoke boots the built container and requires it to serve
`/v1/status/simple`; it also asserts every hook module imports inside the
image, which is what catches a new module added to `hooks/` without its
`COPY` line in the Dockerfile.

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
