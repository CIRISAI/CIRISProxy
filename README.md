# CIRISProxy

LiteLLM-based proxy service that provides secure, credit-gated LLM access for CIRIS Agent mobile clients.

## Overview

CIRISProxy solves the problem of mobile apps needing to make LLM API calls without exposing API keys:

- **API Key Isolation**: Groq, Together, and OpenAI keys are stored server-side only
- **Credit Gating**: Integrates with CIRISBilling to enforce credit-based access
- **Per-Interaction Billing**: 1 credit = 1 user interaction (regardless of LLM call count)

## Quick Start

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your API keys
vim .env

# Start the proxy
docker-compose up -d

# Test it
curl -X POST http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer google:test-user-id" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "groq/llama-3.1-8b",
    "messages": [{"role": "user", "content": "Hello"}],
    "metadata": {"interaction_id": "test-int-123"}
  }'
```

## Architecture

See [CLAUDE.md](CLAUDE.md) for detailed architecture documentation.

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
pytest --cov=hooks --cov-report=term-missing
```

## License

Apache License 2.0 - See [LICENSE](LICENSE) for details.
