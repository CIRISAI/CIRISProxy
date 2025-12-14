#!/bin/bash
# CIRISProxy Entrypoint
# Pre-loads billing callback before starting LiteLLM

set -e

echo "[CIRISProxy] Pre-loading billing callback..."

# Force-import the callback module to ensure it's loaded
# This writes /app/callback_status.json as a side effect
python -c "
import sys
sys.path.insert(0, '/app')
try:
    import billing_callback
    print('[CIRISProxy] Billing callback loaded successfully')
    print(f'[CIRISProxy] Callback instance: {billing_callback.billing_callback_instance}')
except Exception as e:
    print(f'[CIRISProxy] ERROR loading billing callback: {e}')
    sys.exit(1)
"

echo "[CIRISProxy] Starting LiteLLM proxy..."

# Execute the original LiteLLM entrypoint with all arguments
exec litellm "$@"
