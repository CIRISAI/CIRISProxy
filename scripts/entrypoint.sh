#!/bin/bash
# CIRISProxy Entrypoint
# Starts LiteLLM with custom routes for /v1/status

set -e

echo "[CIRISProxy] Preprocessing config..."

# Preprocess config to apply env var overrides (e.g., OPENROUTER_IGNORE_PROVIDERS)
python /app/preprocess_config.py /app/config.yaml /app/config.processed.yaml

echo "[CIRISProxy] Pre-loading modules..."

# Force-import callback and status modules to ensure they're loaded
python -c "
import sys
sys.path.insert(0, '/app')

# Load billing callback
try:
    import billing_callback
    print('[CIRISProxy] Billing callback loaded successfully')
except Exception as e:
    print(f'[CIRISProxy] WARNING: billing callback failed: {e}')

# Load status handler
try:
    import status_handler
    print('[CIRISProxy] Status handler loaded successfully')
except Exception as e:
    print(f'[CIRISProxy] WARNING: status handler failed: {e}')

# Verify custom server can import LiteLLM
try:
    import server
    print('[CIRISProxy] Custom server module loaded successfully')
except Exception as e:
    print(f'[CIRISProxy] ERROR loading custom server: {e}')
    sys.exit(1)
"

echo "[CIRISProxy] Starting server with custom routes..."

# Parse arguments for config and port (use processed config by default)
CONFIG_FILE="/app/config.processed.yaml"
PORT="4000"

while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

# Set config path for LiteLLM
export LITELLM_CONFIG_PATH="$CONFIG_FILE"

# Start uvicorn with our custom server that wraps LiteLLM
exec python -c "
import sys
sys.path.insert(0, '/app')

import uvicorn
from server import main

app = main()
uvicorn.run(app, host='0.0.0.0', port=$PORT, log_level='info')
"
