#!/usr/bin/env python3
"""
CIRISProxy Health Check

Verifies:
1. LiteLLM proxy is responding
2. Billing callback is loaded (via status file)

Exit codes:
0 = Healthy
1 = Unhealthy (callback not loaded or LiteLLM not responding)
"""

import json
import sys
import urllib.request
from pathlib import Path

LITELLM_URL = "http://localhost:4000/health/liveliness"
CALLBACK_STATUS_FILE = "/app/callback_status.json"


def check_litellm() -> bool:
    """Check if LiteLLM proxy is responding."""
    try:
        with urllib.request.urlopen(LITELLM_URL, timeout=5) as resp:
            return resp.status == 200
    except Exception as e:
        print(f"LiteLLM check failed: {e}", file=sys.stderr)
        return False


def check_callback_loaded() -> bool:
    """Check if billing callback was loaded (wrote status file)."""
    status_path = Path(CALLBACK_STATUS_FILE)
    if not status_path.exists():
        print(f"Callback status file not found: {CALLBACK_STATUS_FILE}", file=sys.stderr)
        return False

    try:
        with open(status_path) as f:
            status = json.load(f)

        if not status.get("billing_callback_loaded"):
            print("Billing callback not loaded", file=sys.stderr)
            return False

        return True
    except Exception as e:
        print(f"Failed to read callback status: {e}", file=sys.stderr)
        return False


def main():
    litellm_ok = check_litellm()
    callback_ok = check_callback_loaded()

    if litellm_ok and callback_ok:
        print("OK: LiteLLM responding, billing callback loaded")
        sys.exit(0)
    else:
        if not litellm_ok:
            print("FAIL: LiteLLM not responding", file=sys.stderr)
        if not callback_ok:
            print("FAIL: Billing callback not loaded - POSSIBLE SECURITY ISSUE", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
