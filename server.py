"""
CIRISProxy Custom Server

Wraps LiteLLM proxy with custom endpoints:
- /v1/status - Provider health monitoring
- /v1/search - Web search with billing (future)

This approach properly initializes LiteLLM with config, then adds custom routes.
"""

import asyncio
import os
import sys

# Ensure /app is in path for imports
sys.path.insert(0, "/app")

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# Import status handler
from status_handler import get_status


def add_custom_routes(app: FastAPI) -> None:
    """Add custom routes to the LiteLLM FastAPI app."""

    @app.get("/v1/status")
    async def status_endpoint():
        """
        Provider health status endpoint.

        Returns health status for all CIRISProxy providers:
        - LLM: OpenRouter, Groq, Together AI
        - Billing: CIRISBilling
        - Search: Brave Search
        """
        try:
            status = await get_status()
            return JSONResponse(content=status)
        except Exception:
            # Never expose raw exception messages - may contain sensitive data
            return JSONResponse(
                content={
                    "service": "cirisproxy",
                    "status": "outage",
                    "error": "Internal error",
                },
                status_code=500,
            )

    @app.get("/v1/status/simple")
    async def status_simple():
        """
        Simple status check - just returns OK if service is running.
        Useful for basic health checks that don't need provider details.
        """
        return JSONResponse(content={"status": "ok", "service": "cirisproxy"})


def main():
    """
    Main entry point - initializes LiteLLM with config and adds custom routes.

    IMPORTANT: Must call initialize() to load config (including custom_auth).
    Simply importing the app doesn't load the config file.
    """
    config_path = os.environ.get("LITELLM_CONFIG_PATH", "/app/config.yaml")

    # Import LiteLLM components
    from litellm.proxy.proxy_server import app, initialize

    # Initialize LiteLLM with config - this loads custom_auth, callbacks, etc.
    print(f"[CIRISProxy] Loading config from {config_path}")
    asyncio.get_event_loop().run_until_complete(
        initialize(config=config_path)
    )
    print("[CIRISProxy] LiteLLM initialized with config")

    # Add our custom routes
    add_custom_routes(app)

    print("[CIRISProxy] Custom routes added: /v1/status, /v1/status/simple")

    return app


if __name__ == "__main__":
    # For testing - run with: python server.py
    import uvicorn

    app = main()
    uvicorn.run(app, host="0.0.0.0", port=4000)
