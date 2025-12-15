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

# Import handlers
from status_handler import get_status
from search_handler import handle_search_request


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

    @app.post("/v1/search")
    async def search_endpoint(request: Request):
        """
        Web search endpoint with billing integration.

        Requires:
        - Authorization: Bearer <google_id_token>
        - Body: {"q": "search query", "count": 10}

        Charges 1 web_search credit per request via CIRISBilling.
        """
        # Get auth token
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return JSONResponse(
                content={"error": "Missing or invalid Authorization header"},
                status_code=401,
            )
        token = auth_header[7:]  # Strip "Bearer "

        # Get request body
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(
                content={"error": "Invalid JSON body"},
                status_code=400,
            )

        # Handle search
        result = await handle_search_request(body, token)
        status_code = result.pop("status_code", 200)

        return JSONResponse(content=result, status_code=status_code)


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
