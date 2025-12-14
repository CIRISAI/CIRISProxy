"""
CIRISProxy Custom Server

Wraps LiteLLM proxy with custom endpoints:
- /v1/status - Provider health monitoring
- /v1/search - Web search with billing (future)

This approach adds routes to LiteLLM's FastAPI app before startup.
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
        - LLM: OpenRouter, Groq, Together AI, OpenAI
        - Billing: CIRISBilling
        - Search: Brave Search
        """
        try:
            status = await get_status()
            return JSONResponse(content=status)
        except Exception as e:
            return JSONResponse(
                content={
                    "service": "cirisproxy",
                    "status": "outage",
                    "error": str(e),
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
    Main entry point - imports LiteLLM and adds custom routes.

    This is called by the entrypoint script before starting uvicorn.
    """
    # Import LiteLLM's app
    from litellm.proxy.proxy_server import app

    # Add our custom routes
    add_custom_routes(app)

    print("[CIRISProxy] Custom routes added: /v1/status, /v1/status/simple")

    return app


if __name__ == "__main__":
    # For testing - run with: python server.py
    import uvicorn

    app = main()
    uvicorn.run(app, host="0.0.0.0", port=4000)
