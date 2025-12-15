"""
Web Search Handler for CIRISProxy

Provides /v1/search endpoint that:
1. Authenticates user via Google ID token
2. Charges credits via CIRISBilling Tools API
3. Proxies request to search provider (Exa primary, Brave fallback)
4. Returns search results

Search Providers:
- Exa AI (primary): ZDR-compliant, AI-native semantic search
- Brave Search (fallback): Privacy-focused traditional search

Usage:
  POST /v1/search
  Authorization: Bearer <google_id_token>
  Content-Type: application/json

  {"q": "search query", "count": 10}
"""

import hashlib
import os
import time
from typing import Any

import httpx

# Configuration - Exa (primary, ZDR-compliant)
EXA_API_KEY = os.environ.get("EXA_API_KEY", "")
EXA_API_URL = "https://api.exa.ai/search"

# Configuration - Brave (fallback)
BRAVE_API_KEY = os.environ.get("BRAVE_API_KEY", "")
BRAVE_API_URL = "https://api.search.brave.com/res/v1/web/search"

# Billing
BILLING_API_URL = os.environ.get("BILLING_API_URL", "https://billing.ciris.ai")
BILLING_API_KEY = os.environ.get("BILLING_API_KEY", "")

# Search provider preference
SEARCH_PROVIDER = os.environ.get("SEARCH_PROVIDER", "auto")  # "exa", "brave", or "auto"

# Import Google auth verification from custom_auth
try:
    from custom_auth import verify_google_token
except ImportError:
    # Fallback if custom_auth not available
    async def verify_google_token(_token: str) -> dict | None:
        return None


def _get_active_provider() -> str:
    """Determine which search provider to use."""
    if SEARCH_PROVIDER == "exa" and EXA_API_KEY:
        return "exa"
    elif SEARCH_PROVIDER == "brave" and BRAVE_API_KEY:
        return "brave"
    elif SEARCH_PROVIDER == "auto":
        # Prefer Exa (ZDR-compliant), fallback to Brave
        if EXA_API_KEY:
            return "exa"
        elif BRAVE_API_KEY:
            return "brave"
    return "none"


class SearchHandler:
    """
    Handles web search requests with billing integration.

    Supports multiple search providers:
    - Exa AI: ZDR-compliant semantic search (primary)
    - Brave Search: Privacy-focused traditional search (fallback)
    """

    def __init__(self):
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(30.0, connect=10.0),
            )
        return self._client

    def _generate_idempotency_key(self, user_id: str, query: str) -> str:
        """Generate idempotency key for billing."""
        # Hash query to keep key short but unique
        query_hash = hashlib.sha256(query.encode()).hexdigest()[:8]
        timestamp = int(time.time())
        return f"search-{user_id[:8]}-{query_hash}-{timestamp}"

    async def _charge_credits(
        self,
        oauth_provider: str,
        external_id: str,
        idempotency_key: str,
    ) -> tuple[bool, str | None]:
        """
        Charge web_search credits via billing API.

        Returns:
            (success, error_message)
        """
        try:
            client = await self._get_client()
            resp = await client.post(
                f"{BILLING_API_URL}/v1/tools/charge",
                json={
                    "oauth_provider": oauth_provider,
                    "external_id": external_id,
                    "product_type": "web_search",
                    "idempotency_key": idempotency_key,
                },
                headers={"X-API-Key": BILLING_API_KEY},
            )

            if resp.status_code == 200:
                return True, None
            elif resp.status_code == 402:
                data = resp.json()
                return False, data.get("detail", "No web_search credits available")
            else:
                return False, f"Billing error: {resp.status_code}"

        except httpx.RequestError as e:
            return False, f"Billing service unavailable: {type(e).__name__}"

    async def _call_exa_search(
        self,
        query: str,
        count: int = 10,
        **kwargs,
    ) -> tuple[dict | None, str | None]:
        """
        Call Exa AI Search API (ZDR-compliant).

        Returns:
            (results, error_message)
        """
        if not EXA_API_KEY:
            return None, "Exa API key not configured"

        try:
            client = await self._get_client()

            # Build Exa request body
            request_body = {
                "query": query,
                "numResults": min(count, 100),  # Exa max is 100
                "type": kwargs.get("search_type", "auto"),
            }

            # Add optional parameters if provided
            if "category" in kwargs:
                request_body["category"] = kwargs["category"]
            if "include_domains" in kwargs:
                request_body["includeDomains"] = kwargs["include_domains"]
            if "exclude_domains" in kwargs:
                request_body["excludeDomains"] = kwargs["exclude_domains"]

            # Request text content for RAG use
            request_body["contents"] = {
                "text": {"maxCharacters": 2000},
                "highlights": {"numSentences": 3},
            }

            resp = await client.post(
                EXA_API_URL,
                json=request_body,
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": EXA_API_KEY,
                },
            )

            if resp.status_code == 200:
                exa_response = resp.json()
                # Normalize response to common format
                return self._normalize_exa_response(exa_response), None
            else:
                return None, f"Exa Search error: {resp.status_code}"

        except httpx.RequestError as e:
            return None, f"Exa service unavailable: {type(e).__name__}"

    def _normalize_exa_response(self, exa_response: dict) -> dict:
        """
        Normalize Exa response to a common format compatible with existing clients.

        Converts Exa's response structure to match the expected output format.
        """
        results = []
        for item in exa_response.get("results", []):
            result = {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "description": item.get("text", "")[:500] if item.get("text") else "",
                "published_date": item.get("publishedDate"),
                "author": item.get("author"),
                # Exa-specific fields
                "highlights": item.get("highlights", []),
                "summary": item.get("summary"),
            }
            results.append(result)

        return {
            "web": {
                "results": results,
            },
            "provider": "exa",
            "request_id": exa_response.get("requestId"),
            "search_type": exa_response.get("resolvedSearchType"),
        }

    async def _call_brave_search(
        self,
        query: str,
        count: int = 10,
        **kwargs,
    ) -> tuple[dict | None, str | None]:
        """
        Call Brave Search API (fallback provider).

        Returns:
            (results, error_message)
        """
        if not BRAVE_API_KEY:
            return None, "Brave API key not configured"

        try:
            client = await self._get_client()
            resp = await client.get(
                BRAVE_API_URL,
                params={
                    "q": query,
                    "count": min(count, 20),  # Brave max is 20
                },
                headers={
                    "Accept": "application/json",
                    "Accept-Encoding": "gzip",
                    "X-Subscription-Token": BRAVE_API_KEY,
                },
            )

            if resp.status_code == 200:
                brave_response = resp.json()
                # Add provider tag
                brave_response["provider"] = "brave"
                return brave_response, None
            else:
                return None, f"Brave Search error: {resp.status_code}"

        except httpx.RequestError as e:
            return None, f"Brave service unavailable: {type(e).__name__}"

    async def _execute_search(
        self,
        query: str,
        count: int = 10,
        **kwargs,
    ) -> tuple[dict | None, str | None]:
        """
        Execute search with automatic provider selection and fallback.

        Returns:
            (results, error_message)
        """
        provider = _get_active_provider()

        if provider == "exa":
            results, error = await self._call_exa_search(query, count, **kwargs)
            if results:
                return results, None
            # Fallback to Brave if Exa fails and Brave is configured
            if BRAVE_API_KEY:
                return await self._call_brave_search(query, count, **kwargs)
            return None, error

        elif provider == "brave":
            return await self._call_brave_search(query, count, **kwargs)

        else:
            return None, "No search provider configured (set EXA_API_KEY or BRAVE_API_KEY)"

    async def handle_search(
        self,
        token: str,
        query: str,
        count: int = 10,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Handle a search request.

        Args:
            token: Google ID token for authentication
            query: Search query
            count: Number of results (max 100 for Exa, 20 for Brave)
            **kwargs: Additional search parameters

        Returns:
            Search results or error response
        """
        # 1. Verify Google token
        user_info = await verify_google_token(token)
        if not user_info:
            return {
                "error": "Invalid or expired authentication token",
                "status_code": 401,
            }

        external_id = user_info.get("sub", "")
        oauth_provider = "oauth:google"

        # 2. Generate idempotency key and charge credits
        idempotency_key = self._generate_idempotency_key(external_id, query)
        success, error = await self._charge_credits(
            oauth_provider=oauth_provider,
            external_id=external_id,
            idempotency_key=idempotency_key,
        )

        if not success:
            return {
                "error": error,
                "status_code": 402,  # Payment Required
            }

        # 3. Execute search (Exa primary, Brave fallback)
        results, error = await self._execute_search(query, count, **kwargs)

        if not results:
            # Note: Credit was already charged - this is a search failure
            # Could implement refund logic here if needed
            return {
                "error": error,
                "status_code": 502,  # Bad Gateway
            }

        # 4. Return results
        return {
            "results": results,
            "status_code": 200,
        }

    async def close(self):
        """Close HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None


# Global instance
search_handler = SearchHandler()


# FastAPI route handler (for use with LiteLLM custom routes if supported)
async def handle_search_request(request_data: dict, auth_token: str) -> dict:
    """
    Entry point for search requests.

    Args:
        request_data: {"q": "query", "count": 10, ...}
        auth_token: Bearer token from Authorization header

    Returns:
        Search results or error dict
    """
    query = request_data.get("q", "")
    if not query:
        return {"error": "Missing 'q' parameter", "status_code": 400}

    count = request_data.get("count", 10)

    # Extract search-specific params
    kwargs = {}
    if "category" in request_data:
        kwargs["category"] = request_data["category"]
    if "search_type" in request_data:
        kwargs["search_type"] = request_data["search_type"]
    if "include_domains" in request_data:
        kwargs["include_domains"] = request_data["include_domains"]
    if "exclude_domains" in request_data:
        kwargs["exclude_domains"] = request_data["exclude_domains"]

    return await search_handler.handle_search(
        token=auth_token,
        query=query,
        count=count,
        **kwargs,
    )
