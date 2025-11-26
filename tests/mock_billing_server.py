#!/usr/bin/env python3
"""
Mock CIRISBilling server for local testing.

Run with: python tests/mock_billing_server.py
Listens on: http://localhost:8080
"""

import json
import logging
from http.server import BaseHTTPRequestHandler, HTTPServer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

# Simulated user credits
USER_CREDITS: dict[str, int] = {
    "test-user-123": 10,
    "test-user-456": 0,  # No credits
}

# Track charges for idempotency
CHARGES: dict[str, dict] = {}


class MockBillingHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8")

        try:
            data = json.loads(body) if body else {}
        except json.JSONDecodeError:
            data = {}

        logger.info(f"POST {self.path} - {data}")

        if self.path == "/v1/billing/litellm/auth":
            self._handle_auth(data)
        elif self.path == "/v1/billing/litellm/charge":
            self._handle_charge(data)
        elif self.path == "/v1/billing/litellm/usage":
            self._handle_usage(data)
        else:
            self._send_response(404, {"error": "Not found"})

    def _handle_auth(self, data: dict):
        external_id = data.get("external_id", "")
        interaction_id = data.get("interaction_id", "")
        model = data.get("model", "unknown")

        credits = USER_CREDITS.get(external_id, 0)
        authorized = credits > 0

        logger.info(
            f"AUTH: user={external_id}, interaction={interaction_id}, "
            f"model={model}, credits={credits}, authorized={authorized}"
        )

        self._send_response(
            200,
            {
                "authorized": authorized,
                "credits_remaining": credits,
                "interaction_id": interaction_id,
                "reason": None if authorized else "No credits remaining",
            },
        )

    def _handle_charge(self, data: dict):
        external_id = data.get("external_id", "")
        interaction_id = data.get("interaction_id", "")
        idempotency_key = data.get("idempotency_key") or f"litellm:{interaction_id}"

        # Check for existing charge (idempotency)
        if idempotency_key in CHARGES:
            existing = CHARGES[idempotency_key]
            logger.info(
                f"CHARGE (idempotent): user={external_id}, interaction={interaction_id}, "
                f"existing_charge={existing['charge_id']}"
            )
            self._send_response(
                200,
                {
                    "charged": True,
                    "credits_deducted": 1,
                    "credits_remaining": existing.get("credits_after", 0),
                    "charge_id": existing["charge_id"],
                },
            )
            return

        # New charge
        credits = USER_CREDITS.get(external_id, 0)
        if credits > 0:
            USER_CREDITS[external_id] = credits - 1
            charge_id = f"charge-{len(CHARGES) + 1}"
            CHARGES[idempotency_key] = {
                "charge_id": charge_id,
                "external_id": external_id,
                "interaction_id": interaction_id,
                "credits_after": credits - 1,
            }

            logger.info(
                f"CHARGE: user={external_id}, interaction={interaction_id}, "
                f"charge_id={charge_id}, credits_remaining={credits - 1}"
            )

            self._send_response(
                200,
                {
                    "charged": True,
                    "credits_deducted": 1,
                    "credits_remaining": credits - 1,
                    "charge_id": charge_id,
                },
            )
        else:
            logger.warning(f"CHARGE FAILED: user={external_id}, no credits")
            self._send_response(
                402,
                {
                    "charged": False,
                    "error": "Insufficient credits",
                },
            )

    def _handle_usage(self, data: dict):
        external_id = data.get("external_id", "")
        interaction_id = data.get("interaction_id", "")
        llm_calls = data.get("total_llm_calls", 0)
        prompt_tokens = data.get("total_prompt_tokens", 0)
        completion_tokens = data.get("total_completion_tokens", 0)
        cost_cents = data.get("actual_cost_cents", 0)

        logger.info(
            f"USAGE: user={external_id}, interaction={interaction_id}, "
            f"calls={llm_calls}, tokens={prompt_tokens}/{completion_tokens}, "
            f"cost={cost_cents} cents"
        )

        self._send_response(
            200,
            {
                "logged": True,
                "usage_log_id": f"usage-{interaction_id}",
            },
        )

    def _send_response(self, status: int, data: dict):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))

    def log_message(self, format, *args):
        # Suppress default logging
        pass


def main():
    port = 8080
    server = HTTPServer(("0.0.0.0", port), MockBillingHandler)
    logger.info(f"Mock CIRISBilling server running on http://localhost:{port}")
    logger.info("Test users:")
    for user, credits in USER_CREDITS.items():
        logger.info(f"  - {user}: {credits} credits")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
