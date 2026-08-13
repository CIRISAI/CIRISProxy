# CIRISProxy - LiteLLM with Google ID Token Auth + CIRISBilling
# Self-contained image with all hooks and SDK baked in

# SECURITY: Pin to verified safe version after March 2026 supply chain attack
# v1.82.7 and v1.82.8 were compromised. v1.83.0+ are post-incident verified clean.
# See: https://docs.litellm.ai/blog/security-update-march-2026
# CVE-2026-42208 (SQLi) affects 1.81.16 - 1.83.6 — must be >= 1.83.7
# CVE-2026-49468 (auth bypass via Host Header Injection) affects < 1.84.0 — must be >= 1.84.0
# Image is signed with cosign - verify with:
#   cosign verify --key https://raw.githubusercontent.com/BerriAI/litellm/main/cosign.pub ghcr.io/berriai/litellm:v1.90.0
# Note: -stable channel topped out at v1.83.14; v1.84+ are published without the suffix
FROM ghcr.io/berriai/litellm:v1.90.0

# Install dependencies
RUN python -m ensurepip && python -m pip install --no-cache-dir google-auth>=2.0.0 httpx>=0.24.0

# Create directories
RUN mkdir -p /app/libs /app/hooks /app/logs

# Copy SDK (from git submodule)
COPY libs/ /app/libs/

# Copy hooks (billing callback, custom auth, search handler, and status handler)
COPY hooks/billing_callback.py /app/billing_callback.py
COPY hooks/custom_auth.py /app/custom_auth.py
COPY hooks/search_handler.py /app/search_handler.py
COPY hooks/status_handler.py /app/status_handler.py
COPY hooks/model_audit.py /app/model_audit.py

# Copy custom server wrapper
COPY server.py /app/server.py

# Copy config
COPY litellm_config.yaml /app/config.yaml

# Copy scripts
# hooks/verify_models.py is deliberately NOT shipped: it is a CI and developer
# tool, and the model audit it performs runs inside the image anyway, off the
# health probe's /models response (see hooks/model_audit.py).
COPY scripts/healthcheck.py /app/healthcheck.py
COPY scripts/entrypoint.sh /app/entrypoint.sh
COPY scripts/preprocess_config.py /app/preprocess_config.py
RUN chmod +x /app/healthcheck.py /app/entrypoint.sh /app/preprocess_config.py

# Set working directory
WORKDIR /app

# Ensure Python can find modules in /app
ENV PYTHONPATH=/app

# Use our entrypoint that pre-loads the callback
ENTRYPOINT ["/app/entrypoint.sh"]

# Default command
CMD ["--config", "/app/config.yaml", "--port", "4000"]
