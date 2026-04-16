# CIRISProxy - LiteLLM with Google ID Token Auth + CIRISBilling
# Self-contained image with all hooks and SDK baked in

# SECURITY: Pin to verified safe version after March 2026 supply chain attack
# v1.82.7 and v1.82.8 were compromised. v1.83.0+ are post-incident verified clean.
# See: https://docs.litellm.ai/blog/security-update-march-2026
# Image is signed with cosign - verify with:
#   cosign verify --key https://raw.githubusercontent.com/BerriAI/litellm/0112e53046018d726492c814b3644b7d376029d0/cosign.pub ghcr.io/berriai/litellm:v1.83.3-stable
FROM ghcr.io/berriai/litellm:v1.83.3-stable

# Install dependencies
RUN pip install --no-cache-dir google-auth>=2.0.0 httpx>=0.24.0

# Create directories
RUN mkdir -p /app/libs /app/hooks /app/logs

# Copy SDK (from git submodule)
COPY libs/ /app/libs/

# Copy hooks (billing callback, custom auth, search handler, and status handler)
COPY hooks/billing_callback.py /app/billing_callback.py
COPY hooks/custom_auth.py /app/custom_auth.py
COPY hooks/search_handler.py /app/search_handler.py
COPY hooks/status_handler.py /app/status_handler.py

# Copy custom server wrapper
COPY server.py /app/server.py

# Copy config
COPY litellm_config.yaml /app/config.yaml

# Copy scripts
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
