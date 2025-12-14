# CIRISProxy - LiteLLM with Google ID Token Auth + CIRISBilling
# Self-contained image with all hooks and SDK baked in

FROM ghcr.io/berriai/litellm:main-latest

# Install dependencies
RUN pip install --no-cache-dir google-auth>=2.0.0 httpx>=0.24.0

# Create directories
RUN mkdir -p /app/sdk /app/hooks /app/logs

# Copy SDK
COPY sdk/ /app/sdk/

# Copy hooks (billing callback, custom auth, search handler, and status handler)
COPY hooks/billing_callback.py /app/billing_callback.py
COPY hooks/custom_auth.py /app/custom_auth.py
COPY hooks/search_handler.py /app/search_handler.py
COPY hooks/status_handler.py /app/status_handler.py

# Copy config
COPY litellm_config.yaml /app/config.yaml

# Copy scripts
COPY scripts/healthcheck.py /app/healthcheck.py
COPY scripts/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/healthcheck.py /app/entrypoint.sh

# Set working directory
WORKDIR /app

# Ensure Python can find modules in /app
ENV PYTHONPATH=/app

# Use our entrypoint that pre-loads the callback
ENTRYPOINT ["/app/entrypoint.sh"]

# Default command
CMD ["--config", "/app/config.yaml", "--port", "4000"]
