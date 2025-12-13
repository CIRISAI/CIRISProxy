# CIRISProxy - LiteLLM with Google ID Token Auth + CIRISBilling
# Self-contained image with all hooks and SDK baked in

FROM ghcr.io/berriai/litellm:main-latest

# Install dependencies
RUN pip install --no-cache-dir google-auth>=2.0.0 httpx>=0.24.0

# Create directories
RUN mkdir -p /app/sdk /app/hooks /app/logs

# Copy SDK
COPY sdk/ /app/sdk/

# Copy hooks (billing callback and custom auth)
COPY hooks/billing_callback.py /app/billing_callback.py
COPY hooks/custom_auth.py /app/custom_auth.py

# Copy config
COPY litellm_config.yaml /app/config.yaml

# Set working directory
WORKDIR /app

# Default command
CMD ["--config", "/app/config.yaml", "--port", "4000"]
