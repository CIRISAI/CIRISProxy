# CIRISProxy - LiteLLM with Google ID Token Auth
# Extends the official LiteLLM image with google-auth for JWT verification

FROM ghcr.io/berriai/litellm:main-latest

# Install google-auth for ID token verification
RUN pip install --no-cache-dir google-auth>=2.0.0

# The rest (config, hooks) are mounted via volumes in docker-compose
