#!/bin/bash
# Initialize Let's Encrypt certificates for llm.ciris.ai
# Run this ONCE on first deployment

set -e

DOMAIN="llm.ciris.ai"
EMAIL="${LETSENCRYPT_EMAIL:-admin@ciris.ai}"
STAGING="${LETSENCRYPT_STAGING:-0}"  # Set to 1 for testing

cd /opt/cirisproxy

echo "=== CIRISProxy TLS Setup ==="
echo "Domain: $DOMAIN"
echo "Email: $EMAIL"
echo "Staging: $STAGING"
echo ""

# Create directories
mkdir -p certbot/www certbot/conf

# Use init config (HTTP only) first
echo "Setting up initial HTTP-only nginx config..."
cp nginx/default-init.conf nginx/default.conf

# Start litellm first
echo "Starting LiteLLM..."
docker compose -f docker-compose.prod.yml up -d litellm
sleep 10

# Start nginx with HTTP-only config
echo "Starting nginx (HTTP only)..."
docker compose -f docker-compose.prod.yml up -d nginx
sleep 5

# Check if nginx is running
if ! docker compose -f docker-compose.prod.yml ps nginx | grep -q "Up"; then
    echo "ERROR: nginx failed to start"
    docker compose -f docker-compose.prod.yml logs nginx
    exit 1
fi

# Request certificate
echo "Requesting Let's Encrypt certificate..."
STAGING_ARG=""
if [ "$STAGING" = "1" ]; then
    STAGING_ARG="--staging"
    echo "(Using staging environment for testing)"
fi

docker compose -f docker-compose.prod.yml run --rm certbot certonly \
    --webroot \
    --webroot-path=/var/www/certbot \
    --email "$EMAIL" \
    --agree-tos \
    --no-eff-email \
    $STAGING_ARG \
    -d "$DOMAIN"

# Check if cert was obtained
if [ ! -f "certbot/conf/live/$DOMAIN/fullchain.pem" ]; then
    echo "ERROR: Certificate not obtained"
    exit 1
fi

echo "Certificate obtained successfully!"

# Restore full nginx config with TLS
echo "Switching to TLS nginx config..."
cat > nginx/default.conf << 'NGINX_CONF'
# CIRISProxy Nginx Configuration
# TLS termination and reverse proxy to LiteLLM

# Rate limiting zone: 10 requests per second per IP
limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;

upstream litellm {
    server litellm:4000;
    keepalive 32;
}

# HTTP -> HTTPS redirect
server {
    listen 80;
    listen [::]:80;
    server_name llm.ciris.ai;

    # Let's Encrypt challenge
    location /.well-known/acme-challenge/ {
        root /var/www/certbot;
    }

    location / {
        return 301 https://$host$request_uri;
    }
}

# HTTPS server
server {
    listen 443 ssl http2;
    listen [::]:443 ssl http2;
    server_name llm.ciris.ai;

    # TLS certificates (managed by certbot)
    ssl_certificate /etc/letsencrypt/live/llm.ciris.ai/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/llm.ciris.ai/privkey.pem;

    # Modern TLS configuration
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384:ECDHE-ECDSA-CHACHA20-POLY1305:ECDHE-RSA-CHACHA20-POLY1305:DHE-RSA-AES128-GCM-SHA256:DHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    ssl_session_timeout 1d;
    ssl_session_cache shared:SSL:10m;
    ssl_session_tickets off;

    # OCSP stapling
    ssl_stapling on;
    ssl_stapling_verify on;
    resolver 1.1.1.1 8.8.8.8 valid=300s;
    resolver_timeout 5s;

    # Security headers
    add_header X-Frame-Options "DENY" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;

    # Health check endpoint (no rate limiting)
    location /health {
        proxy_pass http://litellm/health;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
    }

    # API endpoints with rate limiting
    location / {
        # Rate limiting: burst of 20, no delay for first 10
        limit_req zone=api_limit burst=20 nodelay;
        limit_req_status 429;

        # Proxy to LiteLLM
        proxy_pass http://litellm;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header Connection "";

        # Timeouts for LLM requests (can be slow)
        proxy_connect_timeout 10s;
        proxy_send_timeout 120s;
        proxy_read_timeout 120s;

        # Buffering settings
        proxy_buffering off;
        proxy_request_buffering off;

        # Max request size (for large prompts)
        client_max_body_size 10m;
    }
}
NGINX_CONF

# Restart nginx with TLS config
echo "Restarting nginx with TLS..."
docker compose -f docker-compose.prod.yml restart nginx
sleep 5

# Start certbot for auto-renewal
echo "Starting certbot renewal service..."
docker compose -f docker-compose.prod.yml up -d certbot

# Verify
echo ""
echo "=== Verifying setup ==="
echo "Testing HTTPS endpoint..."
if curl -sf "https://$DOMAIN/health" > /dev/null 2>&1; then
    echo "SUCCESS: https://$DOMAIN is working!"
else
    echo "WARNING: HTTPS health check failed. Checking logs..."
    docker compose -f docker-compose.prod.yml logs --tail=20 nginx
fi

echo ""
echo "=== Setup Complete ==="
echo "Endpoint: https://$DOMAIN"
echo ""
echo "To check status: docker compose -f docker-compose.prod.yml ps"
echo "To view logs: docker compose -f docker-compose.prod.yml logs -f"
