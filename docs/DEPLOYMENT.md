# CIRISProxy Deployment Guide

## Prerequisites

- Vultr VPS (or any VPS with Docker support)
- Cloudflare account with domain
- API keys for at least one LLM provider (Groq, Together, OpenAI)
- CIRISBilling API key

## Quick Start

### 1. Create Vultr VPS

- **Plan**: Cloud Compute, $6/mo (1 vCPU, 1GB RAM) is sufficient
- **OS**: Ubuntu 24.04 LTS
- **Location**: Choose closest to your users

### 2. Initial Server Setup

SSH into your VPS:

```bash
ssh root@your-vps-ip

# Update system
apt update && apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com | sh

# Create deploy user (optional but recommended)
useradd -m -s /bin/bash deploy
usermod -aG docker deploy
su - deploy
```

### 3. Clone and Configure

```bash
# Clone repository
git clone https://github.com/yourorg/CIRISProxy.git /opt/cirisproxy
cd /opt/cirisproxy

# Create environment file
cp .env.production .env
nano .env  # Fill in your values
```

### 4. Set Up Cloudflare Tunnel

1. Go to **Cloudflare Dashboard** → **Zero Trust** → **Networks** → **Tunnels**
2. Click **Create a tunnel**
3. Name it `cirisproxy`
4. Copy the tunnel token
5. Add to `.env`:
   ```
   CLOUDFLARE_TUNNEL_TOKEN=eyJhIjoi...
   ```

6. Configure public hostname in Cloudflare:
   - **Subdomain**: `llm`
   - **Domain**: `ciris.ai`
   - **Service**: `http://litellm:4000`

### 5. Deploy

```bash
./deploy.sh setup
./deploy.sh start
```

### 6. Verify

```bash
# Check status
./deploy.sh status

# Test endpoint
curl https://llm.ciris.ai/health
```

## Commands Reference

```bash
./deploy.sh setup    # Initial setup
./deploy.sh start    # Start services
./deploy.sh stop     # Stop services
./deploy.sh restart  # Restart services
./deploy.sh logs     # View all logs
./deploy.sh logs litellm      # View LiteLLM logs only
./deploy.sh logs cloudflared  # View tunnel logs only
./deploy.sh status   # Check health
./deploy.sh update   # Pull latest and restart
```

## Cloudflare Tunnel Configuration

In the Cloudflare Zero Trust dashboard, configure the tunnel's public hostname:

| Field | Value |
|-------|-------|
| Subdomain | `llm` |
| Domain | `ciris.ai` |
| Path | (leave empty) |
| Type | HTTP |
| URL | `litellm:4000` |

### Optional Security Settings

Under **Access** → **Applications**, create an application to add authentication:

- Skip this for the API endpoint (clients authenticate via billing)
- Consider adding for `/admin` endpoints if exposed

## Monitoring

### View Logs

```bash
# All services
./deploy.sh logs

# Just LiteLLM
./deploy.sh logs litellm

# Last 100 lines
docker compose -f docker-compose.prod.yml logs --tail=100
```

### Health Check

```bash
curl http://localhost:4000/health
# Returns: {"status":"healthy"}
```

### Resource Usage

```bash
docker stats
```

## Troubleshooting

### Container won't start

```bash
# Check logs
docker compose -f docker-compose.prod.yml logs litellm

# Common issues:
# - Missing environment variables
# - Invalid API keys
# - Port already in use
```

### Tunnel not connecting

```bash
# Check tunnel logs
docker compose -f docker-compose.prod.yml logs cloudflared

# Verify token is correct in .env
# Check Cloudflare dashboard for tunnel status
```

### Billing service unreachable

The proxy fails closed - if CIRISBilling is unreachable, requests are denied.

```bash
# Test billing connectivity from container
docker exec ciris-proxy curl -v https://billing.ciris.ai/health
```

## Updating

```bash
cd /opt/cirisproxy
./deploy.sh update
```

This will:
1. Pull latest git changes
2. Pull latest Docker images
3. Restart services with zero downtime

## Backup

The proxy is stateless - no data to backup. Just ensure your `.env` file is backed up securely.

## Security Checklist

- [ ] `.env` file has restrictive permissions (`chmod 600 .env`)
- [ ] VPS firewall only allows SSH (Cloudflare Tunnel handles HTTP)
- [ ] `LITELLM_MASTER_KEY` is a strong random string
- [ ] LLM API keys have appropriate rate limits at provider
- [ ] SSH uses key-based authentication (disable password auth)
