#!/bin/bash
set -euo pipefail

# CIRISProxy Deployment Script
# Usage: ./deploy.sh [setup|start|stop|restart|logs|status|update]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="$SCRIPT_DIR/docker-compose.prod.yml"
ENV_FILE="$SCRIPT_DIR/.env"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

check_env() {
    if [[ ! -f "$ENV_FILE" ]]; then
        log_error ".env file not found!"
        log_info "Copy .env.production to .env and fill in your values:"
        log_info "  cp .env.production .env"
        log_info "  nano .env"
        exit 1
    fi

    # Check required vars
    local missing=()
    source "$ENV_FILE"

    [[ -z "${GROQ_API_KEY:-}" && -z "${TOGETHER_API_KEY:-}" && -z "${OPENAI_API_KEY:-}" ]] && missing+=("At least one LLM API key")
    [[ -z "${BILLING_API_KEY:-}" ]] && missing+=("BILLING_API_KEY")
    [[ -z "${LITELLM_MASTER_KEY:-}" ]] && missing+=("LITELLM_MASTER_KEY")

    if [[ ${#missing[@]} -gt 0 ]]; then
        log_error "Missing required environment variables:"
        for var in "${missing[@]}"; do
            echo "  - $var"
        done
        exit 1
    fi

    log_info "Environment configuration OK"
}

cmd_setup() {
    log_info "Setting up CIRISProxy..."

    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker not installed. Install with:"
        echo "  curl -fsSL https://get.docker.com | sh"
        echo "  sudo usermod -aG docker \$USER"
        exit 1
    fi

    # Check docker compose
    if ! docker compose version &> /dev/null; then
        log_error "Docker Compose not available"
        exit 1
    fi

    # Create .env if needed
    if [[ ! -f "$ENV_FILE" ]]; then
        log_info "Creating .env from template..."
        cp "$SCRIPT_DIR/.env.production" "$ENV_FILE"
        log_warn "Edit .env with your actual values before starting:"
        log_info "  nano $ENV_FILE"
        exit 0
    fi

    check_env

    # Pull images
    log_info "Pulling Docker images..."
    docker compose -f "$COMPOSE_FILE" pull

    log_info "Setup complete! Run './deploy.sh start' to launch"
}

cmd_start() {
    check_env
    log_info "Starting CIRISProxy..."
    docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" up -d

    log_info "Waiting for health check..."
    sleep 5

    if docker compose -f "$COMPOSE_FILE" ps | grep -q "healthy"; then
        log_info "CIRISProxy is running and healthy!"
    else
        log_warn "Services started but may still be initializing. Check with './deploy.sh status'"
    fi
}

cmd_stop() {
    log_info "Stopping CIRISProxy..."
    docker compose -f "$COMPOSE_FILE" down
    log_info "Stopped"
}

cmd_restart() {
    log_info "Restarting CIRISProxy..."
    docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" restart
    log_info "Restarted"
}

cmd_logs() {
    local service="${1:-}"
    if [[ -n "$service" ]]; then
        docker compose -f "$COMPOSE_FILE" logs -f "$service"
    else
        docker compose -f "$COMPOSE_FILE" logs -f
    fi
}

cmd_status() {
    echo "=== Container Status ==="
    docker compose -f "$COMPOSE_FILE" ps
    echo ""
    echo "=== Health Check ==="
    if curl -sf https://llm.ciris.ai/health > /dev/null 2>&1; then
        log_info "HTTPS endpoint: HEALTHY (https://llm.ciris.ai)"
    elif curl -sf http://localhost:4000/health > /dev/null 2>&1; then
        log_info "LiteLLM proxy: HEALTHY (localhost:4000)"
        log_warn "HTTPS may not be configured yet"
    else
        log_warn "LiteLLM proxy: NOT RESPONDING (may still be starting)"
    fi
}

cmd_init_tls() {
    log_info "Initializing TLS certificates..."
    if [[ -x "$SCRIPT_DIR/scripts/init-letsencrypt.sh" ]]; then
        "$SCRIPT_DIR/scripts/init-letsencrypt.sh"
    else
        log_error "scripts/init-letsencrypt.sh not found or not executable"
        exit 1
    fi
}

cmd_update() {
    log_info "Updating CIRISProxy..."

    # Pull latest code
    if [[ -d "$SCRIPT_DIR/.git" ]]; then
        log_info "Pulling latest code..."
        git -C "$SCRIPT_DIR" pull
    fi

    # Pull latest images
    log_info "Pulling latest Docker images..."
    docker compose -f "$COMPOSE_FILE" pull

    # Restart with new images
    log_info "Restarting with new images..."
    docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" up -d

    log_info "Update complete!"
}

cmd_help() {
    echo "CIRISProxy Deployment Script"
    echo ""
    echo "Usage: $0 <command>"
    echo ""
    echo "Commands:"
    echo "  setup    - Initial setup (install deps, create config)"
    echo "  init-tls - Initialize Let's Encrypt TLS certificates"
    echo "  start    - Start the proxy"
    echo "  stop     - Stop the proxy"
    echo "  restart  - Restart the proxy"
    echo "  logs     - View logs (optional: logs litellm|nginx|certbot)"
    echo "  status   - Check service status"
    echo "  update   - Pull latest code and images, restart"
    echo "  help     - Show this help"
}

# Main
case "${1:-help}" in
    setup)    cmd_setup ;;
    init-tls) cmd_init_tls ;;
    start)    cmd_start ;;
    stop)     cmd_stop ;;
    restart)  cmd_restart ;;
    logs)     cmd_logs "${2:-}" ;;
    status)   cmd_status ;;
    update)   cmd_update ;;
    help)     cmd_help ;;
    *)
        log_error "Unknown command: $1"
        cmd_help
        exit 1
        ;;
esac
