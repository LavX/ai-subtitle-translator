#!/bin/bash
# AI Subtitle Translator - Quick Install
# Run alongside your existing Bazarr Docker container
#
# Usage: curl -sSL https://raw.githubusercontent.com/LavX/ai-subtitle-translator/main/install.sh | bash

set -e

CONTAINER_NAME="ai-subtitle-translator"
IMAGE="ghcr.io/lavx/ai-subtitle-translator:latest"
PORT=8765
VOLUME="${CONTAINER_NAME}-data"

echo "=== AI Subtitle Translator Installer ==="
echo ""

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker is not installed. Install Docker first."
    exit 1
fi

# Check if already running
if docker ps -q --filter "name=^${CONTAINER_NAME}$" | grep -q .; then
    echo "Container '${CONTAINER_NAME}' is already running."
    echo "Encryption key: $(docker exec ${CONTAINER_NAME} cat /app/data/encryption.key)"
    echo ""
    echo "To reinstall: docker stop ${CONTAINER_NAME} && docker rm ${CONTAINER_NAME} && run this script again"
    exit 0
fi

# Remove stopped container if exists
if docker ps -aq --filter "name=^${CONTAINER_NAME}$" | grep -q .; then
    echo "Removing stopped container..."
    docker rm ${CONTAINER_NAME} > /dev/null
fi

# Detect Bazarr container and its network
BAZARR_NETWORK=""
BAZARR_CONTAINER=""
for name in bazarr bazarr-ui-test; do
    if docker ps -q --filter "name=^${name}$" | grep -q .; then
        BAZARR_CONTAINER="$name"
        BAZARR_NETWORK=$(docker inspect "$name" --format '{{.HostConfig.NetworkMode}}' 2>/dev/null)
        break
    fi
done

# Build network args
NETWORK_ARGS=""
URL_HINT=""
if [ -n "$BAZARR_CONTAINER" ]; then
    echo "Found Bazarr container: ${BAZARR_CONTAINER} (network: ${BAZARR_NETWORK})"
    if [ "$BAZARR_NETWORK" = "host" ]; then
        NETWORK_ARGS="--network host"
        URL_HINT="http://localhost:${PORT}"
    elif [ "$BAZARR_NETWORK" != "default" ] && [ "$BAZARR_NETWORK" != "bridge" ]; then
        NETWORK_ARGS="--network ${BAZARR_NETWORK}"
        URL_HINT="http://${CONTAINER_NAME}:${PORT}"
    else
        NETWORK_ARGS="-p ${PORT}:${PORT}"
        URL_HINT="http://localhost:${PORT}"
    fi
else
    echo "No Bazarr container found. Using port mapping."
    NETWORK_ARGS="-p ${PORT}:${PORT}"
    URL_HINT="http://localhost:${PORT}"
fi

# Pull and run
echo "Pulling latest image..."
docker pull ${IMAGE} > /dev/null

echo "Starting ${CONTAINER_NAME}..."
docker run -d \
    --name ${CONTAINER_NAME} \
    --restart unless-stopped \
    ${NETWORK_ARGS} \
    -v ${VOLUME}:/app/data \
    ${IMAGE} > /dev/null

# Wait for startup
echo "Waiting for service to start..."
for i in $(seq 1 10); do
    if docker exec ${CONTAINER_NAME} test -f /app/data/encryption.key 2>/dev/null; then
        break
    fi
    sleep 1
done

# Show results
echo ""
echo "=== Setup Complete ==="
echo ""
echo "Service URL:     ${URL_HINT}"
echo "Encryption Key:  $(docker exec ${CONTAINER_NAME} cat /app/data/encryption.key)"
echo ""
echo "Next steps:"
echo "  1. Copy the encryption key above"
echo "  2. Open Bazarr Settings > AI Subtitle Translator"
echo "  3. Set URL to: ${URL_HINT}"
echo "  4. Paste the encryption key"
echo "  5. Add your OpenRouter API key from https://openrouter.ai/keys"
echo "  6. Click Test, then Save"
echo ""
echo "Logs:  docker logs -f ${CONTAINER_NAME}"
echo "Docs:  ${URL_HINT}/docs"
