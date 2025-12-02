#!/bin/bash

# Diagnostic script for Opticam Docker environment

# Load environment variables from .env file
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | grep -v '^$' | xargs)
else
    echo "⚠️  Warning: .env file not found"
    JUPYTER_PORT=8888
    CONTAINER_NAME=opticam-jupyter
fi

# Set defaults
JUPYTER_PORT=${JUPYTER_PORT:-8888}
CONTAINER_NAME=${CONTAINER_NAME:-opticam-jupyter}

echo "================================================"
echo "Opticam Docker Diagnostics"
echo "================================================"
echo ""
echo "Configuration:"
echo "  Port:      ${JUPYTER_PORT}"
echo "  Container: ${CONTAINER_NAME}"
echo ""

echo "1. Checking Docker status..."
if docker info > /dev/null 2>&1; then
    echo "   ✓ Docker is running"
else
    echo "   ❌ Docker is not running!"
    exit 1
fi
echo ""

echo "2. Checking if container exists..."
if docker ps -a --filter "name=${CONTAINER_NAME}" --format "{{.Names}}" | grep -q "${CONTAINER_NAME}"; then
    echo "   ✓ Container exists"
    
    if docker ps --filter "name=${CONTAINER_NAME}" --format "{{.Names}}" | grep -q "${CONTAINER_NAME}"; then
        echo "   ✓ Container is running"
    else
        echo "   ⚠️  Container exists but is not running"
        echo ""
        echo "   Last exit status:"
        docker ps -a --filter "name=${CONTAINER_NAME}" --format "table {{.Status}}"
    fi
else
    echo "   ⚠️  Container does not exist (not built yet)"
fi
echo ""

echo "3. Checking port ${JUPYTER_PORT}..."
if lsof -Pi :${JUPYTER_PORT} -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "   ⚠️  Port ${JUPYTER_PORT} is in use by:"
    lsof -i :${JUPYTER_PORT} | grep LISTEN
else
    echo "   ✓ Port ${JUPYTER_PORT} is available"
fi
echo ""

echo "4. Checking Docker network..."
if docker network ls | grep -q "opticam_default"; then
    echo "   ✓ Docker network exists"
else
    echo "   ⚠️  Docker network not created yet"
fi
echo ""

echo "5. Checking required directories..."
for dir in "data" "notebooks" "output"; do
    if [ -d "$dir" ]; then
        echo "   ✓ $dir/ exists"
    else
        echo "   ⚠️  $dir/ does not exist (will be created on startup)"
    fi
done
echo ""

if docker ps --filter "name=${CONTAINER_NAME}" --format "{{.Names}}" | grep -q "${CONTAINER_NAME}"; then
    echo "6. Checking container port mapping..."
    docker port ${CONTAINER_NAME} 2>/dev/null || echo "   ⚠️  No port mapping found"
    echo ""
    
    echo "7. Checking container volumes..."
    docker inspect ${CONTAINER_NAME} --format='{{range .Mounts}}{{.Source}} -> {{.Destination}}{{println}}{{end}}' 2>/dev/null
    echo ""
    
    echo "8. Recent container logs (last 20 lines)..."
    echo "   ----------------------------------------"
    docker logs --tail 20 ${CONTAINER_NAME} 2>&1
    echo "   ----------------------------------------"
    echo ""
    
    echo "9. Testing connection to Jupyter..."
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:${JUPYTER_PORT} 2>/dev/null || echo "000")
    
    if [ "$HTTP_CODE" = "200" ] || [ "$HTTP_CODE" = "302" ]; then
        echo "   ✓ Jupyter is responding on localhost:${JUPYTER_PORT}"
        echo ""
        echo "   Access Jupyter at:"
        echo "   http://localhost:${JUPYTER_PORT}"
    else
        echo "   ❌ Jupyter is not responding on localhost:${JUPYTER_PORT} (HTTP ${HTTP_CODE})"
        echo ""
        echo "   Possible issues:"
        echo "   - Jupyter hasn't finished starting (check logs above)"
        echo "   - Jupyter is configured incorrectly"
        echo "   - Network configuration issue"
        echo "   - Port mapping is incorrect"
    fi
    
    echo ""
    echo "10. Checking opticam installation..."
    docker exec ${CONTAINER_NAME} python -c "import opticam; print(f'Opticam version: {opticam.__version__}')" 2>/dev/null || echo "   ⚠️  Could not verify opticam installation"
else
    echo "6. Container is not running."
    echo ""
    echo "   To start the container, run:"
    echo "   ./start.sh"
    echo ""
    echo "   Or manually:"
    echo "   docker-compose up"
fi

echo ""
echo "================================================"
echo "Diagnostic complete"
echo "================================================"
echo ""
echo "Need help?"
echo "  - Check DOCKER.md for detailed documentation"
echo "  - Review .env configuration"
echo "  - Check container logs: docker-compose logs"
