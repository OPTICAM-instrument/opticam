#!/bin/bash

# Opticam Docker Startup Script

set -e

# Load environment variables from .env file
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | grep -v '^$' | xargs)
else
    echo "⚠️  Warning: .env file not found"
    echo ""
    echo "Creating .env from .env.example..."
    if [ -f .env.example ]; then
        cp .env.example .env
        echo "✓ Created .env file"
        echo ""
        echo "Please review and edit .env if needed, then run this script again."
        exit 0
    else
        echo "❌ Error: .env.example not found!"
        exit 1
    fi
fi

# Set defaults if not in .env
JUPYTER_PORT=${JUPYTER_PORT:-8888}
CONTAINER_NAME=${CONTAINER_NAME:-opticam-jupyter}
DATA_DIR=${DATA_DIR:-./data}
NOTEBOOKS_DIR=${NOTEBOOKS_DIR:-./notebooks}
OUTPUT_DIR=${OUTPUT_DIR:-./output}

echo "================================================"
echo "Opticam Docker Environment"
echo "================================================"
echo ""
echo "Configuration:"
echo "  Port:       ${JUPYTER_PORT}"
echo "  Container:  ${CONTAINER_NAME}"
echo "  Data:       ${DATA_DIR}"
echo "  Notebooks:  ${NOTEBOOKS_DIR}"
echo "  Output:     ${OUTPUT_DIR}"
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Error: Docker is not running!"
    echo "Please start Docker Desktop and try again."
    exit 1
fi

echo "✓ Docker is running"
echo ""

# Create directories if they don't exist
for dir in "$DATA_DIR" "$NOTEBOOKS_DIR" "$OUTPUT_DIR"; do
    if [ ! -d "$dir" ]; then
        echo "Creating directory: $dir"
        mkdir -p "$dir"
    fi
done

echo "✓ All required directories exist"
echo ""

# Check if port is available
if lsof -Pi :${JUPYTER_PORT} -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "⚠️  Warning: Port ${JUPYTER_PORT} is already in use!"
    echo ""
    echo "Current process using port ${JUPYTER_PORT}:"
    lsof -i :${JUPYTER_PORT}
    echo ""
    echo "To use a different port:"
    echo "  1. Edit .env file"
    echo "  2. Change JUPYTER_PORT to an available port"
    echo "  3. Run this script again"
    echo ""
    read -p "Do you want to continue anyway? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "✓ Port ${JUPYTER_PORT} is available"
fi

echo ""
echo "Building Docker image..."
echo "This may take several minutes on first run..."
echo ""

docker compose build

echo ""
echo "================================================"
echo "Starting Opticam Jupyter Lab..."
echo "================================================"
echo ""
echo "When you see 'Jupyter Server is running', access:"
echo ""
echo "    http://localhost:${JUPYTER_PORT}"
echo ""
echo "Available directories in Jupyter:"
echo "  /workspace/data       - Your observation data"
echo "  /workspace/notebooks  - Your notebooks"
echo "  /workspace/output     - Reduction results"
echo "  /workspace/tutorials  - Opticam tutorials"
echo "  /workspace/opticam    - Opticam source code"
echo ""
echo "Press Ctrl+C to stop the server."
echo ""
echo "================================================"
echo ""

docker compose up
