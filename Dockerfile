FROM python:3.12-slim

# Set working directory
WORKDIR /workspace

# Install system dependencies needed for scientific packages
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    pkg-config \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy the entire opticam package
COPY . /workspace/opticam

# Install opticam and its dependencies
WORKDIR /workspace/opticam
RUN pip install --no-cache-dir -e ".[docs]"

# Create directories for data and notebooks
RUN mkdir -p /workspace/data \
    /workspace/notebooks \
    /workspace/output

# Set working directory back to workspace root
WORKDIR /workspace

# Expose Jupyter port
EXPOSE 8888

# Configure Jupyter Lab
CMD ["jupyter-lab", \
     "--ip=0.0.0.0", \
     "--port=8888", \
     "--no-browser", \
     "--allow-root", \
     "--ServerApp.token=''", \
     "--ServerApp.password=''", \
     "--ServerApp.allow_origin='*'", \
     "--ServerApp.base_url='/'", \
     "--notebook-dir=/workspace"]
