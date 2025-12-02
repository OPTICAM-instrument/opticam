# Opticam Docker Setup

This document describes how to run Opticam in a Docker container with Jupyter Lab for data reduction and analysis.

## Quick Reference

| Setting | Default | Description |
|---------|---------|-------------|
| Port | 8888 | Access Jupyter at `http://localhost:8888` |
| Container | opticam-jupyter | Docker container name |
| Auth | Disabled | No password/token (testing only!) |
| Data Dir | ./data | Your FITS files and observations |
| Notebooks Dir | ./notebooks | Your Jupyter notebooks |
| Output Dir | ./output | Reduction results and plots |

**All settings are configurable via `.env` file**

## Prerequisites

- Docker Desktop installed and running
- At least 4GB of available RAM
- Port 8888 available (or configure a different port)

## Quick Start

### 1. Create Configuration

```bash
cd /path/to/your/software/opticam

# Copy the example environment file
cp .env.example .env

# Edit if needed (optional)
nano .env
```

### 2. Start Jupyter Lab

```bash
# Make scripts executable
chmod +x start.sh diagnose.sh

# Start the environment
./start.sh
```

The script will:
- Check your configuration
- Create necessary directories
- Build the Docker image (first time only, ~5-10 minutes)
- Start Jupyter Lab

### 3. Access Jupyter Lab

Open your browser to: **http://localhost:8888**

You'll see:
- `/workspace/data` - Your observation data (empty initially)
- `/workspace/notebooks` - Your notebooks
- `/workspace/output` - Reduction results
- `/workspace/tutorials` - Opticam tutorial notebooks
- `/workspace/opticam` - Opticam source code

## Directory Structure

```
opticam/
├── data/              # Your FITS files go here
├── notebooks/         # Your work notebooks
├── output/            # Reduction results
├── .env               # Your configuration (not in git)
├── .env.example       # Configuration template
├── Dockerfile         # Docker image definition
├── docker-compose.yml # Container orchestration
├── start.sh           # Startup helper script
└── diagnose.sh        # Diagnostic helper script
```

## Configuration

### Basic Configuration

Edit `.env` to customize your setup:

```bash
# Jupyter Lab Configuration
JUPYTER_PORT=8888              # Change if port is in use
CONTAINER_NAME=opticam-jupyter # Container name

# Data Directories (can be absolute or relative paths)
DATA_DIR=./data                # Your FITS files
NOTEBOOKS_DIR=./notebooks      # Your notebooks
OUTPUT_DIR=./output            # Results and plots

# Authentication (leave empty to disable)
JUPYTER_TOKEN=                 # Set for password protection
JUPYTER_PASSWORD=              # Alternative to token
```

### Using Different Ports

If port 8888 is in use:

1. Edit `.env`:
   ```bash
   JUPYTER_PORT=8889
   ```

2. Restart:
   ```bash
   docker-compose down
   ./start.sh
   ```

3. Access at: `http://localhost:8889`

### Using Absolute Paths for Data

If your FITS files are elsewhere:

```bash
# .env
DATA_DIR=/path/to/your/observations
NOTEBOOKS_DIR=/path/to/your/notebooks
OUTPUT_DIR=/path/to/your/results
```

## Working with Opticam

### Running Tutorials

The Opticam tutorials are available in Jupyter Lab at `/workspace/tutorials/`:

1. Open Jupyter Lab: `http://localhost:8888`
2. Navigate to `tutorials/`
3. Open any tutorial notebook (e.g., `reduction.ipynb`)
4. Run the cells to learn Opticam features

### Importing Your Data

Copy your FITS files to the data directory:

```bash
# From your host machine
cp /path/to/observations/*.fits data/
```

In Jupyter, access them at `/workspace/data/`.

### Creating a New Reduction Notebook

1. In Jupyter Lab, navigate to `/workspace/notebooks/`
2. Create a new notebook
3. Import opticam:

```python
import opticam
from opticam.reducer import Reducer
from pathlib import Path

# Your data is at /workspace/data/
data_path = Path("/workspace/data")
output_path = Path("/workspace/output")

# Start your reduction
# ... (follow tutorial examples)
```

### Saving Results

All results saved to `/workspace/output/` in the container are automatically available in your `output/` directory on the host.

## Common Tasks

### Starting the Container

```bash
./start.sh
```

Or manually:
```bash
docker-compose up
```

### Stopping the Container

Press `Ctrl+C` in the terminal, or:

```bash
docker-compose down
```

### Restarting After Changes

```bash
docker-compose restart
```

### Rebuilding the Image

If you modify the Dockerfile or Opticam code:

```bash
docker-compose down
docker-compose build --no-cache
docker-compose up
```

### Accessing the Container Shell

To run commands inside the container:

```bash
docker exec -it opticam-jupyter /bin/bash
```

### Viewing Logs

```bash
# Follow logs in real-time
docker-compose logs -f

# Last 50 lines
docker-compose logs --tail 50
```

### Running Diagnostics

```bash
./diagnose.sh
```

This checks:
- Docker status
- Container status
- Port availability
- Directory structure
- Jupyter connectivity
- Opticam installation

## Troubleshooting

### Issue: Port Already in Use

**Symptoms**: Error message about port 8888 being in use

**Solution**:
1. Check what's using the port:
   ```bash
   lsof -i :8888
   ```

2. Either stop that process, or change the port in `.env`:
   ```bash
   JUPYTER_PORT=8889
   ```

3. Restart:
   ```bash
   docker-compose down
   ./start.sh
   ```

### Issue: Cannot Connect to Jupyter

**Symptoms**: Browser shows "connection refused" or similar

**Diagnostics**:
```bash
./diagnose.sh
```

**Solutions**:

1. Check if container is running:
   ```bash
   docker ps | grep opticam
   ```

2. Check logs for errors:
   ```bash
   docker-compose logs
   ```

3. Verify port mapping:
   ```bash
   docker port opticam-jupyter
   ```

4. Try accessing different URLs:
   - `http://localhost:8888`
   - `http://localhost:8888/lab`
   - `http://127.0.0.1:8888`

### Issue: "Module Not Found" Errors

**Symptoms**: ImportError when importing opticam or dependencies

**Solution**:

1. Verify opticam is installed:
   ```bash
   docker exec -it opticam-jupyter pip show opticam
   ```

2. Check if it's an editable install:
   ```bash
   docker exec -it opticam-jupyter pip list | grep opticam
   ```

3. Rebuild if needed:
   ```bash
   docker-compose down
   docker-compose build --no-cache
   docker-compose up
   ```

### Issue: Permission Denied on Files

**Symptoms**: Cannot write to data/notebooks/output directories

**Solution**:

The container runs as root, which usually avoids permission issues. If you encounter them:

```bash
# Make directories writable
chmod -R 755 data/ notebooks/ output/
```

### Issue: Out of Memory

**Symptoms**: Container exits unexpectedly, "killed" in logs

**Solution**:

1. Increase Docker memory in Docker Desktop settings
2. Recommended: At least 4GB for typical reductions
3. For large datasets: 8GB or more

### Issue: Build Takes Forever

**Symptoms**: Docker build hangs during pip install

**Solution**:

1. Check your internet connection
2. Try building with verbose output:
   ```bash
   docker-compose build --progress=plain
   ```

3. Clear Docker cache and rebuild:
   ```bash
   docker system prune -f
   docker-compose build --no-cache
   ```

### Issue: Changes to Opticam Code Not Reflected

**Symptoms**: Code changes don't appear in Jupyter

**Solution**:

Opticam is installed as editable (`pip install -e`), so changes should be reflected. But:

1. Restart the Jupyter kernel in your notebook
2. If that doesn't work, restart the container:
   ```bash
   docker-compose restart
   ```

3. For Dockerfile changes, rebuild:
   ```bash
   docker-compose down
   docker-compose build
   docker-compose up
   ```

## Advanced Usage

### Custom Jupyter Configuration

Create `jupyter_lab_config.py` in the opticam directory:

```python
# Custom Jupyter configuration
c.ServerApp.allow_remote_access = True
c.ServerApp.port = 8888
# Add more customizations...
```

Then modify the Dockerfile to copy and use it.

### Running in Detached Mode

To run in the background:

```bash
docker-compose up -d

# View logs
docker-compose logs -f

# Stop when done
docker-compose down
```

### Multiple Instances

To run multiple instances (e.g., for different projects):

1. Copy opticam directory:
   ```bash
   cp -r opticam opticam-project2
   ```

2. Edit `.env` in the new directory:
   ```bash
   JUPYTER_PORT=8889
   CONTAINER_NAME=opticam-project2
   ```

3. Start each independently:
   ```bash
   cd opticam && ./start.sh
   cd opticam-project2 && ./start.sh
   ```

### Using with Remote Docker Host

If running Docker on a remote machine:

1. Set `DOCKER_HOST` environment variable
2. Use SSH tunneling for Jupyter:
   ```bash
   ssh -L 8888:localhost:8888 user@remote-host
   ```

### Backing Up Your Work

Your notebooks and data are in host directories, so they're safe even if you delete the container:

```bash
# Backup
tar -czf opticam-backup-$(date +%Y%m%d).tar.gz data/ notebooks/ output/

# Restore
tar -xzf opticam-backup-YYYYMMDD.tar.gz
```

## Production Considerations

⚠️ **The default configuration is for development/testing only!**

For production use:

### 1. Enable Authentication

Edit `.env`:
```bash
JUPYTER_TOKEN=your-secure-random-token-here
```

Or set a password:
```bash
# Generate a hashed password
docker run --rm opticam-jupyter python -c "from jupyter_server.auth import passwd; print(passwd('your-password'))"

# Add to .env
JUPYTER_PASSWORD=sha1:...
```

### 2. Use HTTPS

Set up a reverse proxy (nginx, Caddy) with SSL/TLS termination.

### 3. Restrict Network Access

In `docker-compose.yml`:
```yaml
ports:
  - "127.0.0.1:8888:8888"  # Only localhost
```

### 4. Regular Updates

Keep the base image and dependencies updated:
```bash
docker-compose pull
docker-compose build --no-cache
```

## Getting Help

1. **Run diagnostics**: `./diagnose.sh`
2. **Check logs**: `docker-compose logs`
3. **Opticam documentation**: https://opticam.readthedocs.io
4. **Docker documentation**: https://docs.docker.com

## Useful Commands Reference

```bash
# Start
./start.sh                          # Recommended
docker-compose up                   # Manual start
docker-compose up -d                # Background mode

# Stop
docker-compose down                 # Stop and remove
docker-compose stop                 # Stop without removing

# Status
docker ps                          # Running containers
docker-compose ps                  # This project's containers
./diagnose.sh                      # Full diagnostics

# Logs
docker-compose logs                # All logs
docker-compose logs -f             # Follow logs
docker-compose logs --tail 50      # Last 50 lines

# Rebuild
docker-compose build               # Rebuild image
docker-compose build --no-cache    # Clean rebuild

# Shell Access
docker exec -it opticam-jupyter /bin/bash

# Cleanup
docker-compose down -v             # Remove volumes too
docker system prune                # Clean unused resources
```

## Summary

✅ **Benefits of Docker setup:**
- Consistent environment across machines
- No dependency conflicts with host system
- Easy to share and reproduce
- Isolated from host Python installations
- Quick setup for new users

📝 **Remember:**
- Configuration in `.env` file
- Data in `data/`, notebooks in `notebooks/`, output in `output/`
- Tutorials available at `/workspace/tutorials/`
- Run `./diagnose.sh` if you encounter issues
