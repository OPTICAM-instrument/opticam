# phoptic
A Python package for reducing astronomical images.

## Features
- Customisable. Many of `phoptic`'s reduction methods are fully customisable, allowing for full control over the reduction process.
- Informative. `phoptic` includes informative logging, allowing for reproducable reduction and information about any errors or warnings.
- Robust. `phoptic` is designed to catch many common errors and inform the user how they can be resolved.
- Scalable. `phoptic` can leverage modern multi-core CPUs to drastically speed up reduction.
- Simple. When using `phoptic`'s reduction methods, the default values should "just work" most of the time. Faint sources and/or crowded fields may require some tailoring, however.

## Installation

### Option 1: Docker (Recommended for Avoiding Compatibility Issues)

The safest way to get started with `phoptic` is using Docker, which provides a complete environment with Jupyter Lab. This has the benefit of "containerizing" the installation, avoiding any conflicts or compatibility issues:

```bash
# Copy the environment configuration
cp .env.example .env

# Start Jupyter Lab
chmod +x start.sh
./start.sh
```

Then access Jupyter Lab at `http://localhost:8888`.

For a more complete Docker setup guide, see [DOCKER.md](DOCKER.md).

### Option 2: From GitHub

You can install the latest stable release of `phoptic` directly from GitHub using:

```
pip install git+https://github.com/OPTICAM-instrument/opticam.git
```

### Option 3: Locally

If you have a local copy of `phoptic`, it can be `pip` installed by navigating to the directory and running:

```
pip install .
```

### Requirements

All of `phoptic`'s dependencies are available as Python packages via your preferred package manager, and should be handled automatically via `pip`. If you would prefer to install the dependencies via `conda`, use provided [YAML file](environment.yml) to set up your environment, and then install `opticam` via `pip` as described above.

## Getting Started

Documentation for `phoptic` is available on [Read the Docs](https://phoptic.readthedocs.io/en/latest/index.html). To get started, I recommend checking out the [reduction tutorial](https://phoptic.readthedocs.io/en/latest/tutorials/reduction.html).