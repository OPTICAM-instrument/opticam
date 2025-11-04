# opticam
A Python package for reducing OPTICAM data.

## Features
- Customisable. Many of `opticam`'s reduction methods are fully customisable, allowing for full control over the reduction process.
- Informative. `opticam` includes informative logging, allowing for reproducable reduction and information about any errors or warnings.
- Robust. `opticam` is designed to catch many common errors and inform the user how they can be resolved.
- Scalable. `opticam` can leverage modern multi-core CPUs to drastically speed up reduction.
- Simple. When using `opticam`'s reduction methods, the default values should "just work" most of the time. Faint sources and/or crowded fields may require some tailoring, however.

## Installation

### From GitHub (recommended)

You can install the latest stable release of `opticam` directly from GitHub using:

```
pip install git+https://github.com/zairving/opticam.git
```

### Locally

If you have a local copy of `opticam`, it can be `pip` installed by navigating to the directory and running:

```
pip install .
```

### Requirements

All of `opticam`'s dependencies are available as Python packages via your preferred package manager, and should be handled automatically via `pip`. If you would prefer to install the dependencies via `conda`, use provided [YAML file](environment.yml) to set up your environment, and then install `opticam` via `pip` as described above.

## Getting Started

Documentation for `opticam` is available on [Read the Docs](https://opticam.readthedocs.io/en/latest/index.html). To get started, I recommend checking out the [reduction tutorial](https://opticam.readthedocs.io/en/latest/tutorials/reduction.html).