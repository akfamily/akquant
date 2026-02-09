# Installation Guide

## Prerequisites

Before installing `AKQuant`, please ensure your system meets the following requirements:

*   **Python**: Version >= 3.10.
*   **Rust**: If compiling from source, install the latest stable version of Rust. [Install Rust](https://www.rust-lang.org/tools/install)
*   **OS**: macOS, Linux, Windows.

## Installation Methods

### 1. Install from PyPI (Recommended)

Currently, `AKQuant` is not yet officially published to PyPI. Once published, you can install it via:

```bash
pip install akquant
```

### 2. Install from Source (Development Mode)

If you wish to contribute to development or use the latest features, you can install from source.

First, clone the repository:

```bash
git clone https://github.com/yourusername/akquant.git
cd akquant
```

Create and activate a virtual environment (optional but recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows
```

Install the build tool `maturin`:

```bash
pip install maturin
```

Build and install into the current environment:

```bash
maturin develop
```

Or build a release version (for optimized performance):

```bash
maturin develop --release
```

## Verify Installation

After installation, you can import `akquant` in Python to verify:

```python
import akquant
print(akquant.__version__)
```

If no errors occur, the installation was successful.
