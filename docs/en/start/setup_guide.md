# Environment Setup Guide

Before starting quantitative trading, you need a clean, stable, and isolated Python environment.
This guide provides two mainstream solutions: **Miniconda (Classic & Stable)** and **uv (Fast & Modern)**.

---

## Option A: Miniconda (Classic)

Miniconda is a minimal installer for conda. It is the industry standard for data science, making it easy to manage Python versions and dependencies.

### 1. Install Miniconda

> **Tip**: Users in China can download from [Tsinghua Mirror](https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/) for faster speeds.

=== "Windows"

    1.  Visit [Miniconda Website](https://docs.conda.io/en/latest/miniconda.html) (or [Tsinghua Mirror](https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/)).
    2.  Download the **Windows 64-bit** installer (`.exe`).
    3.  Run the installer. It is recommended to check "Add Miniconda3 to my PATH environment variable" (convenient for beginners).
    4.  After installation, open **Command Prompt (CMD)** or **PowerShell**.

=== "macOS"

    **Method A: Via Homebrew (Recommended)**
    Open Terminal and run:
    ```bash
    brew install --cask miniconda
    init conda
    ```

    **Method B: Via Installer Script**
    1.  Download script for [Apple Silicon](https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh) or [Intel](https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh).
    2.  Run in terminal: `bash Miniconda3-latest-MacOSX-arm64.sh`.

=== "Linux"

    Open terminal and run:
    ```bash
    mkdir -p ~/miniconda3
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
    rm -rf ~/miniconda3/miniconda.sh
    ~/miniconda3/bin/conda init bash
    ```

### 1.5 Configure Mirrors (For Users in China)

If you are located in China, download speeds might be slow. It is recommended to use the Tsinghua University mirror.

```bash
# Configure Conda Mirror
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --set show_channel_urls yes

# Configure Pip Mirror
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

### 2. Create Virtual Environment

Do not install libraries directly into your system Python! We need a dedicated "sandbox".

Open your terminal (or CMD/Anaconda Prompt on Windows) and type:

```bash
# Create an environment named 'akquant' with Python 3.10
conda create -n akquant python=3.10 -y

# Activate the environment
conda activate akquant
```

Once activated, your command prompt prefix will change to `(akquant)`, indicating you are inside the sandbox.

---

## Option B: uv (Fast & Modern)

If you want extreme speed and lightweight management, [uv](https://github.com/astral-sh/uv) is the fastest package manager in the Python ecosystem (written in Rust). It replaces `pip` and `virtualenv`.

### 1. Install uv

=== "Windows"

    Run in PowerShell:
    ```powershell
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```

=== "macOS / Linux"

    Run in Terminal:
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

### 2. Create & Manage Environment

uv does not require pre-installed Python; it manages Python versions for you.

```bash
# 1. Create a project directory
mkdir my_strategy
cd my_strategy

# 2. Initialize virtual env (Specify Python 3.10)
# uv will automatically download Python 3.10 and create .venv folder
uv venv --python 3.10

# 3. Activate environment
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate
```

---

## 3. Install AKQuant & Verify

Whether you used Miniconda or uv, you should now be in an activated virtual environment.

### Install

**If using Miniconda:**
```bash
pip install akquant

# Users in China can use the Tsinghua mirror for speed:
# pip install akquant -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**If using uv:**
```bash
uv pip install akquant

# Users in China:
# uv pip install akquant --index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

### Verify

Create a test script `verify.py`:

```python
import akquant
import pandas as pd

print(f"AKQuant Version: {akquant.__version__}")
print(f"Pandas Version: {pd.__version__}")
print("Environment setup successful! Ready to trade.")
```

Run it:

```bash
python verify.py
```

If you see the success message, your arsenal is ready!
Next, go to [Python for Finance](py_guide.md) to learn syntax, or jump to [Quant Guide](quant_guide.md) to start coding strategies.
