# Python 环境搭建指南 (Environment Setup Guide)

工欲善其事，必先利其器。在开始量化交易之前，你需要一个干净、稳定且独立的 Python 运行环境。
本指南将提供两种主流的环境管理方案：**Miniconda (经典稳定)** 和 **uv (极速现代)**。

---

## 方案一：Miniconda (经典推荐)

Miniconda 是 Anaconda 的精简版，是数据科学领域的行业标准。它能帮你轻松管理 Python 版本和各种依赖包。

### 1. 安装 Miniconda

> **提示**：国内用户访问官网可能较慢，推荐使用 [清华大学开源软件镜像站](https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/) 下载安装包。

=== "Windows"

    1.  访问 [Miniconda 官网](https://docs.conda.io/en/latest/miniconda.html) 或 [清华镜像站](https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/)。
    2.  下载 **Windows 64-bit** 安装包 (`.exe`)。
    3.  运行安装程序，建议勾选 "Add Miniconda3 to my PATH environment variable" (虽然提示不推荐，但对新手更方便)。
    4.  安装完成后，打开 **Command Prompt (CMD)** 或 **PowerShell**。

=== "macOS"

    **方法 A：使用 Homebrew (推荐)**
    打开终端 (Terminal)，运行：
    ```bash
    brew install --cask miniconda
    init conda
    ```

    **方法 B：使用安装脚本**
    1.  下载 [macOS 安装脚本](https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh) (M1/M2/M3) 或 [Intel 版](https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh)。
        *   或者从 [清华镜像](https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/) 下载对应的 `.sh` 文件。
    2.  在终端运行：`bash Miniconda3-latest-MacOSX-arm64.sh`。

=== "Linux"

    打开终端，运行以下命令：
    ```bash
    mkdir -p ~/miniconda3

    # 方式一：官方源
    # wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh

    # 方式二：清华源 (推荐国内用户)
    wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh

    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
    rm -rf ~/miniconda3/miniconda.sh
    ~/miniconda3/bin/conda init bash
    ```

### 1.5 配置国内镜像源 (推荐)

国内用户访问官方源可能较慢，建议配置清华大学镜像源以加速下载。

```bash
# 配置 Conda 镜像源
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --set show_channel_urls yes

# 配置 Pip 镜像源 (永久生效)
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

### 2. 创建虚拟环境

不要直接在系统 Python 中安装库！我们需要创建一个专属的“沙盒”。

打开终端（Windows 用户打开 CMD 或 Anaconda Prompt），输入：

```bash
# 创建一个名为 quant_dev 的环境，指定 Python 版本为 3.10，AKQuant 支持 Python 3.10及以上版本
conda create -n quant_dev python=3.10 -y

# 激活环境
conda activate quant_dev
```

激活成功后，你的命令行前缀会变成 `(quant_dev)`，说明你已经进入了沙盒。

---

## 方案二：uv (极速现代)

如果你追求极致的速度和轻量化，[uv](https://github.com/astral-sh/uv) 是目前 Python 生态中最快的包管理器（由 Rust 编写）。它可以替代 `pip` 和 `virtualenv`。

### 1. 安装 uv

=== "Windows"

    在 PowerShell 中运行：
    ```powershell
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```

=== "macOS / Linux"

    在终端运行：
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

### 2. 创建并管理环境

uv 不需要预先安装 Python，它会自动帮你下载。

```bash
# 1. 创建一个新的项目目录
mkdir my_strategy
cd my_strategy

# 2. 初始化虚拟环境 (指定 Python 3.10)
# uv 会自动下载 Python 3.10 并创建 .venv 目录
uv venv --python 3.10

# 3. 激活环境
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate
```

---

## 3. 安装 AKQuant 并验证

无论你使用了 Miniconda 还是 uv，现在你都应该处于一个激活的虚拟环境中。接下来我们安装交易框架。

### 安装

**如果你使用 Miniconda:**
```bash
# 如果之前未配置镜像源，可添加 -i 参数临时加速
pip install akquant -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**如果你使用 uv:**
```bash
# 使用清华源加速
uv pip install akquant --index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

### 验证

创建一个测试脚本 `verify.py`：

```python
import akquant
import pandas as pd

print(f"AKQuant Version: {akquant.__version__}")
print(f"Pandas Version: {pd.__version__}")
print("环境搭建成功！可以开始写策略了。")
```

运行它：

```bash
python verify.py
```

如果看到“环境搭建成功”，恭喜你，你的兵器库已经准备完毕！
下一步，请前往 [Python 金融入门](py_guide.md) 学习基础语法，或直接查看 [量化新手指南](quant_guide.md) 开始实战。
