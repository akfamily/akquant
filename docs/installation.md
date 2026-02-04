# 安装指南

## 前置要求

在安装 `AKQuant` 之前，请确保您的系统满足以下要求：

*   **Python**: 版本 >= 3.10。
*   **Rust**: 如果需要从源码编译，请安装最新稳定版 Rust。[安装 Rust](https://www.rust-lang.org/tools/install)
*   **操作系统**: 支持 macOS, Linux, Windows。

## 安装方式

### 1. 从 PyPI 安装 (推荐)

目前 `AKQuant` 尚未正式发布到 PyPI。发布后，您将可以通过以下命令安装：

```bash
pip install akquant
```

### 2. 从源码编译安装 (开发模式)

如果您希望参与开发或使用最新功能，可以从源码进行安装。

首先，克隆代码仓库：

```bash
git clone https://github.com/yourusername/akquant.git
cd akquant
```

创建并激活虚拟环境（可选但推荐）：

```bash
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows
```

安装构建工具 `maturin`：

```bash
pip install maturin
```

构建并安装到当前环境：

```bash
maturin develop
```

或者构建 release 版本（优化性能）：

```bash
maturin develop --release
```

## 验证安装

安装完成后，可以在 Python 中导入 `akquant` 进行验证：

```python
import akquant
print(akquant.__version__)
```

如果没有报错，说明安装成功。
