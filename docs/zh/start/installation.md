# 安装指南

## 前置要求

在安装 `AKQuant` 之前，请确保您的系统满足以下要求：

*   **Python**: 版本 >= 3.10。
*   **Rust**: 如果需要从源码编译，请安装最新稳定版 Rust。[安装 Rust](https://www.rust-lang.org/tools/install)
*   **操作系统**: 支持 macOS, Linux, Windows。

## 安装方式

### 1. 从 PyPI 安装 (推荐)

`AKQuant` 已发布到 PyPI，您可以直接通过 `pip` 安装预编译的 Wheel 包（无需 Rust 环境）：

```bash
pip install akquant
```

按需安装可选功能，多个可以组合，如 `pip install "akquant[mcp,ml]"`：

| 可选功能 | 内容 |
|---|---|
| `mcp` | MCP 服务（含 MCP SDK 与 AKShare），让 AI 助手查 API、写策略、跑回测，见 [MCP 服务](../guide/mcp.md) |
| `ml` | 机器学习：scikit-learn、PyTorch |
| `quantstats` | QuantStats 绩效报告 |
| `signal-redis` | 外部信号接入的 Redis Stream 信号源 |
| `full` | `ml` 与 `quantstats` 的全部内容（不含 `mcp`） |

只想在 AI 助手中使用 AKQuant 时，可以不装：按 [MCP 服务](../guide/mcp.md#clients) 的说明用 `uvx` 接入，客户端会自动下载并启动。

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
